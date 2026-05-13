"""
나무 큐브 바둑판 정렬 — RealSense + 학습된 best.pt + 두산 e0509 + RH-P12-RN-A.

흐름:
  1) detect_all_cubes(): 모든 cube 의 base 좌표 (mm) 리스트
  2) 그리드 셀 4x3 (50mm 간격) 미리 계산
  3) 각 cube 에 대해:
     - 이웃과 가장 먼 방향으로 yaw 계산 (인접 cube 회피)
     - approach(top-down) → 그리퍼 40mm pre-open → descend(z=bz)
     - close(700) → lift(+100mm)
     - 그리드 셀로 이동 (yaw=0)
     - descend → 토크 OFF (= 바닥에 천천히 놓기) → lift
  4) 모든 cube 배치 완료 후 HOME

사용:
  python3 15_바둑판_정렬.py            # 실 동작
  python3 15_바둑판_정렬.py --dry-run  # 모션 없이 콘솔 출력만
  python3 15_바둑판_정렬.py --limit N  # 처음 N개만

12번을 importlib 로 재사용 (PickAndPlace, helpers). VisionDetector 는 best.pt 와
호환되도록 별도 작성 (set_classes 회피, 다중 검출 추가).
"""
import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from ultralytics import YOLO

# 12번 모듈 로드 (한글 파일명 + 숫자 시작 → importlib.util 로 강제 load)
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
_spec = importlib.util.spec_from_file_location(
    'p12', os.path.join(_DIR, '12_비전_피크앤플레이스.py'))
p12 = importlib.util.module_from_spec(_spec)
sys.modules['p12'] = p12
_spec.loader.exec_module(p12)


# ===== 설정 =====
# seg_v3: SAM2 점 prompt + 정사각형 강제 라벨로 학습. cube **상단면**만 정확히 mask.
# (seg_v2 는 SAM2 bbox prompt 라 cube 측면까지 포함 → 폐기)
YOLO_WEIGHTS = ('/home/fastcampus/Downloads/test/로봇강의_예제/'
                '02_Doosan_Robot_제어/yolo_dataset/runs/seg_v3/weights/best.pt')
DETECT_CONF_THR = 0.40
NMS_IOU_THR = 0.45
# 학습 데이터 (capture_dataset.py) 와 동일한 1280x720. cube 픽셀 크기·aspect 일치
# → generalize 정확도 ↑.
RS_W, RS_H, RS_FPS = 1280, 720, 30

# 그리드 (base 좌표, mm) — 사용자 환경에 맞게 조정
# z 는 각 cube 의 pick z 와 동일하게 가도록 execute_one_cycle 에서 동적으로 계산.
GRID_ORIGIN_X = 350.0    # 그리드 좌상단 X
GRID_ORIGIN_Y = -150.0   # 그리드 좌상단 Y
GRID_COLS = 4
GRID_ROWS = 3
GRID_SPACING = 50.0      # mm

# 그리퍼
CUBE_WIDTH_MM = 25.0
PRE_OPEN_WIDTH_MM = 50.0
RELEASE_WIDTH_MM = 27.0  # place 시 핑거 폭 — cube(25mm) 보다 살짝만 벌려서 위치 흐트러짐 방지
GRIP_RANGE_MM = 130.0    # RH-P12-RN-A 완전 열림(POS 0)에서 약 130mm 추정 — 환경별 조정

# 모션
Z_APPROACH = 80.0        # pre-grasp z 여유
Z_LIFT = 100.0           # 잡은 후 들어올림
MIN_NEIGHBOR_DIST_MM = 60.0   # 이 거리 안에 이웃 있으면 회피 yaw 계산


def width_mm_to_pos(width_mm: float) -> int:
    """폭(mm) → 그리퍼 POS (0=완전 열림, 700=완전 닫힘) 선형 매핑."""
    width = max(0.0, min(GRIP_RANGE_MM, width_mm))
    pos = 700.0 * (1.0 - width / GRIP_RANGE_MM)
    return int(round(pos))


def _normalize_yaw_pm45(yaw):
    """정사각형 cube → 90° 대칭. 그리퍼 회전을 최소화하도록 -45~+45 로 정규화."""
    while yaw > 45.0:
        yaw -= 90.0
    while yaw < -45.0:
        yaw += 90.0
    return yaw


# ===== Vision (다중 객체 + best.pt) =====
class CubeDetector:
    """12번 VisionDetector 와 같은 구조지만 set_classes 회피 + detect_all 추가."""

    def __init__(self):
        calib_path = p12.CALIB_PATH
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f'캘리브레이션 없음: {calib_path}')
        d = np.load(calib_path)
        self.T_cam2base = d['T_cam2base']
        print(f'   캘리브 로드: translation='
              f'{[round(v*1000, 1) for v in self.T_cam2base[:3, 3]]} mm')

        print(f'   YOLO 로드: {YOLO_WEIGHTS}')
        self.model = YOLO(YOLO_WEIGHTS)

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
        cfg.enable_stream(rs.stream.depth, RS_W, RS_H, rs.format.z16, RS_FPS)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        for _ in range(15):
            self.pipeline.wait_for_frames()

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def _depth_at(self, depth_frame, u, v, patch=5):
        z = depth_frame.get_distance(int(u), int(v))
        if z > 0.05:
            return z
        arr = np.asanyarray(depth_frame.get_data())
        H, W = arr.shape
        u0, v0 = max(0, int(u) - patch), max(0, int(v) - patch)
        u1, v1 = min(W, int(u) + patch + 1), min(H, int(v) + patch + 1)
        win = arr[v0:v1, u0:u1]
        win = win[win > 0]
        return float(win.mean()) * 0.001 if win.size else 0.0

    def _pixel_to_base(self, depth_frame, u, v):
        z = self._depth_at(depth_frame, u, v)
        if z <= 0.05:
            return None
        cam = rs.rs2_deproject_pixel_to_point(self.intr, [float(u), float(v)], z)
        h = np.array([cam[0], cam[1], cam[2], 1.0])
        return (self.T_cam2base @ h)[:3] * 1000.0  # mm

    def _estimate_top_face_from_depth_bbox(self, depth_frame, bbox):
        """
        YOLO가 찾은 bbox 영역 내에서 깊이(Depth) 데이터로 상단면(가장 가까운 면)만 추출.
        bbox: (x1, y1, x2, y2) 원본 이미지 픽셀 좌표
        반환: (yaw_deg, (cu, cv), box_pts) 실패 시 (None, None, None)
        """
        x1, y1, x2, y2 = bbox
        
        # 이미지 경계 처리
        W = depth_frame.get_width()
        H = depth_frame.get_height()
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None, None, None
            
        # 1. Bbox 영역의 깊이 데이터 가져오기 (단위: 미터)
        depth_data = np.asanyarray(depth_frame.get_data())
        crop_depth = depth_data[y1:y2, x1:x2] * 0.001
        
        # 0.05m(5cm) 초과의 유효한 깊이값만 추출
        valid_depth = crop_depth[crop_depth > 0.05]
        if len(valid_depth) < 50:
            return None, None, None
            
        # 2. 상단면 깊이 기준 설정 (노이즈 방지를 위해 상위 5% 값 사용)
        z_top = np.percentile(valid_depth, 5)
        
        # 3. 상단면 픽셀 이진화 (상단면 깊이 기준 + 5mm 이내인 픽셀만 추출)
        mask = np.zeros_like(crop_depth, dtype=np.uint8)
        mask[(crop_depth >= z_top - 0.005) & (crop_depth <= z_top + 0.005)] = 255
        
        # 노이즈 제거 (Morphology)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 4. 외곽선(Contour) 검출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None
            
        # 가장 큰 영역 선택
        largest_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_cnt) < 30:
            return None, None, None
            
        # 5. minAreaRect로 각도 및 중심 계산
        rect = cv2.minAreaRect(largest_cnt)
        (rcx, rcy), (rw, rh), _angle = rect
        if min(rw, rh) <= 0:
            return None, None, None
            
        # Bbox 좌상단 좌표를 더해 원본 이미지 좌표로 복원
        cu = float(rcx + x1)
        cv_ = float(rcy + y1)
        
        box = cv2.boxPoints(rect)
        box[:, 0] += x1
        box[:, 1] += y1
        box = box.astype(np.float32)
        
        # 정사각형 검증 (종횡비가 1.6을 넘어가면 상단면이 아니라고 간주)
        if max(rw, rh) / min(rw, rh) > 1.6:
            return None, None, None
            
        # 6. 긴 변을 이용해 yaw(각도) 계산
        p0, p1, p2 = box[0], box[1], box[2]
        e01 = np.linalg.norm(p1 - p0)
        e12 = np.linalg.norm(p2 - p1)
        pa, pb = (p0, p1) if e01 >= e12 else (p1, p2)
        
        a3 = self._pixel_to_base(depth_frame, pa[0], pa[1])
        b3 = self._pixel_to_base(depth_frame, pb[0], pb[1])
        if a3 is None or b3 is None:
            return None, (cu, cv_), box.astype(np.int32)
            
        dx = b3[0] - a3[0]
        dy = b3[1] - a3[1]
        if dx == 0 and dy == 0:
            return None, (cu, cv_), box.astype(np.int32)
            
        base_yaw = math.degrees(math.atan2(dy, dx))
        return _normalize_yaw_pm45(base_yaw), (cu, cv_), box.astype(np.int32)

    def detect_all(self, save_debug=None):
        """반환: [{'base_xyz_mm': (x,y,z), 'pixel': (u,v), 'conf': c, ...}, ...]"""
        frames = self.align.process(self.pipeline.wait_for_frames())
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df:
            return []
        color = np.asanyarray(cf.get_data())

        results = self.model(color, conf=DETECT_CONF_THR, iou=NMS_IOU_THR,
                             verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        boxes = results[0].boxes
        xywh = boxes.xywh.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        # seg 모델 → polygons (원본 이미지 픽셀 좌표 list)
        polys = results[0].masks.xy if results[0].masks is not None else [None] * len(boxes)

        out = []
        dbg = color.copy() if save_debug else None
        for i in range(len(boxes)):
            cx, cy, w, h = xywh[i]
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            poly = polys[i] if i < len(polys) else None
            cube_yaw, center_uv, box_pts = self._estimate_top_face_from_depth_bbox(df, (x1, y1, x2, y2))
            if center_uv is not None:
                u, v = int(round(center_uv[0])), int(round(center_uv[1]))
            else:
                u, v = int(cx), int(cy)
            base = self._pixel_to_base(df, u, v)
            if base is None:
                continue
            out.append({
                'base_xyz_mm': (float(base[0]), float(base[1]), float(base[2])),
                'pixel': (u, v),
                'conf': float(confs[i]),
                'cube_yaw_deg': cube_yaw,
                'box_pts': box_pts,
                'center_src': 'depth' if center_uv is not None else 'bbox',
            })
            if dbg is not None:
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if poly is not None and len(poly) >= 3:
                    cv2.polylines(dbg, [poly.astype(np.int32)], True, (255, 0, 255), 1)
                cv2.putText(dbg, f'{confs[i]:.2f}', (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(dbg, (u, v), 4, (0, 0, 255), -1)
                if box_pts is not None:
                    cv2.polylines(dbg, [box_pts], True, (0, 200, 255), 2)
                if cube_yaw is not None:
                    cv2.putText(dbg, f'{cube_yaw:+.1f}deg',
                                (x1, min(color.shape[0] - 5, y2 + 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
        if dbg is not None and save_debug:
            cv2.imwrite(save_debug, dbg)
        return out

    def preview_until_confirm(self, save_debug=None):
        """라이브 화면 + 검출 → 사용자 확인 후 그 시점 검출 반환.
        키:  ENTER/SPACE = 확정 (검출 결과 return)
             q / ESC     = 취소 (None return)
             s           = 스크린샷 저장
        """
        win = 'YOLO preview (ENTER=confirm, q=quit, s=shot)'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print('  preview 시작 — 검출 결과 확인 후 ENTER 로 확정 (q=취소)')
        try:
            while True:
                frames = self.align.process(self.pipeline.wait_for_frames())
                cf = frames.get_color_frame()
                df = frames.get_depth_frame()
                if not cf or not df:
                    continue
                color = np.asanyarray(cf.get_data())

                results = self.model(color, conf=DETECT_CONF_THR, iou=NMS_IOU_THR,
                                     verbose=False)
                dets = []
                vis = color.copy()
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    xywh = boxes.xywh.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    polys = (results[0].masks.xy if results[0].masks is not None
                             else [None] * len(boxes))
                    for i in range(len(boxes)):
                        cx, cy, w, h = xywh[i]
                        x1, y1 = int(cx - w / 2), int(cy - h / 2)
                        x2, y2 = int(cx + w / 2), int(cy + h / 2)
                        poly = polys[i] if i < len(polys) else None
                        cube_yaw, center_uv, box_pts = self._estimate_top_face_from_depth_bbox(df, (x1, y1, x2, y2))
                        if center_uv is not None:
                            u, v = int(round(center_uv[0])), int(round(center_uv[1]))
                        else:
                            u, v = int(cx), int(cy)
                        base = self._pixel_to_base(df, u, v)
                        if base is None:
                            continue
                        dets.append({
                            'base_xyz_mm': (float(base[0]), float(base[1]), float(base[2])),
                            'pixel': (u, v),
                            'conf': float(confs[i]),
                            'cube_yaw_deg': cube_yaw,
                            'box_pts': box_pts,
                            'center_src': 'depth' if center_uv is not None else 'bbox',
                        })
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if poly is not None and len(poly) >= 3:
                            cv2.polylines(vis, [poly.astype(np.int32)], True, (255, 0, 255), 1)
                        cv2.circle(vis, (u, v), 4, (0, 0, 255), -1)
                        yaw_str = f'{cube_yaw:+.1f}d' if cube_yaw is not None else 'yaw=?'
                        src_str = 'D' if center_uv is not None else 'BB'
                        cv2.putText(vis, f'#{len(dets)-1} {confs[i]:.2f} {yaw_str} {src_str}',
                                    (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(vis,
                                    f'({base[0]:.0f},{base[1]:.0f},{base[2]:.0f})mm',
                                    (x1, min(color.shape[0] - 5, y2 + 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
                        if box_pts is not None:
                            cv2.polylines(vis, [box_pts], True, (0, 200, 255), 2)

                cv2.putText(vis,
                            f'detected={len(dets)}  ENTER=confirm  q=quit  s=shot',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(win, vis)
                k = cv2.waitKey(20) & 0xFF
                if k in (ord('q'), 27):
                    cv2.destroyWindow(win)
                    cv2.waitKey(1)
                    return None
                if k in (13, 10, ord(' ')):
                    if save_debug:
                        cv2.imwrite(save_debug, vis)
                    cv2.destroyWindow(win)
                    cv2.waitKey(1)
                    return dets
                if k == ord('s'):
                    out = save_debug or '/tmp/cubes_preview.png'
                    cv2.imwrite(out, vis)
                    print(f'  screenshot: {out}')
        except Exception:
            cv2.destroyWindow(win)
            cv2.waitKey(1)
            raise


# ===== 그리드 & yaw =====
def make_grid_cells():
    """[(x_mm, y_mm), ...]  4x3 = 12개, 좌상→우→다음행."""
    cells = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cells.append((
                GRID_ORIGIN_X + c * GRID_SPACING,
                GRID_ORIGIN_Y + r * GRID_SPACING,
            ))
    return cells


def compute_yaw_deg(target, neighbors):
    """그리퍼 yaw(deg) 결정:
       1) cube 자체 회전각이 추출됐으면 그걸 그대로 사용 (cube 변과 핑거 평행).
       2) 추출 실패 시 fallback: 이웃 회피 yaw (가장 가까운 이웃 방향에 직각).
       3) 그것도 없으면 0°.
       반환은 -45~+45 로 정규화 (정사각형 cube 90° 대칭 활용)."""
    cube_yaw = target.get('cube_yaw_deg')
    if cube_yaw is not None:
        return _normalize_yaw_pm45(float(cube_yaw))

    tx, ty, _ = target['base_xyz_mm']
    best_dist = float('inf')
    best_dx = best_dy = 0.0
    for n in neighbors:
        if n is target:
            continue
        nx, ny, _ = n['base_xyz_mm']
        d = math.hypot(nx - tx, ny - ty)
        if d < best_dist:
            best_dist = d
            best_dx, best_dy = nx - tx, ny - ty
    if best_dist > MIN_NEIGHBOR_DIST_MM:
        return 0.0
    nbr_angle = math.degrees(math.atan2(best_dy, best_dx))
    return _normalize_yaw_pm45(nbr_angle + 90.0)


def select_next_target(detections, used_indices):
    """아직 안 옮긴 cube 중 가장 이웃과 멀리 떨어진 것 우선 (잡기 쉬움)."""
    candidates = [(i, d) for i, d in enumerate(detections) if i not in used_indices]
    if not candidates:
        return None
    best = None
    best_iso = -1.0
    for i, d in candidates:
        dx, dy, _ = d['base_xyz_mm']
        min_d = min(
            (math.hypot(d2['base_xyz_mm'][0] - dx, d2['base_xyz_mm'][1] - dy)
             for j, d2 in enumerate(detections) if j != i),
            default=float('inf'))
        if min_d > best_iso:
            best_iso = min_d
            best = i
    return best


# ===== Pick & Place 사이클 =====
# 주의: RH-P12-RN-A 는 spring-loaded 가 아니라서 "토크 OFF 로 천천히 놓기"가 안 통한다.
# 토크가 풀려도 핑거가 자력으로 안 벌어지고, cube 가 friction 으로 그대로 잡혀있음.
# release 는 반드시 명시적 open 명령(gripper_open / pos 0)으로 처리한다.


def gripper_set_width(robot, width_mm: float, settle=0.6):
    """그리퍼 폭(mm) 명령. 12번 gripper_set 과 동일 구조."""
    pos = width_mm_to_pos(width_mm)
    return robot.gripper_set(pos, settle=settle, label=f'W{width_mm:.0f}mm')


def execute_one_cycle(robot, target, all_dets, cell_xy, args):
    """한 cube → 한 그리드 셀."""
    bx, by, bz = target['base_xyz_mm']
    yaw = compute_yaw_deg(target, all_dets)
    yaw_src = 'cube' if target.get('cube_yaw_deg') is not None else 'fallback'
    print(f'\n  >> cube @ ({bx:.1f}, {by:.1f}, {bz:.1f}) yaw={yaw:+.1f}° ({yaw_src}) '
          f'→ cell ({cell_xy[0]:.0f}, {cell_xy[1]:.0f})')

    pick_rpy = [0.0, 180.0, yaw]
    # lift 후 cube 를 잡은 채로 yaw=0 으로 회전시켜 그리드 위에 모두 같은 각도로
    # 정렬되게 한다. friction 으로 cube 도 함께 회전하는 것을 가정 — 회전량 최대 ±45°.
    place_rpy = [0.0, 180.0, 0.0]

    # cube 옆면 그립을 위해 cube 중심 z(=bottom + cube_half) 로 내려감
    # detect z 가 cube 윗면이라고 가정 → 옆면 정확히 잡으려면 z 를 cube_half 만큼 더 내려야
    pick_z = bz - CUBE_WIDTH_MM / 2.0
    approach = [bx, by, pick_z + Z_APPROACH]
    pick = [bx, by, pick_z]
    lift = [bx, by, pick_z + Z_LIFT]

    # 내려놓는 z 는 잡았을 때 z 와 동일 — cube 가 같은 높이에서 release 되어
    # 그리드 표면에 끌리지 않고 자연스럽게 안착.
    place = [cell_xy[0], cell_xy[1], pick_z]
    place_app = [cell_xy[0], cell_xy[1], pick_z + Z_APPROACH]

    # ---- Pick ----
    print('   [Pick]')
    print(f'    1) approach → {[round(v, 1) for v in approach]}')
    if not args.dry_run: robot.move_line_base(approach, rpy_deg=pick_rpy)
    print(f'    2) pre-open ({PRE_OPEN_WIDTH_MM:.0f}mm = pos {width_mm_to_pos(PRE_OPEN_WIDTH_MM)})')
    if not args.dry_run: gripper_set_width(robot, PRE_OPEN_WIDTH_MM)
    print(f'    3) descend → {[round(v, 1) for v in pick]}')
    if not args.dry_run: robot.move_line_base(pick, rpy_deg=pick_rpy)
    print(f'    4) close (잡기, 700 = 완전 닫힘 → cube 가 막아서 멈춤)')
    if not args.dry_run: robot.gripper_close()
    print(f'    5) lift → {[round(v, 1) for v in lift]}')
    if not args.dry_run: robot.move_line_base(lift, rpy_deg=pick_rpy)
    if abs(yaw) > 0.5:
        print(f'    6) align: yaw {yaw:+.1f}° → 0° (cube 와 함께 회전)')
        if not args.dry_run: robot.move_line_base(lift, rpy_deg=place_rpy)

    # ---- Place ----
    print('   [Place]')
    print(f'    1) above cell → {[round(v, 1) for v in place_app]}')
    if not args.dry_run: robot.move_line_base(place_app, rpy_deg=place_rpy)
    print(f'    2) descend → {[round(v, 1) for v in place]}')
    if not args.dry_run: robot.move_line_base(place, rpy_deg=place_rpy)
    print(f'    3) open {RELEASE_WIDTH_MM:.0f}mm (release — RH-P12-RN-A 는 명시적 open 으로만 풀림)')
    if not args.dry_run: gripper_set_width(robot, RELEASE_WIDTH_MM)
    print(f'    4) lift → {[round(v, 1) for v in place_app]}')
    if not args.dry_run: robot.move_line_base(place_app, rpy_deg=place_rpy)


# ===== main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='모션 없이 콘솔만')
    ap.add_argument('--quick', action='store_true',
                    help='드라이버 리셋 SKIP — dsr_bringup2 가 이미 깨끗히 살아있을 때만')
    ap.add_argument('--limit', type=int, default=None, help='처음 N개만 처리')
    ap.add_argument('--debug-img', type=str, default='/tmp/cubes_debug.png')
    args = ap.parse_args()

    print('=== 바둑판 정렬 시작 ===')
    if args.dry_run:
        print('  *** DRY-RUN 모드: 실제 모션/그리퍼 명령은 보내지 않음 ***')
    grid_cells = make_grid_cells()
    print(f'  그리드: {GRID_ROWS}x{GRID_COLS} = {len(grid_cells)}셀, 간격 {GRID_SPACING}mm')
    print(f'  시작 셀: ({grid_cells[0][0]:.0f}, {grid_cells[0][1]:.0f}) → '
          f'끝 셀: ({grid_cells[-1][0]:.0f}, {grid_cells[-1][1]:.0f})')

    # ===== [준비 0/4] 드라이버 리셋 + dsr_bringup2 launch + RViz =====
    if not args.quick:
        p12.reset_robot_driver()
    else:
        print('\n[준비 0/4] 드라이버 리셋 SKIP (--quick)')

    print('\n[준비 1/4] 비전 init')
    vision = CubeDetector()

    print('\n[준비 2/4] 로봇 + 그리퍼 init')
    rclpy.init()
    robot = p12.PickAndPlace()
    try:
        if not args.dry_run:
            robot.activate_robot()
            robot.gripper_init()
            robot.gripper_open()
        print('   준비 OK')

        print(f'\n[준비 3/4] HOME 자세로 이동 {p12.HOME_JOINT_DEG} deg')
        if not args.dry_run:
            try:
                robot.move_joint_deg(p12.HOME_JOINT_DEG)
                print('   HOME 도착')
            except Exception as e:
                print(f'   !! HOME 이동 실패: {e}')
        else:
            print('   (dry-run: HOME 이동 SKIP)')

        print('\n[준비 4/4] 큐브 검출 (preview 윈도우 — ENTER 로 확정)')
        dets = vision.preview_until_confirm(save_debug=args.debug_img)
        if dets is None:
            print('  사용자 취소 — 종료')
            return
        print(f'  확정 {len(dets)}개 (debug 이미지: {args.debug_img})')
        for i, d in enumerate(dets):
            x, y, z = d['base_xyz_mm']
            print(f'   #{i}: ({x:.1f}, {y:.1f}, {z:.1f}) mm  conf={d["conf"]:.2f}')

        # 작업영역 밖 cube 필터링 (12번 WORK_X/Y/Z 기준).
        # approach z 와 pick z 모두 안에 들어와야 IK 실패 안 함.
        def _in_workspace(d):
            bx, by, bz = d['base_xyz_mm']
            pz = bz - CUBE_WIDTH_MM / 2.0
            az = pz + Z_APPROACH
            return (p12.WORK_X[0] <= bx <= p12.WORK_X[1] and
                    p12.WORK_Y[0] <= by <= p12.WORK_Y[1] and
                    p12.WORK_Z[0] <= pz <= p12.WORK_Z[1] and
                    p12.WORK_Z[0] <= az <= p12.WORK_Z[1])

        inside = [d for d in dets if _in_workspace(d)]
        skipped = len(dets) - len(inside)
        if skipped:
            print(f'  !! 작업영역 밖 cube {skipped}개 skip '
                  f'(X∈{p12.WORK_X}, Y∈{p12.WORK_Y}, Z∈{p12.WORK_Z})')
            for d in dets:
                if not _in_workspace(d):
                    x, y, z = d['base_xyz_mm']
                    print(f'     skip: ({x:.1f}, {y:.1f}, {z:.1f}) mm')
        dets = inside

        if not dets:
            print('!! 잡을 수 있는 cube 0개 — 종료')
            return

        n_target = len(dets) if args.limit is None else min(args.limit, len(dets))
        n_target = min(n_target, len(grid_cells))
        print(f'\n  {n_target}개 cube → {n_target}개 셀 매핑')

        used = set()
        for cell_idx in range(n_target):
            idx = select_next_target(dets, used)
            if idx is None:
                break
            used.add(idx)
            execute_one_cycle(robot, dets[idx], dets, grid_cells[cell_idx], args)

        print('\n  HOME 복귀')
        if not args.dry_run:
            try:
                robot.move_joint_deg(p12.HOME_JOINT_DEG)
            except Exception as e:
                print(f'   !! HOME 복귀 실패: {e}')
        print('\n=== 완료 ===')
    finally:
        try:
            vision.stop()
        except Exception:
            pass
        try:
            robot.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
