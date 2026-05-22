"""
나무 큐브 바둑판 정렬 — RealSense + 학습된 best.pt + 두산 e0509 + RH-P12-RN-A.

흐름:
  1) detect_all(): YOLO 검출 → 각 bbox center 의 base 좌표 (mm)
  2) 그리드 셀 4x3 (50mm 간격) 미리 계산
  3) 각 cube 에 대해:
     - approach(top-down) → 그리퍼 pre-open → descend → close → lift
     - 그리드 셀로 이동 → descend → release → lift
  4) 모든 cube 배치 완료 후 HOME

사용:
  python3 15_바둑판_정렬.py            # 실 동작
  python3 15_바둑판_정렬.py --dry-run  # 모션 없이 콘솔 출력만
  python3 15_바둑판_정렬.py --limit N  # 처음 N개만

12번을 importlib 로 재사용 (PickAndPlace, helpers).
검출은 YOLO 객체 인식만 — 상단면 추출/yaw/트래킹 등 후처리 없음.
"""
import argparse
import importlib.util
import math
import os
import sys
import time
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
YOLO_WEIGHTS = str(Path(__file__).parent / 'yolo_dataset/runs/seg_v6/weights/best.pt')
DETECT_CONF_THR = 0.40
NMS_IOU_THR = 0.45
RS_W, RS_H, RS_FPS = 1280, 720, 30
RS_DEPTH_W, RS_DEPTH_H = 1280, 720

# 그리드 (base 좌표, mm)
GRID_ORIGIN_X = 350.0
GRID_ORIGIN_Y = -150.0
GRID_COLS = 5
GRID_ROWS = 5
GRID_SPACING = 50.0

# cube center 가 grid 셀 중심에서 이 거리 이내면 "그 셀 자리에 있다" → 충돌 감지용
CELL_OCCUPIED_RADIUS_MM = 30.0
# cube 가 자기 target cell 에 이미 정확히 정렬돼 있다고 판단하는 거리 — 옮기지 않음
# 너무 크면 어긋난 cube 도 skip 됨 → 정렬 흐트러짐
ALREADY_PLACED_RADIUS_MM = 15.0

# 인접 cube 회피 — finger 가 cube 양옆에 들어갈 빈공간 검사 기준
# cube center 반지름 + finger 절반 + margin = 17.5mm 정도가 finger 위치
FINGER_DIST_MM = 17.5
# finger 위치 ↔ 다른 cube center 가 이 거리 이내면 finger 들어갈 공간 부족
FINGER_CLEARANCE_MIN_MM = 20.0

# mask polygon 품질 검증 — false positive (학습 부족으로 cube 아닌 영역 검출) 차단
# 1280x720, cube 25mm. 거리 200~700mm 범위에서 cube top 픽셀 크기 변동 큼
# (200mm 가까이서 ~80px = area ~6400, 700mm 멀리서 ~22px = area ~480)
MIN_POLY_AREA_PX = 150      # 너무 작은 mask → 그림자/노이즈
MAX_POLY_AREA_PX = 15000    # 매우 가까운 cube + 측면 포함도 통과하도록 관대
MAX_POLY_ASPECT = 2.0       # 정사각형 cube top → 1 근처. 가까운 perspective 도 통과

# preview 에서 사용자가 클릭으로 제거할 때 — 클릭 위치 ↔ detection pixel 매칭 반경 (px)
CLICK_REMOVE_RADIUS_PX = 30

# 트래킹 (마름모 요동 방지) — 프레임 간 같은 cube 매칭 + yaw/위치 EMA
TRACK_MATCH_RADIUS_MM = 25.0  # base XY 이 거리 안이면 같은 cube
TRACK_EMA_ALPHA = 0.15        # 새 관측 비율 (낮을수록 안정, 반응 느림)
TRACK_LOCK_FRAMES = 5         # 이만큼 연속 보이면 locked → 굵게 표시
TRACK_STALE_FRAMES = 10       # 이만큼 안 보이면 트랙 삭제

# 그리퍼
CUBE_WIDTH_MM = 25.0
# pre-open 폭 (잡기 전 descend 중 cube 안 치게). cube 25mm + 대각 35.4mm 둘 다 안 닿게 45mm.
GRIPPER_OPEN_WIDTH_MM = 45.0
# release 폭 (놓을 때 살짝만 벌려서 부드럽게 안착). cube 25mm + 10mm 마진.
RELEASE_OPEN_WIDTH_MM = 35.0
# Half-close 폭 (자가정렬용). cube 옆면에 살짝 닿아 회전 토크 발생, 코너 잡기 회피.
# cube 25mm + 3mm 여유. half-close 후 HALF_CLOSE_SETTLE_SEC 대기 → cube 자가 회전 시간.
HALF_CLOSE_WIDTH_MM = 28.0
HALF_CLOSE_SETTLE_SEC = 0.3
# Wiggle (yaw 회전) — half-close 후 그리퍼 yaw 를 ±N° 흔들어서 cube 강제 회전 정렬.
# finger 가 cube 옆면 살짝 잡은 상태에서 회전 → cube 가 따라 회전 → 옆면 평행 정렬.
WIGGLE_YAW_DEG = 10.0
WIGGLE_DURATION_SEC = 0.5
GRIP_RANGE_MM = 130.0    # RH-P12-RN-A 완전 열림(POS 0)에서 약 130mm 추정

# 0° snap 보정 — 학습 라벨이 axis-aligned 라 검출 yaw 가 0° 부근으로 collapse 함.
# 실제 흩뿌려진 cube 는 거의 0° 가 아니므로, 검출이 0° 근처면 작은 회전 부여.
YAW_NEAR_ZERO_THR_DEG = 10.0     # |yaw| 이값 이내면 "0° 로 잘못 snap" 으로 간주
YAW_NEAR_ZERO_BIAS_DEG = 22.5    # 위 경우에 사용할 회전량 (부호는 첫 두 변 중 finger clearance 좋은 쪽)


# 모션
Z_APPROACH = 80.0
Z_LIFT = 100.0
# 사용자 셋업: 바닥 = robot base origin 기준 z=-30mm.
FLOOR_Z = -30.0
# pick/place 시 TCP 의 절대 z = 바닥 위 N mm. 검출 cube 윗면(bz) 과 무관.
# 모든 cube 가 바닥에 안착해 있으므로 일정 절대 높이에서 잡고 놓음.
PICK_HEIGHT_ABOVE_FLOOR_MM = 17.0
PICK_Z = FLOOR_Z + PICK_HEIGHT_ABOVE_FLOOR_MM   # = -13.0

# Place 강하 시작 z (바닥 위 N mm). place_app → place 가 같은 X,Y 라 수직 일직선 강하.
PLACE_APP_HEIGHT_ABOVE_FLOOR_MM = 80.0
PLACE_APP_Z = FLOOR_Z + PLACE_APP_HEIGHT_ABOVE_FLOOR_MM   # = +50.0

# 시차 보정 — detect center 를 이미지 v 방향으로 N 픽셀 위로 옮긴 후 base 재투영.
# 카메라가 완전 top-down 이 아니라 살짝 비스듬해서 mask 가 전체적으로 아래쪽으로 치우치는
# 현상 보정. 0 이면 보정 없음. 큐브 거리/카메라 각도에 따라 튜닝.
CENTER_UP_PIXEL = 3

# 일반 모션 segment 이동 시간 (빠름). transit/HOME 등.
MOVE_DURATION_SEC = 1.5
# 잡기/놓기 직전 강하 (느림, 정밀). pick descend + place descend + yaw_retry descend.
DESCEND_DURATION_SEC = 2.5
# 그리퍼 settle — RH-P12-RN-A 작은 변화는 0.1s 면 충분 (close 안정성 모니터 후 조정).
GRIPPER_SETTLE_SEC = 0.1

# 15번 자체 작업영역 — 12번 기본값 (-600~600) 보다 사용자 셋업이 클 경우 override.
# 두산 e0509 reach ≈ 920mm 이므로 X 한계는 그 이하로.
WORK_X = (-1000.0, 1000.0)
WORK_Y = (-1000.0, 1000.0)
WORK_Z = (-150.0, 600.0)


def width_mm_to_pos(width_mm: float) -> int:
    """폭(mm) → 그리퍼 POS (0=완전 열림, 700=완전 닫힘) 선형 매핑."""
    width = max(0.0, min(GRIP_RANGE_MM, width_mm))
    pos = 700.0 * (1.0 - width / GRIP_RANGE_MM)
    return int(round(pos))




# ===== Vision (YOLO 객체 인식만) =====
class CubeDetector:
    """RealSense + YOLO. bbox center 의 base 좌표만 반환."""

    def __init__(self, conf_thr=DETECT_CONF_THR):
        self.conf_thr = float(conf_thr)
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
        cfg.enable_stream(rs.stream.depth, RS_DEPTH_W, RS_DEPTH_H, rs.format.z16, RS_FPS)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        for _ in range(15):
            self.pipeline.wait_for_frames()
        self._reset_tracking()

    def _reset_tracking(self):
        self._tracks = {}
        self._next_tid = 0
        self._frame_idx = 0
        # 사용자가 preview 에서 클릭으로 제거한 detection — track_id 기준 (locked
        # 된 cube). track_id 없는 detection 은 pixel 좌표 기준 (proximity).
        self._user_excluded_track_ids = set()
        self._user_excluded_pixels = []   # [(u, v), ...] 매 frame proximity 매칭

    @staticmethod
    def _yaw_ema_pm45(old_deg, new_deg, alpha):
        """4-fold(90° 주기) 각도 EMA — ±45° 경계 wrap 안전."""
        if old_deg is None:
            return new_deg
        if new_deg is None:
            return old_deg
        f = 4.0
        o = math.radians(old_deg * f)
        n = math.radians(new_deg * f)
        c = (1 - alpha) * math.cos(o) + alpha * math.cos(n)
        s = (1 - alpha) * math.sin(o) + alpha * math.sin(n)
        avg = math.degrees(math.atan2(s, c)) / f
        while avg > 45.0:
            avg -= 90.0
        while avg <= -45.0:
            avg += 90.0
        return avg

    def _stabilize(self, dets):
        """매 프레임 dets 의 yaw/base/pixel 을 EMA 로 평활.
        같은 cube 는 base XY 거리로 매칭 (TRACK_MATCH_RADIUS_MM).
        반환 없음 — dets 각 항목에 'track_id', 'locked' 추가하고 값을 in-place 갱신."""
        self._frame_idx += 1
        a = TRACK_EMA_ALPHA
        used = set()
        for d in dets:
            cx, cy, cz = d['base_xyz_mm']
            best_tid, best_d = None, TRACK_MATCH_RADIUS_MM
            for tid, t in self._tracks.items():
                if tid in used:
                    continue
                tx, ty = t['center_xy']
                dd = math.hypot(cx - tx, cy - ty)
                if dd < best_d:
                    best_d, best_tid = dd, tid

            if best_tid is None:
                best_tid = self._next_tid
                self._next_tid += 1
                self._tracks[best_tid] = {
                    'center_xy': (cx, cy),
                    'base_z': cz,
                    'pixel': tuple(map(float, d['pixel'])),
                    'yaw': d.get('cube_yaw_deg'),
                    'seen': 1,
                    'last_seen': self._frame_idx,
                }
            else:
                t = self._tracks[best_tid]
                t['center_xy'] = ((1 - a) * t['center_xy'][0] + a * cx,
                                  (1 - a) * t['center_xy'][1] + a * cy)
                t['base_z'] = (1 - a) * t['base_z'] + a * cz
                t['pixel'] = ((1 - a) * t['pixel'][0] + a * d['pixel'][0],
                              (1 - a) * t['pixel'][1] + a * d['pixel'][1])
                t['yaw'] = self._yaw_ema_pm45(t['yaw'], d.get('cube_yaw_deg'), a)
                t['seen'] += 1
                t['last_seen'] = self._frame_idx
            used.add(best_tid)

            t = self._tracks[best_tid]
            d['base_xyz_mm'] = (t['center_xy'][0], t['center_xy'][1], t['base_z'])
            d['pixel'] = (int(round(t['pixel'][0])), int(round(t['pixel'][1])))
            d['cube_yaw_deg'] = t['yaw']
            d['track_id'] = best_tid
            d['locked'] = t['seen'] >= TRACK_LOCK_FRAMES

        # 깜빡임 방지: locked 된 track 은 STALE_FRAMES 무관 영구 유지.
        # 비-locked track 만 STALE_FRAMES 후 삭제 (false positive 정리).
        stale = [tid for tid, t in self._tracks.items()
                 if self._frame_idx - t['last_seen'] > TRACK_STALE_FRAMES
                 and t.get('seen', 0) < TRACK_LOCK_FRAMES]
        for tid in stale:
            del self._tracks[tid]

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    @staticmethod
    def _as_depth_arr(depth_src):
        """depth_frame 또는 이미 변환된 numpy uint16(mm) 배열 둘 다 허용."""
        if isinstance(depth_src, np.ndarray):
            return depth_src
        return np.asanyarray(depth_src.get_data())

    def _depth_at(self, depth_frame, u, v, patch=7):
        """7x7 패치의 median depth (m)와 EMA(α=0.3) 로 노이즈를 억제합니다.
        기존 5x5 → 7x7 로 확대하고, 연속 프레임 간 깊이 값을 부드럽게 평균화합니다.
        """
        arr = self._as_depth_arr(depth_frame)   # uint16, mm
        H, W = arr.shape
        u0, v0 = max(0, int(u) - patch), max(0, int(v) - patch)
        u1, v1 = min(W, int(u) + patch + 1), min(H, int(v) + patch + 1)
        win = arr[v0:v1, u0:u1]
        win = win[win > 0]
        if win.size == 0:
            return 0.0
        depth = float(np.median(win)) * 0.001
        # EMA 저장 (프레임 별 깊이 평균)
        if not hasattr(self, '_depth_ema'):
            self._depth_ema = depth
        else:
            self._depth_ema = 0.7 * self._depth_ema + 0.3 * depth
        return float(self._depth_ema)

    def _pixel_to_base(self, depth_frame, u, v):
        z = self._depth_at(depth_frame, u, v)
        if z <= 0.05:
            return None
        cam = rs.rs2_deproject_pixel_to_point(self.intr, [float(u), float(v)], z)
        h = np.array([cam[0], cam[1], cam[2], 1.0])
        return (self.T_cam2base @ h)[:3] * 1000.0  # mm

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """마스크 후처리: open/close → 강한 close → 가장 큰 연결 컴포넌트 유지.
        마스크가 작은 잡음이나 구멍으로 인해 불안정해지는 상황을 방지합니다."""
        # 기본 open/close
        kernel = np.ones((3, 3), np.uint8)
        refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        # 추가 강한 close
        kernel_big = np.ones((5, 5), np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_big)
        # 가장 큰 연결 영역만 남기기
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
        if num_labels <= 1:
            return refined
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_out = np.zeros_like(refined)
        mask_out[labels == largest] = 255
        return mask_out

    def _polygon_centroid_px(self, polygon_xy):
        """polygon 의 무게중심 픽셀 (u, v) — cube top 의 실제 중앙."""
        if polygon_xy is None or len(polygon_xy) < 3:
            return None
        M = cv2.moments(polygon_xy.astype(np.float32))
        if M['m00'] <= 0:
            return None
        return (int(round(M['m10'] / M['m00'])),
                int(round(M['m01'] / M['m00'])))

    def _polygon_quality_ok(self, polygon_xy):
        """polygon 이 cube top 모양 (정사각형, 적정 크기) 인지 검증.
        반환: (ok, reason_str). false positive 차단용."""
        if polygon_xy is None or len(polygon_xy) < 3:
            return False, 'no-poly'
        pts = polygon_xy.astype(np.float32).reshape(-1, 1, 2)
        area = cv2.contourArea(pts)
        if area < MIN_POLY_AREA_PX:
            return False, f'area={area:.0f}<min'
        if area > MAX_POLY_AREA_PX:
            return False, f'area={area:.0f}>max'
        rect = cv2.minAreaRect(pts)
        rw, rh = rect[1]
        if min(rw, rh) < 1:
            return False, 'degenerate'
        aspect = max(rw, rh) / min(rw, rh)
        if aspect > MAX_POLY_ASPECT:
            return False, f'aspect={aspect:.2f}>{MAX_POLY_ASPECT}'
        return True, 'ok'

    def _polygon_to_base(self, depth_frame, polygon_xy, center_uv):
        """polygon 영역 안 픽셀들의 depth 중 25th percentile (cube top 표면) 으로
        center_uv 픽셀을 deproject → base 좌표. 단일 픽셀 depth 보다 robust."""
        if polygon_xy is None or len(polygon_xy) < 3:
            return self._pixel_to_base(depth_frame, center_uv[0], center_uv[1])
        arr = np.asanyarray(depth_frame.get_data())   # uint16 mm
        H, W = arr.shape
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_xy.astype(np.int32)], 1)
        vals = arr[(mask > 0) & (arr > 0)]
        if vals.size < 10:
            return self._pixel_to_base(depth_frame, center_uv[0], center_uv[1])
        # cube top 표면 = 가장 가까운 깊이 → 하위 25% 평균
        z_m = float(np.percentile(vals, 25)) * 0.001
        if z_m <= 0.05:
            return None
        cam = rs.rs2_deproject_pixel_to_point(
            self.intr, [float(center_uv[0]), float(center_uv[1])], z_m)
        h = np.array([cam[0], cam[1], cam[2], 1.0])
        return (self.T_cam2base @ h)[:3] * 1000.0   # mm

    def _polygon_to_base_yaw(self, depth_frame, polygon_xy):
        """seg polygon (Nx2 픽셀) → base frame yaw (deg).
        2D 픽셀에서 minAreaRect를 구하면 원근 왜곡(perspective)으로 인해 오차가 발생합니다.
        따라서 외곽선 픽셀들을 3D Base 좌표계(평면)로 변환한 뒤 minAreaRect를 구하여
        원근 왜곡 없이 완벽한 정사각형의 실제 회전 각도를 계산합니다.
        실패 시 None.
        """
        if polygon_xy is None or len(polygon_xy) < 3:
            return None
            
        base_pts_xy = []
        for pt in polygon_xy:
            u, v = pt
            p_base = self._pixel_to_base(depth_frame, float(u), float(v))
            if p_base is not None:
                base_pts_xy.append([p_base[0], p_base[1]])
                
        if len(base_pts_xy) < 3:
            return None
            
        base_pts_xy = np.array(base_pts_xy, dtype=np.float32)
        # Base XY 평면(3D)에서 최소 면적 사각형 계산 (원근 왜곡 없음)
        rect = cv2.minAreaRect(base_pts_xy)
        box = cv2.boxPoints(rect).astype(np.float32)
        
        # 긴 변의 각도 계산
        p0, p1, p2 = box[0], box[1], box[2]
        e01 = np.linalg.norm(p1 - p0)
        e12 = np.linalg.norm(p2 - p1)
        pa, pb = (p0, p1) if e01 >= e12 else (p1, p2)
        
        dx = pb[0] - pa[0]
        dy = pb[1] - pa[1]
        if dx == 0 and dy == 0:
            return 0.0
            
        yaw = math.degrees(math.atan2(dy, dx))
        
        # 90도 대칭 정규화 (-45 ~ +45)
        while yaw > 45.0:
            yaw -= 90.0
        while yaw <= -45.0:
            yaw += 90.0
            
        return float(yaw)

    def _compute_top_face_center_and_yaw(self, color, depth_frame, bbox):
        """
        YOLO bounding box 안에서 깊이(Depth)를 이용해 큐브의 '순수 상단면'만 분리하고,
        해당 마스크를 RGB 이미지에 덮어씌워 선명한 엣지로 회전각(yaw)을 구합니다.
        (현재 모델이 윗면/옆면을 구분하지 못할 때 윗면만 정확히 분리하는 강력한 로직)
        """
        x1, y1, x2, y2 = bbox
        arr = np.asanyarray(depth_frame.get_data())
        H, W = arr.shape
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(W, int(x2)); y2 = min(H, int(y2))
        if x2 - x1 < 5 or y2 - y1 < 5: return None
        
        crop = arr[y1:y2, x1:x2].astype(np.float32) * 0.001
        valid = crop[crop > 0.05]
        if valid.size < 30: return None
        z_top_m = float(np.percentile(valid, 5))

        # 사용자 요청에 따라 깊이 허용 오차를 ±12mm로 설정함 (z_top_m 기준 ±0.012)
        mask_local = ((crop >= z_top_m - 0.012) & (crop <= z_top_m + 0.012)).astype(np.uint8) * 255
        
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_local = cv2.morphologyEx(mask_local, cv2.MORPH_OPEN, k)
        mask_local = cv2.morphologyEx(mask_local, cv2.MORPH_CLOSE, k)
        
        # 윗면 가장자리에서 옆면으로 꺾이는 경계선 노이즈를 깎아냄 (아래쪽 쏠림 방지)
        k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_local = cv2.erode(mask_local, k_erode, iterations=1)

        # 가장 큰 connected component 만 유지 — 옆면 fragment, 인접 큐브 잔여 mask 제거.
        # 이게 없으면 minAreaRect 가 본체 + fragment 를 모두 감싸 중심이 크게 이동 ("사선 이동").
        n_lbl, labels = cv2.connectedComponents(mask_local)
        if n_lbl > 2:                                    # 1=배경, 2+ = 여러 덩어리
            sizes = [(labels == i).sum() for i in range(1, n_lbl)]
            largest = int(np.argmax(sizes)) + 1
            mask_local = (labels == largest).astype(np.uint8) * 255

        ys_local, xs_local = np.where(mask_local > 0)
        if len(ys_local) < 20: return None

        # 사용자 요청에 따라 깊이 임계값을 12mm(0.012)로 완화
        # (단, 위쪽에서 mask_local을 12mm로 재정의함)
        
        base_pts = []
        for v_loc, u_loc in zip(ys_local, xs_local):
            u = int(u_loc + x1)
            v = int(v_loc + y1)
            d = float(arr[v, u]) * 0.001
            if d < 0.05: continue
            cam = rs.rs2_deproject_pixel_to_point(self.intr, [float(u), float(v)], d)
            cam_h = np.array([cam[0], cam[1], cam[2], 1.0])
            bp = (self.T_cam2base @ cam_h)[:3] * 1000.0
            base_pts.append(bp)
            
        if len(base_pts) < 20: return None
        base_pts = np.array(base_pts, dtype=np.float32)
        
        # base 좌표로 변환된 윗면 점들의 외접 사각형(minAreaRect) 중심을 큐브 중심으로 사용.
        # 무게중심(mean)은 카메라 yaw 에 따라 옆면 픽셀이 mask 에 일부 포함되면 그쪽으로
        # 편향됨. minAreaRect 중심은 외접 사각형의 기하 중심이라 mask 형태 변형에 강함.
        xy_mm10 = (base_pts[:, :2] * 10.0).astype(np.int32)   # cv2 정수 입력 (0.1mm 정밀도)
        rect_center = cv2.minAreaRect(xy_mm10)[0]
        cx = float(rect_center[0]) / 10.0
        cy = float(rect_center[1]) / 10.0
        cz = float(np.median(base_pts[:, 2]))                  # 윗면 점들의 z median (depth 노이즈 흡수)

        # 시차 보정 — center 를 픽셀로 역투영 → v -CENTER_UP_PIXEL → 다시 base 재투영.
        # mask 가 전체적으로 아래쪽으로 치우치는 현상 (카메라 살짝 비스듬) 일률 보정.
        # 주의: rs2_deproject 는 *camera frame depth* (m) 필요 — base z 가 아니라 cam_m[2] 사용.
        if CENTER_UP_PIXEL != 0:
            T_base2cam = np.linalg.inv(self.T_cam2base)
            p_base_m = np.array([cx / 1000.0, cy / 1000.0, cz / 1000.0, 1.0])
            cam_m = T_base2cam @ p_base_m
            u_c, v_c = rs.rs2_project_point_to_pixel(self.intr, cam_m[:3])
            v_c_corrected = max(0.0, float(v_c) - float(CENTER_UP_PIXEL))
            cam_corr = rs.rs2_deproject_pixel_to_point(
                self.intr, [float(u_c), v_c_corrected], float(cam_m[2]))
            cam_corr_h = np.array([cam_corr[0], cam_corr[1], cam_corr[2], 1.0])
            base_corr = (self.T_cam2base @ cam_corr_h)[:3] * 1000.0
            cx = float(base_corr[0])
            cy = float(base_corr[1])

        # yaw — base_pts (mask 정제 + largest CC 거친 윗면 점들) 의 외접 사각형 각도.
        # 기존엔 RGB Canny+Hough 직선 검출로 yaw 추정했는데 옆면/그림자/조명 edge 가 섞여
        # ±몇도 어긋남이 발생. depth mask 정제된 base 좌표 점들의 minAreaRect 각도가
        # RGB 노이즈 영향 없이 더 안정적.
        xy_int = (base_pts[:, :2] * 10.0).astype(np.int32)
        rect_yaw = cv2.minAreaRect(xy_int)
        yaw_deg = float(rect_yaw[2])
        while yaw_deg > 45.0: yaw_deg -= 90.0
        while yaw_deg <= -45.0: yaw_deg += 90.0

        return {'center_xyz_mm': (cx, cy, cz), 'cube_yaw_deg': yaw_deg, 'yaw_src': 'rect'}

    def _yaw_from_depth_bbox(self, depth_frame, bbox):

        """YOLO bbox 내부의 깊이 데이터로 cube top 평면을 mask 화 → minAreaRect → base yaw.
        seg polygon 이 axis-aligned 라도 깊이는 회전을 보존해서 진짜 마름모 방향을 추출.
        실패 시 None."""
        x1, y1, x2, y2 = bbox
        arr = np.asanyarray(depth_frame.get_data())   # uint16 mm
        H, W = arr.shape
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(W, int(x2)); y2 = min(H, int(y2))
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        crop = arr[y1:y2, x1:x2].astype(np.float32) * 0.001   # m
        valid = crop[crop > 0.05]
        if valid.size < 50:
            return None
        # 가장 가까운 깊이 = cube top 표면. 하위 5% percentile 로 outlier 안전화
        z_top = float(np.percentile(valid, 5))
        mask = ((crop >= z_top - 0.005) & (crop <= z_top + 0.005)).astype(np.uint8) * 255
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 30:
            return None
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        # bbox 로컬 → 원본 이미지 좌표
        box_full = box + np.array([x1, y1], dtype=np.float32)
        return self._polygon_to_base_yaw(depth_frame, box_full)

    def _yaw_from_edges_bbox(self, color, depth_frame, bbox):
        """YOLO bbox 내부에서 Canny+HoughLinesP 로 cube 모서리 검출 → 4-fold 대칭
        circular mean 으로 이미지 평면 yaw → 가상 회전 박스 → base yaw.
        cube 모서리 대비가 또렷할 때 효과적. 실패 시 None."""
        x1, y1, x2, y2 = bbox
        H, W = color.shape[:2]
        pad = 5
        x1p = max(0, int(x1) - pad); y1p = max(0, int(y1) - pad)
        x2p = min(W, int(x2) + pad); y2p = min(H, int(y2) + pad)
        if x2p - x1p < 10 or y2p - y1p < 10:
            return None
        crop = color[y1p:y2p, x1p:x2p]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 150)
        min_len = max(8, int(0.4 * min(x2p - x1p, y2p - y1p)))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=20, minLineLength=min_len, maxLineGap=5)
        if lines is None or len(lines) < 2:
            return None
        # cube 4-fold 대칭 (90° 주기) → 각도×4 unfold 후 circular mean
        f = 4.0
        sin_sum, cos_sum, n = 0.0, 0.0, 0
        for x_1, y_1, x_2, y_2 in lines.reshape(-1, 4):
            if x_1 == x_2 and y_1 == y_2:
                continue
            a = math.atan2(y_2 - y_1, x_2 - x_1)
            sin_sum += math.sin(a * f)
            cos_sum += math.cos(a * f)
            n += 1
        if n == 0:
            return None
        yaw_img = math.degrees(math.atan2(sin_sum, cos_sum)) / f
        while yaw_img > 45.0:
            yaw_img -= 90.0
        while yaw_img <= -45.0:
            yaw_img += 90.0
        # 이미지 yaw → base yaw: bbox 중심 기준 가상 회전 박스 4 꼭짓점을
        # _polygon_to_base_yaw 로 통과시켜 카메라 기울기/calibration 보정까지 흡수
        cu = (x1 + x2) / 2.0
        cv_ = (y1 + y2) / 2.0
        half = 0.4 * min(x2 - x1, y2 - y1)
        rad = math.radians(yaw_img)
        cs, sn = math.cos(rad), math.sin(rad)
        local = np.array([[-half, -half], [half, -half],
                          [half, half], [-half, half]], dtype=np.float32)
        R = np.array([[cs, -sn], [sn, cs]], dtype=np.float32)
        box_px = (local @ R.T) + np.array([cu, cv_], dtype=np.float32)
        return self._polygon_to_base_yaw(depth_frame, box_px)

    def _refine_yaw_multi(self, color, depth_frame, bbox, polygon_xy):
        """cube yaw 를 3가지 방법 순서로 시도:
           1) depth plane fit — 회전 보존, 가장 robust
           2) RGB Canny+Hough — 모서리 또렷할 때
           3) seg polygon — 학습 라벨이 axis-aligned 일 때의 fallback
        반환: (yaw_deg, src_str). 모두 실패하면 (None, 'fail')."""
        yaw = self._yaw_from_depth_bbox(depth_frame, bbox)
        if yaw is not None:
            return yaw, 'depth'
        yaw = self._yaw_from_edges_bbox(color, depth_frame, bbox)
        if yaw is not None:
            return yaw, 'edges'
        yaw = self._polygon_to_base_yaw(depth_frame, polygon_xy)
        return (yaw, 'poly') if yaw is not None else (None, 'fail')

    def _get_projected_rhombus(self, base_xyz_mm, yaw_deg):
        """
        계산된 3D Base 좌표(중심점, 25x25mm, yaw)를 기반으로
        2D 카메라 화면에 어떻게 마름모 꼴로 투영되는지 픽셀 좌표를 계산합니다.
        물체를 '정사각형'으로 완벽히 인지하고 있음을 가시화하기 위함입니다.
        """
        if base_xyz_mm is None or yaw_deg is None:
            return None
            
        cx, cy, cz = base_xyz_mm
        half = CUBE_WIDTH_MM / 2.0
        rad = math.radians(yaw_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        # Base XY 평면 상의 정사각형 4 꼭짓점 (상단면)
        corners = [
            (-half, -half),
            ( half, -half),
            ( half,  half),
            (-half,  half)
        ]
        
        T_base2cam = np.linalg.inv(self.T_cam2base)
        rhombus_pts = []
        for dx, dy in corners:
            # yaw 회전
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            
            # Base 3D 좌표 -> m 단위
            p_base_m = np.array([(cx + rx) / 1000.0, (cy + ry) / 1000.0, cz / 1000.0, 1.0])
            # 카메라 좌표계로 역변환
            cam_m = T_base2cam @ p_base_m
            
            # 2D 픽셀로 투영
            pixel = rs.rs2_project_point_to_pixel(self.intr, cam_m[:3])
            rhombus_pts.append([int(round(pixel[0])), int(round(pixel[1]))])
            
        return np.array(rhombus_pts, dtype=np.int32)

    def _run_yolo(self, color):
        results = self.model(color, conf=self.conf_thr, iou=NMS_IOU_THR,
                             verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        return results[0]

    def detect_all(self, save_debug=None):
        """반환: [{'base_xyz_mm': (x,y,z), 'pixel': (u,v), 'conf': c}, ...]"""
        frames = self.align.process(self.pipeline.wait_for_frames())
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df:
            return []
        color = np.asanyarray(cf.get_data())

        r = self._run_yolo(color)
        if r is None:
            return []
        xywh = r.boxes.xywh.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        polys = (r.masks.xy if r.masks is not None else [None] * len(xywh))

        out = []
        dbg = color.copy() if save_debug else None
        n_reject = 0
        for i in range(len(xywh)):
            cx, cy, w, h = xywh[i]
            poly = polys[i] if i < len(polys) else None
            # 품질 검증 — false positive (cube 아닌 mask) reject
            ok, reason = self._polygon_quality_ok(poly)
            if not ok:
                n_reject += 1
                if dbg is not None:
                    x1, y1 = int(cx - w / 2), int(cy - h / 2)
                    x2, y2 = int(cx + w / 2), int(cy + h / 2)
                    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 200), 1)  # 빨강
                    cv2.putText(dbg, f'rej:{reason}', (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
                continue
            cen = self._polygon_centroid_px(poly)
            if cen is None:
                u, v = int(cx), int(cy)
            else:
                u, v = cen
            base = self._polygon_to_base(df, poly, (u, v))
            if base is None:
                continue
            bbox = (int(cx - w / 2), int(cy - h / 2),
                    int(cx + w / 2), int(cy + h / 2))
                    
            fit = self._compute_top_face_center_and_yaw(color, df, bbox)
            if fit is not None:
                base = fit['center_xyz_mm']
                yaw = fit['cube_yaw_deg']
                yaw_src = fit['yaw_src']
                T_base2cam = np.linalg.inv(self.T_cam2base)
                p_base_m = np.array([base[0]/1000.0, base[1]/1000.0, base[2]/1000.0, 1.0])
                cam_m = T_base2cam @ p_base_m
                pixel = rs.rs2_project_point_to_pixel(self.intr, cam_m[:3])
                u, v = int(round(pixel[0])), int(round(pixel[1]))
            else:
                yaw, yaw_src = self._refine_yaw_multi(color, df, bbox, poly)
            rhombus_pts = self._get_projected_rhombus(base, yaw)
            out.append({
                'base_xyz_mm': (float(base[0]), float(base[1]), float(base[2])),
                'pixel': (u, v),
                'conf': float(confs[i]),
                'cube_yaw_deg': yaw,
                'yaw_src': yaw_src,
                'rhombus_pts': rhombus_pts,
            })
            if dbg is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if poly is not None and len(poly) >= 3:
                    cv2.polylines(dbg, [poly.astype(np.int32)], True, (255, 0, 255), 1)
                cv2.circle(dbg, (u, v), 4, (0, 0, 255), -1)

                # 계산된 3D 정사각형(마름모) 가시화 (노란색 굵은 선)
                if rhombus_pts is not None:
                    cv2.polylines(dbg, [rhombus_pts], True, (0, 255, 255), 2)

                label = f'{confs[i]:.2f}'
                if yaw is not None:
                    label += f' yaw={yaw:+.1f}d[{yaw_src[0].upper()}]'
                cv2.putText(dbg, label, (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if dbg is not None and save_debug:
            cv2.imwrite(save_debug, dbg)
        return out

    def preview_until_confirm(self, save_debug=None, custom_msg=None, valid_keys=None):
        """라이브 화면 + 검출 → 사용자 확인 후 그 시점 검출 반환.
        키:  ENTER/SPACE = 확정,  q/ESC = 취소,  s = 스크린샷
        마우스: 좌클릭 = 클릭한 detection 제거 (회색 X 로 표시),
               우클릭 = 제거 목록 모두 복원
        custom_msg: 화면에 표시할 텍스트 리스트
        valid_keys: 지정된 키가 눌리면 (dets, key) 반환
        """
        win = 'YOLO preview (ENTER=confirm, click=remove, s=shot, q=quit)'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print('  preview 시작 — ENTER 로 확정 / 좌클릭 = 제거 / 우클릭 = 복원 / q = 취소')
        self._reset_tracking()
        # 마우스 콜백이 현재 frame 의 dets 를 보고 가장 가까운 것을 제거 → 클로저로
        # 매 frame 갱신되는 mutable container 공유.
        last_dets_ref = {'dets': []}

        def _on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                dets_now = last_dets_ref['dets']
                if not dets_now:
                    return
                best = None
                best_d2 = (CLICK_REMOVE_RADIUS_PX + 1) ** 2
                for d in dets_now:
                    pu, pv = d.get('pixel', (None, None))
                    if pu is None:
                        continue
                    d2 = (pu - x) ** 2 + (pv - y) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best = d
                if best is None:
                    print(f'  [click-remove] ({x},{y}) 근처 detection 없음 '
                          f'(반경 {CLICK_REMOVE_RADIUS_PX}px)')
                    return
                tid = best.get('track_id')
                if tid is not None:
                    self._user_excluded_track_ids.add(int(tid))
                    print(f'  [click-remove] track #{tid} 제거 '
                          f'(누적 제거 {len(self._user_excluded_track_ids)}개)')
                else:
                    pu, pv = best['pixel']
                    self._user_excluded_pixels.append((int(pu), int(pv)))
                    print(f'  [click-remove] pixel ({pu},{pv}) 제거 (track_id 없음)')
            elif event == cv2.EVENT_RBUTTONDOWN:
                n = (len(self._user_excluded_track_ids)
                     + len(self._user_excluded_pixels))
                if n == 0:
                    print('  [click-remove] 복원할 항목 없음')
                    return
                self._user_excluded_track_ids.clear()
                self._user_excluded_pixels.clear()
                print(f'  [click-remove] {n}개 제거 항목 모두 복원')

        cv2.setMouseCallback(win, _on_mouse)
        try:
            while True:
                frames = self.align.process(self.pipeline.wait_for_frames())
                cf = frames.get_color_frame()
                df = frames.get_depth_frame()
                if not cf or not df:
                    continue
                color = np.asanyarray(cf.get_data())

                r = self._run_yolo(color)
                dets = []
                draw_meta = []  # (bbox, conf, poly) — 그리기는 stabilize 후로 지연
                vis = color.copy()
                if r is not None:
                    xywh = r.boxes.xywh.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    cls_ids = r.boxes.cls.cpu().numpy()
                    polys = (r.masks.xy if r.masks is not None else [None] * len(xywh))
                    for i in range(len(xywh)):
                        cx, cy, w, h = xywh[i]
                        poly = polys[i] if i < len(polys) else None
                        # seg_v6 (단일 클래스 'wood cube') — 클래스 필터링 불필요.

                        # 품질 검증 — false positive reject
                        ok, reason = self._polygon_quality_ok(poly)
                        if not ok:
                            x1, y1 = int(cx - w / 2), int(cy - h / 2)
                            x2, y2 = int(cx + w / 2), int(cy + h / 2)
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 200), 1)
                            cv2.putText(vis, f'rej:{reason}',
                                        (x1, max(15, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
                            continue
                            
                        cen = self._polygon_centroid_px(poly)
                        if cen is None:
                            u, v = int(cx), int(cy)
                        else:
                            u, v = cen
                        base = self._polygon_to_base(df, poly, (u, v))
                        if base is None:
                            continue
                        bbox = (int(cx - w / 2), int(cy - h / 2),
                                int(cx + w / 2), int(cy + h / 2))
                                
                        fit = self._compute_top_face_center_and_yaw(color, df, bbox)
                        if fit is not None:
                            base = fit['center_xyz_mm']
                            yaw = fit['cube_yaw_deg']
                            yaw_src = fit['yaw_src']
                            T_base2cam = np.linalg.inv(self.T_cam2base)
                            p_base_m = np.array([base[0]/1000.0, base[1]/1000.0, base[2]/1000.0, 1.0])
                            cam_m = T_base2cam @ p_base_m
                            pixel = rs.rs2_project_point_to_pixel(self.intr, cam_m[:3])
                            u, v = int(round(pixel[0])), int(round(pixel[1]))
                        else:
                            yaw = self._polygon_to_base_yaw(df, poly)
                            yaw_src = 'top_poly'
                            if yaw is None:
                                yaw, yaw_src = 0.0, 'fail'
                            
                        dets.append({
                            'base_xyz_mm': (float(base[0]), float(base[1]), float(base[2])),
                            'pixel': (u, v),
                            'conf': float(confs[i]),
                            'cube_yaw_deg': yaw,
                            'yaw_src': yaw_src,
                        })
                        draw_meta.append((bbox, float(confs[i]), poly))

                # 트래킹 + EMA — yaw/base/pixel 안정화 (마름모 요동 방지)
                self._stabilize(dets)

                # 깜빡임 방지: locked 된 track 중 이번 frame 에서 YOLO 가 놓친 것들을
                # ghost detection 으로 추가 → 항상 표시 (cube 가 한번 인식되면 고정)
                seen_tids = {d.get('track_id') for d in dets if 'track_id' in d}
                for tid, t in self._tracks.items():
                    if tid in seen_tids:
                        continue
                    if t.get('seen', 0) < TRACK_LOCK_FRAMES:
                        continue
                    ghost = {
                        'base_xyz_mm': (t['center_xy'][0], t['center_xy'][1], t['base_z']),
                        'pixel': (int(round(t['pixel'][0])), int(round(t['pixel'][1]))),
                        'conf': 0.0,
                        'cube_yaw_deg': t['yaw'],
                        'yaw_src': 'lock',
                        'track_id': tid,
                        'locked': True,
                        'is_ghost': True,
                    }
                    dets.append(ghost)
                    draw_meta.append((None, 0.0, None))

                # 사용자가 클릭으로 제거한 detection 표시 (track_id 또는 pixel 기준)
                for d in dets:
                    excluded = False
                    tid = d.get('track_id')
                    if tid is not None and int(tid) in self._user_excluded_track_ids:
                        excluded = True
                    else:
                        pu, pv = d.get('pixel', (None, None))
                        if pu is not None:
                            for (eu, ev) in self._user_excluded_pixels:
                                if (pu - eu) ** 2 + (pv - ev) ** 2 <= CLICK_REMOVE_RADIUS_PX ** 2:
                                    excluded = True
                                    break
                    d['user_excluded'] = excluded

                # 마우스 콜백이 참조할 dets 갱신
                last_dets_ref['dets'] = dets

                # 그리기 (평활된 값으로 rhombus 재계산 → 화면에 표시)
                for d, meta in zip(dets, draw_meta):
                    bbox, conf, poly = meta
                    is_ghost = d.get('is_ghost', False)
                    is_excluded = d.get('user_excluded', False)
                    d['rhombus_pts'] = self._get_projected_rhombus(
                        d['base_xyz_mm'], d['cube_yaw_deg'])
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        box_color = (120, 120, 120) if is_excluded else (0, 255, 0)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 1)
                        if poly is not None and len(poly) >= 3:
                            poly_color = (120, 120, 120) if is_excluded else (255, 0, 255)
                            cv2.polylines(vis, [poly.astype(np.int32)], True, poly_color, 1)
                    else:
                        # ghost — bbox 없음, rhombus 기준 좌상 모서리로 라벨 위치 잡기
                        rp = d['rhombus_pts']
                        x1 = int(rp[:, 0].min())
                        y1 = int(rp[:, 1].min())
                        x2 = int(rp[:, 0].max())
                        y2 = int(rp[:, 1].max())
                    cv2.circle(vis, d['pixel'], 2, (0, 0, 255), -1)
                    if d.get('rhombus_pts') is not None:
                        thick = 2 if d.get('locked') else 1
                        if is_excluded:
                            color_rh = (100, 100, 100)
                        else:
                            color_rh = (180, 220, 255) if is_ghost else (0, 255, 255)
                        cv2.polylines(vis, [d['rhombus_pts']], True, color_rh, thick)
                    yaw, yaw_src = d['cube_yaw_deg'], d.get('yaw_src', '?')
                    yaw_str = (f' y={yaw:+.0f}d[{yaw_src[0].upper()}]'
                               if yaw is not None else '')
                    lock_str = ' GHOST' if is_ghost else (' LOCK' if d.get('locked') else ' trk')
                    if is_excluded:
                        label_color = (100, 100, 100)
                        lock_str += ' [REMOVED]'
                    else:
                        label_color = (200, 200, 200) if is_ghost else (0, 255, 0)
                    cv2.putText(vis,
                                f'#{d.get("track_id", "?")} {conf:.2f}{yaw_str}{lock_str}',
                                (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)
                    bx, by, bz = d['base_xyz_mm']
                    cv2.putText(vis, f'({bx:.0f},{by:.0f},{bz:.0f})mm',
                                (x1, min(color.shape[0] - 5, y2 + 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
                    # excluded 면 회색 X 오버레이 (가장 잘 보이게)
                    if is_excluded:
                        cv2.line(vis, (x1, y1), (x2, y2), (80, 80, 80), 2)
                        cv2.line(vis, (x1, y2), (x2, y1), (80, 80, 80), 2)

                n_kept = sum(1 for d in dets if not d.get('user_excluded'))
                n_removed = len(dets) - n_kept
                status = f'detected={len(dets)}  kept={n_kept}'
                if n_removed:
                    status += f'  removed={n_removed}'
                status += '  L-click=remove  R-click=undo all  ENTER=confirm  q=quit'
                if custom_msg:
                    cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset = 55
                    for line in custom_msg:
                        cv2.putText(vis, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        y_offset += 25
                else:
                    cv2.putText(vis, status, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow(win, vis)
                k = cv2.waitKey(20) & 0xFF
                if k in (ord('q'), 27):
                    cv2.destroyWindow(win)
                    cv2.waitKey(1)
                    return (None, None) if valid_keys else None

                if valid_keys:
                    char_k = chr(k) if 0 <= k < 256 else ''
                    if char_k in valid_keys:
                        if save_debug:
                            cv2.imwrite(save_debug, vis)
                        cv2.destroyWindow(win)
                        cv2.waitKey(1)
                        kept = [d for d in dets if not d.get('user_excluded')]
                        return kept, char_k
                else:
                    if k in (13, 10, ord(' ')):
                        if save_debug:
                            cv2.imwrite(save_debug, vis)
                        cv2.destroyWindow(win)
                        cv2.waitKey(1)
                        kept = [d for d in dets if not d.get('user_excluded')]
                        if n_removed:
                            print(f'  preview 확정: {len(dets)}개 중 {n_removed}개 '
                                  f'사용자 제거 → {len(kept)}개 사용')
                        return kept
                        
                if k == ord('s'):
                    out = save_debug or '/tmp/cubes_preview.png'
                    cv2.imwrite(out, vis)
                    print(f'  screenshot: {out}')
        except Exception:
            cv2.destroyWindow(win)
            cv2.waitKey(1)
            raise


# ===== 그리드 / Staging =====
# 명시적 5×5 grid 좌표 (mm, base frame). 계산식이 아닌 hardcoded list — 디버그 시 코드와
# 콘솔 출력이 100% 일치하게 함. 변경 시 GRID_ORIGIN_*/GRID_SPACING 도 함께 갱신 권장.
GRID_CELLS = [
    # row 0 (y=-150)
    (350.0, -150.0), (400.0, -150.0), (450.0, -150.0), (500.0, -150.0), (550.0, -150.0),
    # row 1 (y=-100)
    (350.0, -100.0), (400.0, -100.0), (450.0, -100.0), (500.0, -100.0), (550.0, -100.0),
    # row 2 (y= -50)
    (350.0,  -50.0), (400.0,  -50.0), (450.0,  -50.0), (500.0,  -50.0), (550.0,  -50.0),
    # row 3 (y=   0)
    (350.0,    0.0), (400.0,    0.0), (450.0,    0.0), (500.0,    0.0), (550.0,    0.0),
    # row 4 (y= +50)
    (350.0,   50.0), (400.0,   50.0), (450.0,   50.0), (500.0,   50.0), (550.0,   50.0),
]


def make_grid_cells():
    """[(x_mm, y_mm), ...]  GRID_ROWS x GRID_COLS 개, 좌상→우→다음행.
    hardcoded GRID_CELLS 반환 — 계산 X, 디버그 일관성 우선."""
    return list(GRID_CELLS)


def _wrap_yaw_pm90(yaw):
    """parallel-jaw 그리퍼는 180° 대칭 → ±90° 안으로 wrap.
    grasp 동일하지만 손목(J6) 회전량 최소화 → joint tolerance/Sudden-jump abort 회피."""
    while yaw > 90.0:
        yaw -= 180.0
    while yaw <= -90.0:
        yaw += 180.0
    return yaw


def pick_yaw_avoiding_neighbors(target_idx, dets, base_yaw, used_indices=None):
    """cube 가 정사각형 → yaw 와 yaw+90 두 방향 중 finger 가 인접 cube 와 충돌
    안 하는 쪽 선택. used_indices 의 cube 는 이미 옮겨져서 검사 안 함.
    둘 다 충돌이면 45°/135° (대각선 잡기) 도 시도. 모두 충돌이면 clearance 더 큰 쪽.
    반환: (yaw_deg, is_diagonal). yaw 는 -90~+90 범위 (parallel-jaw 180° wrap).
          is_diagonal=True 면 대각선 잡기 (모서리에 finger) — 호출자가 pre-open 폭 ↑."""
    used_indices = used_indices or set()
    cx, cy, _ = dets[target_idx]['base_xyz_mm']
    candidates = []
    for yaw_try in (base_yaw, base_yaw + 90.0):
        # blade face 위치 = cube center 에서 closing 방향 (= yaw) 으로 ±FINGER_DIST_MM
        # (이전 버전은 yaw+90° 로 계산해서 fingertip 위치를 봤고, 그 결과 collision 판정이
        #  실제와 90° 반대로 나옴 → neighbor 와 더 가까운 yaw 가 잘못 선택되던 버그)
        rad = math.radians(yaw_try)
        fdx = FINGER_DIST_MM * math.cos(rad)
        fdy = FINGER_DIST_MM * math.sin(rad)
        f1 = (cx + fdx, cy + fdy)
        f2 = (cx - fdx, cy - fdy)
        min_clearance = float('inf')
        worst_neighbor = None
        for i, d in enumerate(dets):
            if i == target_idx or i in used_indices:
                continue
            bx, by, _ = d['base_xyz_mm']
            for fp in (f1, f2):
                dist = math.hypot(bx - fp[0], by - fp[1])
                if dist < min_clearance:
                    min_clearance = dist
                    worst_neighbor = i
        candidates.append((yaw_try, min_clearance, worst_neighbor))

    best_yaw, best_clr, best_nb = max(candidates, key=lambda c: c[1])
    is_diagonal = False

    # 변 잡기 (0°/90°) 둘 다 finger 공간 부족 → 대각선 잡기 (45°/135°, cube 코너 끼움)
    # 시도. cube 가 정사각형이라 4-fold 대칭 + 45° 회전한 두 후보가 추가 가능.
    # 대각선 잡기는 finger 가 cube 변이 아닌 코너에 닿아 인접 cube 와 대각 공간으로 진입.
    if best_clr < FINGER_CLEARANCE_MIN_MM:
        diag_candidates = []
        for yaw_try in (base_yaw + 45.0, base_yaw + 135.0):
            rad = math.radians(yaw_try)
            fdx = FINGER_DIST_MM * math.cos(rad)
            fdy = FINGER_DIST_MM * math.sin(rad)
            f1 = (cx + fdx, cy + fdy)
            f2 = (cx - fdx, cy - fdy)
            min_clearance = float('inf')
            worst_neighbor = None
            for i, d in enumerate(dets):
                if i == target_idx or i in used_indices:
                    continue
                bx, by, _ = d['base_xyz_mm']
                for fp in (f1, f2):
                    dist = math.hypot(bx - fp[0], by - fp[1])
                    if dist < min_clearance:
                        min_clearance = dist
                        worst_neighbor = i
            diag_candidates.append((yaw_try, min_clearance, worst_neighbor))

        diag_best = max(diag_candidates, key=lambda c: c[1])
        if diag_best[1] > best_clr:
            best_yaw, best_clr, best_nb = diag_best
            is_diagonal = True
            print(f'    대각선 잡기 채택: yaw={best_yaw:+.1f}° clr={best_clr:.0f}mm '
                  f'(변 잡기 공간 부족 → 코너로 진입, pre-open ↑)')

    if best_clr < FINGER_CLEARANCE_MIN_MM:
        print(f'    !! cube #{target_idx} — 변/대각 모든 방향 finger 공간 부족 '
              f'(최대 clearance {best_clr:.0f}mm, 인접 cube #{best_nb}). 그대로 시도')
    else:
        # 어느 방향이 선택됐는지 로그 (대각 fallback 채택은 위에서 이미 출력)
        original_clr = candidates[0][1]
        if best_yaw == base_yaw + 90.0:
            print(f'    finger 회피: 원래 yaw={base_yaw:+.1f}° clr={original_clr:.0f}mm → '
                  f'90° 회전 yaw={best_yaw:+.1f}° clr={best_clr:.0f}mm '
                  f'(인접 cube #{best_nb} 회피)')
    # 180° 대칭 wrap → 손목 회전 최소화 (joint 5/6 tolerance abort 회피)
    wrapped = _wrap_yaw_pm90(best_yaw)
    if abs(wrapped - best_yaw) > 0.1:
        print(f'    180° wrap: yaw {best_yaw:+.1f}° → {wrapped:+.1f}° '
              f'(grasp 동일, 손목 회전량 ↓)')
    # 0° snap 보정 — 검출이 axis-aligned 로 collapse 한 경우 작은 회전 부여
    if abs(wrapped) < YAW_NEAR_ZERO_THR_DEG:
        # 부호: 이미 평가한 두 후보 중 base_yaw 쪽 부호를 따라감 (clearance 우선 결정 보존)
        sign = 1.0 if base_yaw >= 0 else -1.0
        biased = sign * YAW_NEAR_ZERO_BIAS_DEG
        print(f'    0° snap 보정: yaw {wrapped:+.1f}° → {biased:+.1f}° '
              f'(축정렬 학습 편향 보상)')
        wrapped = biased
    return float(wrapped), bool(is_diagonal)


def pick_yaw_for_grid_cell(target_idx, cell_positions, base_yaw, used_indices=None):
    """Grid 좌표 기반 neighbor-aware pick yaw (16번/17번 용).
    cell_positions: [(x_mm, y_mm), ...]  — 사용 중인 grid cell 좌표 리스트
    used_indices  : 이미 픽업 완료된 cell index set (남은 cube 만 neighbor 로 고려)
    base_yaw 와 base_yaw + 90 중 finger blade 가 neighbor 와 충돌 안 하는 쪽 선택 → ±90° wrap.

    물리 모델: wrist yaw 방향 = closing 방향 (= blade face 가 cube 를 누르는 방향).
    blade face 위치 = cube 중심에서 yaw 방향으로 ±FINGER_DIST_MM. 이 face 가 neighbor 와
    가까우면 잡으러 descend 할 때 충돌. 따라서 max blade-to-neighbor 거리 선택.
    """
    used_indices = used_indices or set()
    cx, cy = cell_positions[target_idx]
    candidates = []
    for yaw_try in (base_yaw, base_yaw + 90.0):
        rad = math.radians(yaw_try)
        fdx = FINGER_DIST_MM * math.cos(rad)
        fdy = FINGER_DIST_MM * math.sin(rad)
        f1 = (cx + fdx, cy + fdy)
        f2 = (cx - fdx, cy - fdy)
        min_clr = float('inf')
        worst_nb = None
        for i, (nx, ny) in enumerate(cell_positions):
            if i == target_idx or i in used_indices:
                continue
            for fp in (f1, f2):
                d = math.hypot(nx - fp[0], ny - fp[1])
                if d < min_clr:
                    min_clr = d
                    worst_nb = i
        candidates.append((yaw_try, min_clr, worst_nb))
    base_clr = candidates[0][1]
    max_yaw, max_clr, max_nb = max(candidates, key=lambda c: c[1])
    # base_yaw 가 안전 (clearance ≥ FINGER_CLEARANCE_MIN_MM) 이면 시각 일관성 위해 그대로.
    # base_yaw 가 위험할 때만 alternative 로 switch — 불필요한 yaw 변화 방지.
    if base_clr >= FINGER_CLEARANCE_MIN_MM:
        best_yaw, best_clr, best_nb = base_yaw, base_clr, candidates[0][2]
    else:
        best_yaw, best_clr, best_nb = max_yaw, max_clr, max_nb
        if best_yaw != base_yaw:
            print(f'    finger 회피: base yaw={base_yaw:+.1f}° clr={base_clr:.0f}mm '
                  f'(< {FINGER_CLEARANCE_MIN_MM:.0f}mm 위험) → '
                  f'{best_yaw:+.1f}° clr={best_clr:.0f}mm (인접 cell #{best_nb} 회피)')
        else:
            print(f'    !! cell #{target_idx} — 양 yaw 모두 blade 공간 부족 '
                  f'(최대 clearance {best_clr:.0f}mm). 그대로 시도')
    return _wrap_yaw_pm90(best_yaw)


def find_cube_at_cell(dets, used_indices, cell_xy, radius=CELL_OCCUPIED_RADIUS_MM):
    """cell_xy 위치 (radius 이내) 에 있는 cube 의 index 반환. 없으면 None.
    used_indices 에 포함된 cube 는 무시 (이미 처리된 것)."""
    cx, cy = cell_xy
    for i, d in enumerate(dets):
        if i in used_indices:
            continue
        bx, by, _ = d['base_xyz_mm']
        if math.hypot(bx - cx, by - cy) < radius:
            return i
    return None


def assign_cubes_to_cells(dets, grid_cells, n):
    """cube N 개 ↔ cell N 개 1:1 매핑 (Hungarian, 거리 최소화).
    반환: mapping[i] = cube i 가 갈 cell index."""
    from scipy.optimize import linear_sum_assignment
    cost = np.zeros((n, n))
    for i in range(n):
        bx, by, _ = dets[i]['base_xyz_mm']
        for j in range(n):
            cx, cy = grid_cells[j]
            cost[i, j] = math.hypot(bx - cx, by - cy)
    row, col = linear_sum_assignment(cost)
    mapping = [0] * n
    for r, c in zip(row, col):
        mapping[r] = int(c)
    return mapping


def select_next_target(detections, used_indices):
    """아직 안 옮긴 cube 중 가장 이웃과 멀리 떨어진 것 우선."""
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


def unify_z_with_outlier_reject(dets: list, tol_mm: float = 25.0) -> list:
    """같은 판 위 cube 들의 검출 z 를 median 으로 통일.
    median 에서 ±tol_mm 벗어난 cube 는 outlier 로 skip (잘못된 검출).
    depth 노이즈/calibration 미세 오차를 흡수해 pick/place 일관성 확보.
    dets 가 2개 미만이면 그대로 반환."""
    if len(dets) < 2:
        return dets
    zs = [d['base_xyz_mm'][2] for d in dets]
    zmed = float(np.median(zs))
    good, bad = [], []
    for d in dets:
        if abs(d['base_xyz_mm'][2] - zmed) <= tol_mm:
            good.append(d)
        else:
            bad.append(d)
    if bad:
        print(f'  !! z outlier cube {len(bad)}개 skip '
              f'(median z={zmed:.1f}mm, 허용 ±{tol_mm}mm)')
        for d in bad:
            x, y, z = d['base_xyz_mm']
            print(f'     skip: ({x:.1f}, {y:.1f}, {z:.1f}) mm (Δz={z-zmed:+.1f})')
    print(f'  z 통일: median {zmed:.1f}mm 로 모든 cube z 통일 '
          f'(같은 판 위 가정, depth 노이즈 흡수)')
    return [{**d, 'base_xyz_mm': (d['base_xyz_mm'][0], d['base_xyz_mm'][1], zmed)}
            for d in good]


# ===== Pick & Place 사이클 =====
# RH-P12-RN-A 는 spring-loaded 가 아니라서 토크 OFF 만으로 release 안 됨.
# 반드시 명시적 open 명령(pos < 700) 으로 풀어야 함.


def gripper_set_width(robot, width_mm: float, settle=GRIPPER_SETTLE_SEC):
    """그리퍼 폭(mm) 명령. 12번 gripper_set 과 동일 구조."""
    pos = width_mm_to_pos(width_mm)
    return robot.gripper_set(pos, settle=settle, label=f'W{width_mm:.0f}mm')


def _descend_cartesian_with_yaw_retry(robot, target_xyz, intermediate_xyz, base_rpy):
    """Descend joint motion (이전엔 cartesian native). IK 실패 시 등가 yaw 로 재시도.

    이전엔 Doosan native motion/move_line 으로 cartesian 직선 강하 → 컨트롤러 mix
    (ros2_control ↔ Doosan native) 전환 정지가 사이클 stutter 의 주 원인. 80mm 짧은
    descend 는 joint 보간 곡선이라도 옆 cube 안 침 → 컨트롤러 통일이 더 큰 이득.

    cube 가 정사각형 (4-fold 회전 대칭) + 그리퍼 평행조 (180° 대칭) → yaw, yaw±90°,
    yaw+180° 가 물리적으로 동일한 grasp. 원래 yaw 가 wrist singular / Doosan IK 한계로
    실패해도 등가 yaw 로 풀리는 경우가 많음.

    Returns: 최종 사용된 rpy_deg 리스트 (이후 close/lift 도 같은 yaw 써야 함).
    """
    # descend 직선은 DESCEND_DURATION_SEC 으로 느림 (잡기/놓기 정밀). intermediate 는 빠름.
    try:
        robot.move_line_base(target_xyz, rpy_deg=base_rpy, duration=DESCEND_DURATION_SEC)
        return list(base_rpy)
    except RuntimeError as e:
        if 'Ikin' not in str(e):
            raise
        print(f'    [ik-debug] base yaw {base_rpy[2]:+.1f}° → {e}')
    base_yaw = base_rpy[2]
    # Doosan IK 의 current-pose continuity bias 때문에 base_yaw 자세 직후 ±90°/+180°
    # 회전이 IK fail 하는 케이스가 있음. yaw=0 으로 wrist 를 한 번 align — place
    # 단계에서 검증된 자세라 거의 항상 풀림. 이후 alt yaw 회전이 짧아짐.
    try:
        robot.move_line_base(intermediate_xyz,
                             rpy_deg=[base_rpy[0], base_rpy[1], 0.0],
                             duration=MOVE_DURATION_SEC)
        print(f'    [retry] wrist yaw=0° align 완료 → alt yaw 시도')
    except RuntimeError as e_align:
        if 'Ikin' not in str(e_align):
            raise
        print(f'    [ik-debug] yaw=0 align 도 실패 → {e_align}')
    # 180° 우선 (j6 한 바퀴만 돌면 됨) → ±90° (cube 4-fold 대칭 활용).
    for delta in (180.0, 90.0, -90.0):
        alt_yaw = base_yaw + delta
        alt_rpy = [base_rpy[0], base_rpy[1], alt_yaw]
        print(f'    [retry] descend yaw {base_yaw:+.1f}° IK 실패 → 등가 yaw '
              f'{alt_yaw:+.1f}° 재시도 (cube 4-fold / jaw 180° 대칭 활용)')
        try:
            robot.move_line_base(intermediate_xyz, rpy_deg=alt_rpy, duration=MOVE_DURATION_SEC)
            robot.move_line_base(target_xyz, rpy_deg=alt_rpy, duration=DESCEND_DURATION_SEC)
            return alt_rpy
        except RuntimeError as e2:
            if 'Ikin' not in str(e2):
                raise
            print(f'    [ik-debug] alt yaw {alt_yaw:+.1f}° → {e2}')
            continue
    raise RuntimeError(
        f'descend 실패: 원 yaw={base_yaw:+.1f}° 와 모든 등가 yaw (±90°/+180°) 도 IK '
        f'실패. target={[round(v,1) for v in target_xyz]} 도달 불가 — 캘리브레이션/'
        f'그리퍼 tcp_z_offset 확인.'
    )


def execute_one_cycle(robot, target, cell_xy, args):
    """한 cube → 한 그리드 셀. cube top yaw 에 맞춰 그리퍼 회전 후 잡기 → 들어올린
    후 cube 와 함께 yaw=0 으로 회전 → place. 그리드 위 cube 모두 같은 각도로 정렬."""
    bx, by, bz = target['base_xyz_mm']
    yaw = target.get('cube_yaw_deg')
    yaw_used = float(yaw) if yaw is not None else 0.0
    yaw_str = f'{yaw_used:+.1f}°' if yaw is not None else 'no-yaw(0°)'

    # detect 가 측정한 cube 크기 (RGB+Depth+intrinsics). 비정상이면 default 25mm.
    # 정사각형 cube 전제 — w/h ratio 가 0.75~1.33 밖이거나 한 변이 22~30mm 밖이면 부정확.
    eff_w = target.get('cube_w_mm', CUBE_WIDTH_MM)
    eff_h = target.get('cube_h_mm', CUBE_WIDTH_MM)
    size_ok = (22.0 <= eff_w <= 30.0 and 22.0 <= eff_h <= 30.0
               and 0.75 <= eff_w / max(eff_h, 1.0) <= 1.33)
    if not size_ok:
        print(f'  ! cube size 측정 부정확 ({eff_w:.1f}x{eff_h:.1f}) → default {CUBE_WIDTH_MM}')
        eff_w = CUBE_WIDTH_MM
        eff_h = CUBE_WIDTH_MM
    print(f'\n  >> cube @ ({bx:.1f}, {by:.1f}, {bz:.1f}) yaw={yaw_str} '
          f'→ cell ({cell_xy[0]:.0f}, {cell_xy[1]:.0f})  size={eff_w:.1f}x{eff_h:.1f}mm')

    pick_rpy = [0.0, 180.0, yaw_used]                       # descent / lift 자세 = cube 변 정렬
    # 모든 cube 는 grid 에 yaw=0° 로 release — 다음 단계(탑쌓기 등) 가 axis-aligned 가정.
    place_rpy = [0.0, 180.0, 0.0]

    # pick/place 절대 z = FLOOR_Z + PICK_HEIGHT_ABOVE_FLOOR_MM.
    # 모든 cube 가 동일 크기(25mm) 라서 윗면 z 도 일정 — 검출 bz 마다 따로 계산할 필요 없이
    # 절대 좌표 PICK_Z 로 통일 (사용자 셋업, depth 노이즈/검출 오차도 흡수됨).
    pick_z = PICK_Z
    approach = [bx, by, pick_z + Z_APPROACH]
    pick = [bx, by, pick_z]
    lift = [bx, by, pick_z + Z_LIFT]

    place = [cell_xy[0], cell_xy[1], pick_z]
    # place 강하 시작 z = 바닥 + PLACE_APP_HEIGHT_ABOVE_FLOOR_MM (사용자 셋업, 일직선 강하).
    place_app = [cell_xy[0], cell_xy[1], PLACE_APP_Z]

    dur = MOVE_DURATION_SEC
    is_diagonal = bool(target.get('is_diagonal_grip', False))
    # 모든 open 동작 통일 폭 (45mm) — cube 25mm + 대각 35.4mm 둘 다 안 닿게 충분.
    pre_open_width = GRIPPER_OPEN_WIDTH_MM

    # ---- Pick: approach + descend 한 trajectory 로 (multi-WP, 사이 정지 없음) ----
    # 이전엔 approach 따로 + cartesian descend 따로 (컨트롤러 mix 정지). 모두 joint
    # 으로 통일하고 multi-WP 로 묶어서 한 ros2_control trajectory 안에 처리.
    # IK 실패 시 (예: descend 점 도달 불가) approach 만 가고 yaw retry 로 fallback.
    print('   [Pick]')
    print(f'    1) pre-open async ({pre_open_width:.0f}mm'
          f'{" — diagonal" if is_diagonal else ""}) + '
          f'approach→descend merged → {[round(v, 1) for v in pick]}')
    if not args.dry_run:
        robot.gripper_set(width_mm_to_pos(pre_open_width),
                          settle=0.0, label=f'PRE_OPEN_async_{pre_open_width:.0f}mm')
        try:
            # approach 는 빠른 dur, descend (pick) 직전은 DESCEND_DURATION_SEC 으로 느림 → 정밀 잡기.
            robot.move_line_base_multi([(approach, pick_rpy), (pick, pick_rpy)],
                                        [MOVE_DURATION_SEC, DESCEND_DURATION_SEC])
        except RuntimeError as e:
            if 'Ikin' not in str(e):
                raise
            print(f'    [merge-fallback] approach+descend IK fail → approach only + yaw retry')
            robot.move_line_base(approach, rpy_deg=pick_rpy, duration=dur)
            pick_rpy = _descend_cartesian_with_yaw_retry(robot, pick, approach, pick_rpy)
        yaw_used = pick_rpy[2]

    # 2) close (잡기) — 3단계 자가정렬:
    #    (a) half-close (28mm) + 0.3s 대기 → finger 가 cube 옆면 살짝 잡음
    #    (b) wiggle (yaw +N°) → 원위치: cube 가 finger 와 함께 회전해 옆면 평행 정렬
    #        IK 실패 시 wiggle skip (안전, 그냥 완전 close)
    #    (c) 완전 close + 1.0s settle → 강하게 잡고 lift 시작 보장
    print(f'    2) close: half ({HALF_CLOSE_WIDTH_MM:.0f}mm) → wiggle ±{WIGGLE_YAW_DEG:.0f}° → 완전 닫음')
    if not args.dry_run:
        gripper_set_width(robot, HALF_CLOSE_WIDTH_MM, settle=HALF_CLOSE_SETTLE_SEC)
        wiggle_rpy = [pick_rpy[0], pick_rpy[1], pick_rpy[2] + WIGGLE_YAW_DEG]
        try:
            robot.move_line_base(pick, rpy_deg=wiggle_rpy, duration=WIGGLE_DURATION_SEC)
            robot.move_line_base(pick, rpy_deg=pick_rpy, duration=WIGGLE_DURATION_SEC)
        except RuntimeError as e:
            if 'Ikin' not in str(e):
                raise
            print(f'    [wiggle] IK 실패 → 회전 skip, 완전 close 로 진행')
        robot.gripper_set(p12.GRIP_CLOSE_POS, settle=1.0, label='CLOSE')

    # ---- Transit (place_app 위까지) + 별도 place descend (수직 강하 보장) ----
    # transit + place 를 한 multi-WP 에 묶으면 segment blend 로 place_app → place 가
    # 사선이 됨. transit (multi) 와 place descend (단발 cartesian) 분리해서 수직 강제.
    transit = [(lift, pick_rpy)]
    if abs(yaw_used) > 0.5:
        transit.append((lift, place_rpy))          # 같은 XYZ 에서 yaw=0 align
    transit.append((place_app, place_rpy))           # target cell 위
    align_note = ' + align yaw→0' if abs(yaw_used) > 0.5 else ''
    print(f'    3a) transit (lift{align_note} → cell 위) — {len(transit)} WP')
    for wp, rp in transit:
        print(f'       · {[round(v, 1) for v in wp]} rpy={[round(v,1) for v in rp]}')
    print(f'    3b) place descend (수직 강하) → {[round(v, 1) for v in place]}')
    if not args.dry_run:
        # (a) Transit (multi)
        robot.move_line_base_multi(transit, [MOVE_DURATION_SEC] * len(transit))
        # (b) Place descend — 단발 cartesian, 수직 강하 보장 + 느린 정밀 속도
        try:
            robot.move_line_base(place, rpy_deg=place_rpy, duration=DESCEND_DURATION_SEC)
        except RuntimeError as e:
            if 'Ikin' not in str(e):
                raise
            print(f'    [place-fallback] place descend IK fail → yaw retry')
            place_rpy = _descend_cartesian_with_yaw_retry(robot, place, place_app, place_rpy)

    # ---- Place release + lift up ----
    # multi-WP descend 의 마지막 WP 라 종점 도착 보장됨 → 추가 sleep 불필요.
    # release 는 RELEASE_OPEN_WIDTH_MM (살짝만 벌림). open 후 0.5초 대기 — cube 안착.
    print(f'    4) release → {RELEASE_OPEN_WIDTH_MM:.0f}mm + 0.5s 대기')
    if not args.dry_run:
        gripper_set_width(robot, RELEASE_OPEN_WIDTH_MM)
        time.sleep(0.5)

    print(f'    5) lift up → {[round(v, 1) for v in place_app]}')
    if not args.dry_run:
        robot.move_line_base(place_app, rpy_deg=place_rpy, duration=dur)


# ===== main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='모션 없이 콘솔만')
    ap.add_argument('--quick', action='store_true',
                    help='드라이버 리셋 SKIP — dsr_bringup2 가 이미 살아있을 때만')
    ap.add_argument('--limit', type=int, default=None, help='처음 N개만 처리')
    ap.add_argument('--conf', type=float, default=DETECT_CONF_THR,
                    help=f'YOLO 검출 conf 임계 (default {DETECT_CONF_THR})')
    ap.add_argument('--debug-img', type=str, default='/tmp/cubes_debug.png')
    args = ap.parse_args()

    print('=== 바둑판 정렬 시작 ===')
    if args.dry_run:
        print('  *** DRY-RUN 모드: 실제 모션/그리퍼 명령은 보내지 않음 ***')
    grid_cells = make_grid_cells()
    print(f'  그리드: {GRID_ROWS}x{GRID_COLS} = {len(grid_cells)}셀, 간격 {GRID_SPACING}mm')
    print(f'  시작 셀: ({grid_cells[0][0]:.0f}, {grid_cells[0][1]:.0f}) → '
          f'끝 셀: ({grid_cells[-1][0]:.0f}, {grid_cells[-1][1]:.0f})')
    # 전체 25셀 좌표 명시 출력 — 17번/16번 이 잡으러 가는 좌표와 직접 대조용
    print('  ── 전체 grid cell 좌표 (mm, base frame) ──')
    for i, (x, y) in enumerate(grid_cells):
        r, c = i // GRID_COLS, i % GRID_COLS
        print(f'    cell[{i:2d}] row{r} col{c}: ({x:+7.1f}, {y:+7.1f})')

    if not args.quick:
        p12.reset_robot_driver()
    else:
        print('\n[준비 0/4] 드라이버 리셋 SKIP (--quick)')

    print(f'\n[준비 1/4] 비전 init (conf={args.conf})')
    vision = CubeDetector(conf_thr=args.conf)

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
                robot.move_joint_deg(p12.HOME_JOINT_DEG, duration=MOVE_DURATION_SEC)
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
        print(f'  확정 {len(dets)}개 (debug: {args.debug_img})')
        for i, d in enumerate(dets):
            x, y, z = d['base_xyz_mm']
            print(f'   #{i}: ({x:.1f}, {y:.1f}, {z:.1f}) mm  conf={d["conf"]:.2f}')

        # 작업영역 밖 cube 필터 — 15번 자체 한계 (12번 -600~600 보다 큰 셋업용)
        def _in_workspace(d):
            bx, by, _bz = d['base_xyz_mm']
            pz = PICK_Z
            az = pz + Z_APPROACH
            return (WORK_X[0] <= bx <= WORK_X[1] and
                    WORK_Y[0] <= by <= WORK_Y[1] and
                    WORK_Z[0] <= pz <= WORK_Z[1] and
                    WORK_Z[0] <= az <= WORK_Z[1])

        inside = [d for d in dets if _in_workspace(d)]
        skipped = len(dets) - len(inside)
        if skipped:
            print(f'  !! 작업영역 밖 cube {skipped}개 skip '
                  f'(X∈{WORK_X}, Y∈{WORK_Y}, Z∈{WORK_Z})')
            for d in dets:
                if not _in_workspace(d):
                    x, y, z = d['base_xyz_mm']
                    print(f'     skip: ({x:.1f}, {y:.1f}, {z:.1f}) mm')
        dets = inside

        dets = unify_z_with_outlier_reject(dets)

        if not dets:
            print('!! 잡을 수 있는 cube 0개 — 종료')
            return

        n_target = len(dets) if args.limit is None else min(args.limit, len(dets))
        n_target = min(n_target, len(grid_cells))
        print(f'\n  {n_target}개 cube → grid {n_target}개 셀 1:1 매핑 (Hungarian)')

        # --limit 안전성: 처리 안 되는 cube (n_target 이후 dets) 가 grid 셀 자리에 있으면
        # 우리가 place 할 때 그 cube 와 충돌. 사용자에게 경고 (자동 회피 안 함).
        if n_target < len(dets):
            unprocessed = dets[n_target:]
            blockers = []
            for d in unprocessed:
                bx, by, _ = d['base_xyz_mm']
                for ci, (cx, cy) in enumerate(grid_cells[:n_target]):
                    if math.hypot(bx - cx, by - cy) < CELL_OCCUPIED_RADIUS_MM:
                        blockers.append((d, ci))
                        break
            if blockers:
                print(f'\n  !! 경고: --limit 으로 처리 안 되는 cube 중 {len(blockers)}개가 '
                      f'우리가 사용할 grid 셀 자리에 있음. 충돌 위험.')
                for d, ci in blockers:
                    bx, by, _ = d['base_xyz_mm']
                    print(f'     ({bx:.0f}, {by:.0f}) — cell {ci} 자리. cube 치우거나 --limit 키우세요.')

        # cube ↔ cell 매핑 미리 결정 (거리 최소화). 옮겨야 할 위치에 cube 있으면
        # 그 cube 를 자기 target 으로 직접 — staging 없음.
        mapping = assign_cubes_to_cells(dets, grid_cells, n_target)
        for ci in range(n_target):
            bx, by, _ = dets[ci]['base_xyz_mm']
            cx, cy = grid_cells[mapping[ci]]
            print(f'   cube #{ci} ({bx:.0f}, {by:.0f}) → cell {mapping[ci]} ({cx:.0f}, {cy:.0f})')

        used = set()
        in_progress = set()  # 사이클 검출용

        def resolve_and_place(cube_idx, depth=0):
            """cube_idx 를 자기 target cell 로 옮김. target 에 다른 cube 있으면
            그 cube 를 자기 target 으로 먼저 옮김 (재귀)."""
            if cube_idx in used:
                return
            if cube_idx in in_progress:
                print(f'    !! cube #{cube_idx} 사이클 — 처리 불가 (staging 없이)')
                return
            in_progress.add(cube_idx)
            target_cell = grid_cells[mapping[cube_idx]]
            bx, by, _ = dets[cube_idx]['base_xyz_mm']
            d_to_cell = math.hypot(bx - target_cell[0], by - target_cell[1])
            # 이미 자기 target 에 정확히 정렬돼 있으면 옮기지 않음
            if d_to_cell < ALREADY_PLACED_RADIUS_MM:
                print(f'    cube #{cube_idx} 이미 cell {mapping[cube_idx]} 자리에 정렬됨 '
                      f'(거리 {d_to_cell:.1f}mm < {ALREADY_PLACED_RADIUS_MM:.0f}mm) — skip')
                used.add(cube_idx)
                in_progress.discard(cube_idx)
                return
            # target 자리에 다른 cube 가 있으면 그 cube 를 자기 target 으로 먼저
            blocking = find_cube_at_cell(dets, used, target_cell)
            if blocking is not None and blocking != cube_idx:
                bbx, bby, _ = dets[blocking]['base_xyz_mm']
                print(f'    cell {mapping[cube_idx]} 자리 blocking cube #{blocking} '
                      f'({bbx:.0f}, {bby:.0f}) — 먼저 cell {mapping[blocking]} 으로')
                resolve_and_place(blocking, depth + 1)
            # 이제 target_cell 이 비어있음 → cube_idx 이동
            # 인접 cube 회피: yaw 와 yaw+90 중 finger 공간 있는 쪽 선택
            target = dets[cube_idx]
            raw_yaw = target.get('cube_yaw_deg', 0.0) or 0.0
            safe_yaw, is_diagonal = pick_yaw_avoiding_neighbors(cube_idx, dets, raw_yaw, used)
            target_with_safe_yaw = {**target, 'cube_yaw_deg': safe_yaw,
                                     'is_diagonal_grip': is_diagonal}
            print(f'    cube #{cube_idx} → cell {mapping[cube_idx]} {target_cell}')
            try:
                execute_one_cycle(robot, target_with_safe_yaw, target_cell, args)
            except RuntimeError as exc:
                # 한 cube IK/모션 실패 → 그 cube 만 skip, 다음 cube 진행. HOME 복귀 보장.
                print(f'    !! cube #{cube_idx} 처리 실패 — skip 하고 계속: {exc}')
                if not args.dry_run:
                    # (a) Doosan firmware state 복구 — IK 실패 시 group=5 alarm 으로 SAFE_STOP
                    # 등 비-STANDBY 상태에 빠질 수 있음. 명시적으로 복구해서 다음 cube 의
                    # motion 명령이 silent reject 되지 않게 함.
                    try:
                        robot.recover_safety(verbose=False)
                    except Exception as e_rec:
                        print(f'    !! recover_safety 실패: {e_rec} — 무시하고 계속')
                    # (b) IK 캐시 초기화 — 실패한 cube 에서 마지막 성공 sol_space 가 캐시에
                    # 남아있으면 다음 cube IK 가 부적절한 분기를 우선 시도해서 실패율 ↑.
                    robot._last_ik_sol = None
                    # (c) 현재 위치에서 LIFT 높이로 안전 회피 (다음 cube approach 충돌 방지)
                    try:
                        bx_s, by_s, _bz_s = target['base_xyz_mm']
                        robot.move_line_base([bx_s, by_s, PICK_Z + Z_LIFT],
                                              rpy_deg=[0.0, 180.0, 0.0],
                                              duration=MOVE_DURATION_SEC)
                    except Exception as e_safe:
                        print(f'    !! 안전 lift 도 실패: {e_safe} — 그대로 진행')
                used.add(cube_idx)            # 재시도 막기
                in_progress.discard(cube_idx)
                return                          # dets 업데이트 안 함 (cube 안 옮겨짐)
            old = dets[cube_idx]
            dets[cube_idx] = {
                **old,
                'base_xyz_mm': (target_cell[0], target_cell[1], old['base_xyz_mm'][2]),
                'cube_yaw_deg': 0.0,
            }
            used.add(cube_idx)
            in_progress.discard(cube_idx)

        # cell 순서대로 처리 — 각 cell 의 target cube resolve
        for cell_idx in range(n_target):
            target_cube = None
            for ci in range(n_target):
                if mapping[ci] == cell_idx and ci not in used:
                    target_cube = ci
                    break
            if target_cube is None:
                continue
            print(f'\n  === [{cell_idx+1}/{n_target}] cell {grid_cells[cell_idx]} '
                  f'← cube #{target_cube} ===')
            resolve_and_place(target_cube)

        print('\n  HOME 복귀')
        if not args.dry_run:
            try:
                robot.move_joint_deg(p12.HOME_JOINT_DEG, duration=MOVE_DURATION_SEC)
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
