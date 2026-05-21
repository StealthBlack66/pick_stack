"""
큐브 각도 → 그리퍼 yaw 정렬 픽업 데모.

흐름:
  1) seg_v8_angle6 (15°×6 클래스 다중 클래스 seg) 로 큐브 검출
  2) 각 검출 마스크에서 cv2.minAreaRect 로 회전각 (base 좌표계 기준) 추출
  3) 가장 confidence 높은 큐브를 그 각도에 그리퍼 yaw 맞춰 픽업
  4) 원위치에서 X 방향 -80mm 떨어진 곳으로 옮겨놓기

사용:
  python3 22_각도_그리퍼_데모.py --dry-run    # 모션 없이 검출만 (각도 출력)
  python3 22_각도_그리퍼_데모.py              # 실제 동작
  python3 22_각도_그리퍼_데모.py --conf 0.3   # 검출 conf 임계

12번의 PickAndPlace (로봇/그리퍼) + 캘리브레이션을 importlib 로 재사용.
"""
import argparse
import importlib
import importlib.util
import math
import os
import queue
import sys
import threading
import time as _time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# 12_, 15_ 동적 로드 (한글 + 숫자시작 파일명)
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p15 = _load('p15', '15_바둑판_정렬.py')   # 검증된 pick/place 머신
p12 = p15.p12                              # 15_ 가 이미 12_ 로드함

# 모션을 천천히 + 부드럽게 — 15번 모듈 상수를 런타임에 덮어쓰기.
# execute_one_cycle 가 이 값을 사용함. 사용자 요청 "속도 느리게" → 5.0초/세그먼트.
p15.MOVE_DURATION_SEC = 5.0
# 12번의 multi-WP 도 같은 5.0 으로 (default 3.0 이라 path tolerance abort status=6 빈발)
p12.MOVE_DURATION_SEC = 5.0
p15.GRIPPER_SETTLE_SEC = 0.5
# Release 시 cube 바닥이 표면 +5mm 위에서 떨어지도록 (사용자 안전 요청).
p15.RELEASE_LIFT_MM = 5.0
# Z_LIFT 100mm 면 그리퍼 끝(wrist) 이 z=217mm 까지 올라가서 작업영역 밖 →
# 60mm 로 낮춰 wrist 가 approach(z=197) 근방에 머물게.
p15.Z_LIFT = 60.0


# ==== 설정 ====
ANGLE_MODEL = str(Path(__file__).parent /
                  'yolo_dataset/runs/seg_v8_angle18/weights/best.pt')
DETECT_CONF_THR = 0.30
RS_W, RS_H, RS_FPS = 1280, 720, 30
# 5° bins, 18 classes — 각 bin 중심: 2.5, 7.5, ..., 87.5°
CLASS_NAMES = [f'{i*5}-{(i+1)*5}' for i in range(18)]
BIN_DEG = 5

# 픽업 모션 — Z 값은 15번 모듈에서 가져와서 호환 유지
Z_APPROACH = p15.Z_APPROACH                 # 80.0
Z_LIFT = p15.Z_LIFT                          # 100.0
CUBE_WIDTH_MM = p15.CUBE_WIDTH_MM            # 25.0
# pre-open / release 모두 "잡힐 폭 + 클리어런스" — 단, 클리어런스는 별도.
#   잡힐 폭 = CUBE_WIDTH_MM (22번은 yaw 정렬 잡기 → edge grip).
#   pre-open  = CUBE_WIDTH_MM + PRE_OPEN_CLEARANCE_MM (= 40mm) — descend 중 cube 안 치게 여유 큼.
#   release   = CUBE_WIDTH_MM + RELEASE_CLEARANCE_MM  (= 27mm) — 살짝만 벌려서 부드럽게 안착.
PRE_OPEN_CLEARANCE_MM = 15.0  # 1.5cm
RELEASE_CLEARANCE_MM = 2.0    # 0.2cm

# 목적지 셀 (15번 grid 의 첫 행 — 여러 개 픽업 시 순서대로 사용)
PLACE_CELLS = [(350.0, -150.0), (400.0, -150.0), (450.0, -150.0),
               (500.0, -150.0), (550.0, -150.0),
               (350.0, -100.0), (400.0, -100.0), (450.0, -100.0),
               (500.0, -100.0), (550.0, -100.0)]
# HOME 자세
HOME_JOINTS = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]

# 두산 e0509 wrist singularity 회피: |yaw| 가 이 값보다 작으면 부호 유지하며
# bias 만큼 키워서 J4≈0 singular 영역을 피한다.
YAW_NEAR_ZERO_THR_DEG = 15.0
YAW_NEAR_ZERO_BIAS_DEG = 20.0

# pick_z 가 이 값보다 낮으면 (= 테이블 안쪽으로 들어감) 안전 거부.
# detect 평균 ~ -32mm 이라 -60mm 면 약 30mm 마진 — depth 노이즈 거른 정상 cube 만 통과.
PICK_Z_MIN_MM = -60.0


def avoid_wrist_singular(yaw: float) -> float:
    if abs(yaw) >= YAW_NEAR_ZERO_THR_DEG:
        return yaw
    return YAW_NEAR_ZERO_BIAS_DEG if yaw >= 0 else -YAW_NEAR_ZERO_BIAS_DEG


def check_ik(robot, x, y, z_user, yaw) -> bool:
    """pose [x,y,z,0,180,yaw] 의 IK 도달 가능 여부.

    중요: 12번의 move_line_* 가 사용자 z 에 self.tcp_z_offset(그리퍼 길이) 을
    더해서 wrist 좌표로 IK 호출. 사전체크도 동일하게 더해야 정확.
    """
    z_wrist = float(z_user) + float(getattr(robot, 'tcp_z_offset', 0.0))
    try:
        robot._ikin_with_fallback([float(x), float(y), z_wrist,
                                    0.0, 180.0, float(yaw)])
        return True
    except RuntimeError:
        return False


def assess_reachability(robot, det) -> tuple[bool, float]:
    """approach / pick / lift 3 pose 가 모두 IK 통과하는 yaw 를 찾는다.
    1) bias 적용된 yaw 로 시도 → 모두 OK면 그걸로 사용
    2) 안 되면 yaw=0 (회전 X) 로 fallback
    3) 둘 다 실패면 (False, 0)"""
    bx, by, bz = det['base_xyz_mm']
    yaw_raw = det['cube_yaw_deg']
    yaw_biased = avoid_wrist_singular(yaw_raw)
    pick_z = bz - CUBE_WIDTH_MM / 2.0
    # 안전: pick_z 가 표면보다 너무 깊으면 거부 (depth 노이즈로 인한 충돌 위험)
    if pick_z < PICK_Z_MIN_MM:
        return False, 0.0
    approach_z = pick_z + Z_APPROACH
    lift_z = pick_z + Z_LIFT

    for trial_yaw in (yaw_biased, 0.0):
        if (check_ik(robot, bx, by, approach_z, trial_yaw)
                and check_ik(robot, bx, by, pick_z, trial_yaw)
                and check_ik(robot, bx, by, lift_z, trial_yaw)):
            return True, trial_yaw
    return False, 0.0


def deproject_box_to_base_angle(box_px: np.ndarray,
                                z_m: float,
                                intr,
                                T_cam2base: np.ndarray) -> float:
    """이미지 평면 회전박스 4점 → base XY 평면에서 minAreaRect 각도.

    좌표계 차이로 이미지 각도 ≠ 실제 base 각도 (특히 카메라가 비스듬하면).
    그래서 box 4점을 depth 로 deproject → base frame 으로 변환 → 다시 minAreaRect.
    반환: [-45, 45) 범위 (큐브 90° 대칭).
    """
    bp_base = []
    for pt in box_px:
        c = rs.rs2_deproject_pixel_to_point(
            intr, [float(pt[0]), float(pt[1])], z_m)
        ch = np.array([c[0], c[1], c[2], 1.0])
        b = (T_cam2base @ ch)[:3] * 1000.0
        bp_base.append([b[0], b[1]])
    bp_base = np.array(bp_base, dtype=np.float32)
    rect = cv2.minAreaRect(bp_base)
    (_, _), (w, h), ang = rect
    if w < h:
        ang = ang + 90.0
    # ±45 로 wrap (그리퍼 회전 최소화)
    while ang > 45.0:
        ang -= 90.0
    while ang <= -45.0:
        ang += 90.0
    return float(ang)


class CubeAngleDetector:
    def __init__(self):
        if not os.path.exists(p12.CALIB_PATH):
            raise FileNotFoundError(f'캘리브레이션 없음: {p12.CALIB_PATH}')
        d = np.load(p12.CALIB_PATH)
        self.T_cam2base = d['T_cam2base']
        print(f'   캘리브 로드: translation='
              f'{[round(v*1000,1) for v in self.T_cam2base[:3,3]]} mm')

        if not os.path.exists(ANGLE_MODEL):
            raise FileNotFoundError(f'각도 모델 없음: {ANGLE_MODEL}')
        print(f'   YOLO 로드: {ANGLE_MODEL}')
        from ultralytics import YOLO
        self.model = YOLO(ANGLE_MODEL)

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
        cfg.enable_stream(rs.stream.depth, RS_W, RS_H, rs.format.z16, RS_FPS)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.intr = (profile.get_stream(rs.stream.color)
                     .as_video_stream_profile().get_intrinsics())
        # 자동 노출 워밍업
        for _ in range(15):
            self.pipeline.wait_for_frames()

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def _depth_m_at(self, depth_frame, u, v, patch=5):
        z = depth_frame.get_distance(int(u), int(v))
        if z > 0.05:
            return z
        arr = np.asanyarray(depth_frame.get_data())
        H, W = arr.shape
        u0, v0 = max(0, int(u) - patch), max(0, int(v) - patch)
        u1, v1 = min(W, int(u) + patch + 1), min(H, int(v) + patch + 1)
        win = arr[v0:v1, u0:u1]
        win = win[win > 0]
        if win.size == 0:
            return 0.0
        return float(win.mean()) * 0.001

    def detect(self, conf=DETECT_CONF_THR):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            return None, []
        bgr = np.asanyarray(color.get_data())
        # 어두운 가장자리 cube 검출률 향상 — gamma 1/1.3 으로 brighten.
        # 학습 데이터(밝은 종이 위)와 분포 가깝게 만들어 detection rate 개선.
        if not hasattr(self, '_gamma_lut'):
            self._gamma_lut = np.array(
                [((i / 255.0) ** (1.0 / 1.3)) * 255 for i in range(256)],
                dtype=np.uint8)
        bgr_eq = cv2.LUT(bgr, self._gamma_lut)
        # augment=True: YOLO TTA (flip+scale crop 평균) — 추론 ~4x 느림, 검출률 ↑.
        r = self.model.predict(bgr_eq, conf=conf, imgsz=640, verbose=False,
                               agnostic_nms=True, iou=0.5, augment=True)[0]
        if r.boxes is None or r.masks is None or len(r.boxes) == 0:
            return bgr, []

        boxes_xywh = r.boxes.xywh.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        masks = r.masks.data.cpu().numpy()
        H, W = bgr.shape[:2]

        depth_arr = np.asanyarray(depth.get_data())   # uint16, mm — mask 영역 median 용
        out = []
        for i in range(len(boxes_xywh)):
            mk = masks[i]
            if mk.shape != (H, W):
                mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
            m8 = (mk > 0.5).astype(np.uint8) * 255

            # 마스크 → 이미지 평면 minAreaRect → box 4점 (yaw 추정용)
            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            rect_px = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
            box_px = cv2.boxPoints(rect_px)

            # 사용자 요청: RealSense RGB(마스크) + Depth 결합으로 cube 중심 잡기
            #   ① 픽셀 좌표 = mask centroid (bbox center 보다 robust)
            #   ② Depth = mask 영역 픽셀 median (단일 픽셀 노이즈 제거)
            M = cv2.moments(m8)
            if M['m00'] < 1.0:
                continue
            cx_px = M['m10'] / M['m00']
            cy_px = M['m01'] / M['m00']

            # mask erode 로 cube 안쪽 (윗면) 만 남김 — 측면/그림자/경계 노이즈 제거
            m_inner = cv2.erode(m8, np.ones((7, 7), np.uint8), iterations=1)
            inner_mask = m_inner > 0
            if inner_mask.sum() < 20:
                inner_mask = m8 > 0   # erode 후 너무 작으면 원본 사용
            mvals = depth_arr[inner_mask]
            mvals = mvals[(mvals > 50) & (mvals < 2000)]   # 5~200cm 유효 범위
            if mvals.size == 0:
                z_m = self._depth_m_at(depth, cx_px, cy_px)   # fallback
            else:
                # cube 윗면 (= 카메라에 가장 가까운 = depth 가장 작은) 우선.
                # 가장 작은 15% pixel 만 mean → cube 윗면 안정적 측정 (mask 안쪽 erode 후).
                n_top = max(int(mvals.size * 0.15), 5)
                z_m = float(np.partition(mvals, n_top - 1)[:n_top].mean()) * 0.001
            if z_m < 0.05:
                continue

            # 카메라→베이스 center 변환
            c = rs.rs2_deproject_pixel_to_point(
                self.intr, [float(cx_px), float(cy_px)], z_m)
            ch = np.array([c[0], c[1], c[2], 1.0])
            b = (self.T_cam2base @ ch)[:3] * 1000.0

            # cube 가로/세로 mm 측정 — minAreaRect 픽셀 크기 + depth + intrinsics
            #   w_mm = w_px * z_m * 1000 / fx,   h_mm 동일 (fy)
            #   참고: cube 25mm 표준. 측정값이 30 이상이면 mask 가 측면 covers — 부정확.
            rect_w_px, rect_h_px = rect_px[1]
            cube_w_mm = float(rect_w_px) * z_m * 1000.0 / self.intr.fx
            cube_h_mm = float(rect_h_px) * z_m * 1000.0 / self.intr.fy

            # 박스 4점 base 평면 각도 (마스크 해상도 한계로 axis-aligned 로 snap 되는
            # 경향 있음 — 보조 정보로만 사용)
            yaw_mask_deg = deproject_box_to_base_angle(
                box_px, z_m, self.intr, self.T_cam2base)

            # 클래스 bin 중심을 yaw 의 주 신호로 사용.
            # BIN_DEG=5 → bin 0..17 (0~5°, 5~10°, ..., 85~90°) → 중심 2.5, 7.5, ..., 87.5°
            # 큐브 90° 대칭이므로 그리퍼는 ±45° 안에서 최단 회전:
            #   center <= 45 → 그대로,  > 45 → center - 90 (반대 방향 회전)
            cls = int(cls_ids[i])
            bin_center = (cls + 0.5) * BIN_DEG
            yaw_bin_deg = bin_center if bin_center <= 45.0 else bin_center - 90.0

            out.append({
                'pixel': (int(cx_px), int(cy_px)),
                'base_xyz_mm': (float(b[0]), float(b[1]), float(b[2])),
                'conf': float(confs[i]),
                'class_id': cls,
                'class_name': CLASS_NAMES[cls],
                'cube_yaw_deg': yaw_bin_deg,         # 그리퍼에 쓸 값
                'yaw_bin_center_deg': bin_center,    # 0~90 원본 bin 중심
                'yaw_mask_deg': yaw_mask_deg,        # 마스크에서 뽑은 각도 (참고용)
                'cube_w_mm': cube_w_mm,              # RGB+Depth+intrinsics 측정
                'cube_h_mm': cube_h_mm,              # 표준 cube 25x25mm 기준 검증용
            })
        # 작업 영역 필터 — 테이블 위 (z=-50~50mm) 안에 있는 큐브만.
        # 가장자리 false positive 가 depth 가 비어서 -800mm 같은 값이 나옴 → 폐기.
        out = [d for d in out if -50.0 <= d['base_xyz_mm'][2] <= 50.0
               and 100.0 <= d['base_xyz_mm'][0] <= 700.0
               and -300.0 <= d['base_xyz_mm'][1] <= 300.0]
        out.sort(key=lambda d: -d['conf'])
        return bgr, out

    def preview_live(self, conf=DETECT_CONF_THR, save_debug=None,
                     auto_after=None):
        """라이브 웹캠 + YOLO 검출 표시. ENTER/SPACE 로 그 시점 검출 확정,
        q/ESC 로 취소, s 로 스크린샷. auto_after 초 지나면 자동 확정."""
        import time as _t
        win = '22 각도 분류 라이브 (ENTER=확정 q=취소 s=screenshot)'
        # WINDOW_AUTOSIZE: frame 크기(1280x720) 그대로 윈도우 강제. resizeWindow 가
        # Qt 백엔드에서 무시되는 issue 회피.
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        if auto_after is not None:
            print(f'  preview 시작 — {auto_after:.0f}초 후 자동 확정 (ENTER 즉시)')
        else:
            print('  preview 시작 — ENTER 확정 / q 취소 / s 스크린샷')
        last_bgr, last_dets = None, []
        t_start = _t.time()
        try:
            while True:
                bgr, dets = self.detect(conf=conf)
                if bgr is None:
                    continue
                last_bgr, last_dets = bgr, dets
                vis = bgr.copy()
                for i, d in enumerate(dets):
                    u, v = d['pixel']
                    bx, by, bz = d['base_xyz_mm']
                    cv2.circle(vis, (u, v), 6, (0, 255, 0), 2)
                    # yaw 방향 시각화 (그리퍼 핑거 방향)
                    yaw_rad = math.radians(d['cube_yaw_deg'])
                    dx = int(40 * math.cos(yaw_rad))
                    dy = int(40 * math.sin(yaw_rad))
                    cv2.line(vis, (u - dx, v - dy), (u + dx, v + dy),
                             (0, 200, 255), 2)
                    label_lines = [
                        f"#{i} class={d['class_name']}° conf={d['conf']:.2f}",
                        f"yaw={d['cube_yaw_deg']:+.1f}° (mask={d['yaw_mask_deg']:+.1f}°)",
                        f"base=({bx:.0f},{by:.0f},{bz:.0f})mm",
                    ]
                    for k, line in enumerate(label_lines):
                        cv2.putText(vis, line, (u + 10, v - 10 + k * 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1)
                cv2.putText(vis, f'detected: {len(dets)}  (ENTER 확정 / q 취소)',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)
                cv2.imshow(win, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (13, 32):                        # ENTER / SPACE
                    break
                if key in (ord('q'), 27):                  # q / ESC
                    last_bgr, last_dets = None, []
                    break
                if key == ord('s') and save_debug:
                    cv2.imwrite(save_debug, vis)
                    print(f'  screenshot 저장: {save_debug}')
                if auto_after is not None and (_t.time() - t_start) >= auto_after:
                    print(f'  auto-confirm ({auto_after:.0f}s 경과) — {len(dets)}개')
                    break
        finally:
            cv2.destroyWindow(win)
        return last_bgr, last_dets


def render_live(bgr: np.ndarray, dets: list, status_msg: str = '') -> np.ndarray:
    """라이브 preview 시각화 — main loop 와 preview_live 가 공통 사용."""
    vis = bgr.copy()
    for i, d in enumerate(dets):
        u, v = d['pixel']
        bx, by, bz = d['base_xyz_mm']
        cv2.circle(vis, (u, v), 6, (0, 255, 0), 2)
        yaw_rad = math.radians(d['cube_yaw_deg'])
        dx = int(40 * math.cos(yaw_rad))
        dy = int(40 * math.sin(yaw_rad))
        cv2.line(vis, (u - dx, v - dy), (u + dx, v + dy),
                 (0, 200, 255), 2)
        cw = d.get('cube_w_mm', 0.0)
        ch_m = d.get('cube_h_mm', 0.0)
        label_lines = [
            f"#{i} class={d['class_name']}° conf={d['conf']:.2f}",
            f"yaw={d['cube_yaw_deg']:+.1f}° (mask={d['yaw_mask_deg']:+.1f}°)",
            f"base=({bx:.0f},{by:.0f},{bz:.0f})mm",
            f"size={cw:.1f}x{ch_m:.1f}mm (target 25x25)",
        ]
        for k, line in enumerate(label_lines):
            cv2.putText(vis, line, (u + 10, v - 10 + k * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    hdr = f'detected: {len(dets)}'
    if status_msg:
        hdr += f'   |   {status_msg}'
    cv2.putText(vis, hdr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)
    return vis


def draw_debug(bgr: np.ndarray, dets: list, out_path: str) -> None:
    if bgr is None:
        return
    img = bgr.copy()
    for i, d in enumerate(dets):
        u, v = d['pixel']
        bx, by, bz = d['base_xyz_mm']
        label = (f"#{i} {d['class_name']}° | yaw={d['cube_yaw_deg']:+.1f}° "
                 f"conf={d['conf']:.2f}\n"
                 f"base=({bx:.0f},{by:.0f},{bz:.0f})mm")
        cv2.circle(img, (u, v), 6, (0, 255, 0), 2)
        for k, line in enumerate(label.split('\n')):
            cv2.putText(img, line, (u + 10, v - 10 + k * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(out_path, img)
    print(f'   debug image: {out_path}')


PLACE_OCCUPIED_THR_MM = 40.0   # cube 중심이 cell 중심에서 이 거리 이내면 점유로 판정.
# 큐브 25mm 라 25mm 임계는 너무 작음 — 옆에 있는 cube 도 포함하도록 40mm.


def find_free_cell(dets, pick_cand, start_idx=0):
    """PLACE_CELLS 중 점유되지 않은 첫 cell index 반환. None 이면 모두 점유.
    pick_cand 는 픽업할 큐브라 점유 판정에서 제외 (어차피 픽업되니까)."""
    occ_xy = [d['base_xyz_mm'][:2] for d in dets if d is not pick_cand]
    n = len(PLACE_CELLS)
    for i in range(n):
        idx = (start_idx + i) % n
        cx, cy = PLACE_CELLS[idx]
        occupied = any((cx - ox) ** 2 + (cy - oy) ** 2 <
                       PLACE_OCCUPIED_THR_MM ** 2 for ox, oy in occ_xy)
        if not occupied:
            return idx
    return None


def _reapply_overrides():
    """p15 / p12 reload 후 22.py 의 런타임 override 다시 적용."""
    p15.MOVE_DURATION_SEC = 5.0
    p15.GRIPPER_SETTLE_SEC = 0.5
    p15.Z_LIFT = 60.0
    p12.MOVE_DURATION_SEC = 5.0
    # p15 reload 시 p15 안의 p12 참조도 새 모듈로 바뀌므로 22.py 와 통일.
    p15.p12 = p12


def _do_one_pick(robot, cand, yaw_use, cell, args):
    """① J6 yaw 회전 → ② execute_one_cycle → ③ HOME 복귀.
    사용자 요청: 강하 전에 그리퍼 yaw 가 먼저 회전되어있어야 함."""
    bx, by, bz = cand['base_xyz_mm']
    print(f'\n  [Pick] cube=({bx:.0f},{by:.0f},{bz:.0f}) '
          f'yaw={yaw_use:+.1f}° cell={cell}')

    # ① yaw 회전 단독 (HOME 자세에서 J6 만)
    print(f'    1) yaw 회전: J6 → {yaw_use:+.1f}° (강하 전 정렬)')
    rot_home = list(HOME_JOINTS)
    rot_home[5] = yaw_use
    try:
        robot.move_joint_deg(rot_home, duration=2.0)
    except Exception as e:
        print(f'    !! yaw 회전 실패: {e}')

    # ② execute_one_cycle — multi-WP pick + transit + place
    target = {**cand, 'cube_yaw_deg': yaw_use}
    success = False
    try:
        p15.execute_one_cycle(robot, target, cell, args)
        success = True
    except RuntimeError as e:
        print(f'    !! 픽업 실패 — gripper_open 후 HOME 으로\n       원인: {e}')
        try:
            robot.gripper_open()        # 큐브 잡고 있을 수 있으니 강제 release
        except Exception:
            pass

    # ③ HOME 복귀
    try:
        robot.move_joint_deg(HOME_JOINTS, duration=2.0)
    except Exception:
        pass
    return success


def capture_loop(detector, conf, shared, lock, quit_evt):
    """RealSense + YOLO detect 무한 loop — capture thread.
    ROS 안 만짐 (메인의 cv2.imshow + worker 의 ROS service 와 충돌 없음).
    매 frame 결과를 shared dict 에 갱신 → 메인이 lock 으로 읽기."""
    while not quit_evt.is_set():
        try:
            bgr, dets = detector.detect(conf=conf)
        except Exception as e:
            print(f'  capture error: {e}')
            _time.sleep(0.05)
            continue
        if bgr is not None:
            with lock:
                shared['bgr'] = bgr
                shared['dets'] = dets


def pick_worker(robot, args, pick_q, status, status_lock,
                reload_req, quit_evt, pick_count_ref):
    """SPACE 로 enqueue 된 dets snapshot 받아서 픽업 1회 수행.
    모든 ROS service call 이 이 thread 안에서만 발생 → 메인의 cv2.imshow 와
    충돌 없음. rclpy.spin_until_future_complete 는 호출자 thread 에서 spin."""
    while not quit_evt.is_set():
        try:
            dets_snapshot = pick_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if dets_snapshot is None:
            break
        # reload 요청 처리 (다음 픽업 직전에 반영)
        if reload_req.is_set():
            try:
                importlib.reload(p15)
                _reapply_overrides()
                print('  [reload] p15 새로 로드 — override 재적용')
            except Exception as e:
                print(f'  [reload] 실패: {e}')
            reload_req.clear()
        # IK OK 인 첫 cube (conf 높은 순)
        cand, yaw_use = None, 0.0
        for d in dets_snapshot:
            ok, y = assess_reachability(robot, d)
            if ok:
                cand, yaw_use = d, y
                break
        if cand is None:
            with status_lock:
                status['msg'] = 'IK OK cube 없음 (pick z sanity 거부 포함)'
            continue
        free_idx = find_free_cell(dets_snapshot, cand,
                                   start_idx=pick_count_ref[0])
        if free_idx is None:
            with status_lock:
                status['msg'] = '모든 place cell 점유 — 비울 cell 없음'
            continue
        cell = PLACE_CELLS[free_idx]
        bx, by, _ = cand['base_xyz_mm']
        with status_lock:
            status['msg'] = (f'picking ({bx:.0f},{by:.0f}) yaw={yaw_use:+.1f}° '
                             f'→ cell[{free_idx}]')
        ok = _do_one_pick(robot, cand, yaw_use, cell, args)
        if ok:
            pick_count_ref[0] += 1
            with status_lock:
                status['msg'] = f'✓ {pick_count_ref[0]} 완료 (다음 SPACE)'
        else:
            with status_lock:
                status['msg'] = 'fail — 다음 SPACE 로 재시도'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='로봇 모션 SKIP — 검출/각도만 표시')
    ap.add_argument('--quick', action='store_true',
                    help='dsr_bringup2 reset SKIP — bringup 이 이미 살아있을 때만')
    ap.add_argument('--conf', type=float, default=DETECT_CONF_THR)
    ap.add_argument('--debug-img', type=str, default='/tmp/angle_demo_debug.png')
    args = ap.parse_args()

    if not args.dry_run and not args.quick:
        p12.reset_robot_driver()

    print('\n[1/3] 비전 init')
    detector = CubeAngleDetector()

    robot = None
    if not args.dry_run:
        print('[2/3] 로봇 + 그리퍼 init')
        import rclpy
        rclpy.init()
        robot = p12.PickAndPlace()
        robot.recover_safety()
        robot.activate_robot()
        robot.gripper_init()
        # tcp_z_offset 보정 — default 160 이면 finger 가 cube 옆 못 닿고 cube 위에
        # 떨어짐 (사용자 보고). 실측 기반 147.5 (12.5mm 줄임 = cube width/2).
        robot.tcp_z_offset = 147.5
        print(f'   tcp_z_offset override → {robot.tcp_z_offset}mm '
              f'(default 160 → finger 가 cube 옆 잡기)')
        print('[3/3] HOME 자세로 이동')
        robot.move_joint_deg(HOME_JOINTS, duration=2.5)
    else:
        print('[2/3] dry-run — robot init skip')
        print('[3/3] dry-run — HOME skip')

    _reapply_overrides()

    print('\n=== 라이브 픽업 모드 (멀티스레드 — preview 안 멈춤) ===')
    print('  SPACE/ENTER  =  최고 confidence + IK OK cube 1개 픽업 큐 등록')
    print('  r            =  p15 hot-reload (다음 픽업 직전 반영)')
    print('  s            =  현재 frame 디스크 저장')
    print('  q/ESC        =  종료')
    print()

    win = '22 라이브 픽업 (SPACE=pick r=reload s=save q=quit)'
    # WINDOW_AUTOSIZE: frame 크기 그대로 강제 (WINDOW_NORMAL+resizeWindow 는 Qt 무시).
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    # ── 스레드 공유 자원 ──
    shared = {'bgr': None, 'dets': []}
    shared_lock = threading.Lock()
    status = {'msg': 'ready'}
    status_lock = threading.Lock()
    pick_q = queue.Queue(maxsize=1)
    quit_evt = threading.Event()
    reload_req = threading.Event()
    pick_count_ref = [0]

    # capture thread — 항상 라이브
    cap_thr = threading.Thread(
        target=capture_loop, daemon=True,
        args=(detector, args.conf, shared, shared_lock, quit_evt))
    cap_thr.start()

    # pick worker — 모든 ROS service call 담당 (dry-run 이면 skip)
    pick_thr = None
    if not args.dry_run:
        pick_thr = threading.Thread(
            target=pick_worker, daemon=True,
            args=(robot, args, pick_q, status, status_lock,
                  reload_req, quit_evt, pick_count_ref))
        pick_thr.start()

    try:
        while not quit_evt.is_set():
            with shared_lock:
                bgr = shared['bgr']
                dets = list(shared['dets'])
            with status_lock:
                msg = status['msg']
            if bgr is None:
                cv2.waitKey(30)
                continue
            vis = render_live(bgr, dets,
                              f'{msg}   picks={pick_count_ref[0]}')
            cv2.imshow(win, vis)
            # waitKeyEx — & 0xFF 안 거치고 full keycode 반환 (F3 같은 function key 인식 위해).
            key = cv2.waitKeyEx(30)

            if key in (ord('q'), 27):
                quit_evt.set()
                break
            # F3 또는 'f' 단축키 — Claude 가 자동으로 읽도록 fixed path 에 share.
            # F3 keycode 가 OpenCV/Qt 환경마다 달라 후보 + 'f' alias.
            F3_CANDIDATES = (0x70, 200, 7340035, 65472, 0xFFC0)
            if key in F3_CANDIDATES or key in (ord('f'), ord('F')):
                share_path = '/tmp/22_share.png'
                cv2.imwrite(share_path, vis)
                print(f'[F3/f share] saved: {share_path}')
                with status_lock:
                    status['msg'] = f'shared → {share_path}'
                continue
            # 알 수 없는 key — F3 진짜 keycode 디버그용 (waitKeyEx 라 no-key = -1)
            if key != -1 and key not in (
                    ord('q'), 27, ord('s'), ord('r'),
                    13, 32, ord('f'), ord('F'),
                    *F3_CANDIDATES):
                print(f'  [key debug] keycode={key} (0x{key:x})')
            if key == ord('s'):
                cv2.imwrite(args.debug_img, vis)
                print(f'  screenshot 저장: {args.debug_img}')
                with status_lock:
                    status['msg'] = f'saved → {args.debug_img}'
            elif key == ord('r'):
                reload_req.set()
                with status_lock:
                    status['msg'] = 'reload 요청됨 (다음 픽업에 적용)'
            elif key in (13, 32):  # ENTER / SPACE
                if args.dry_run:
                    with status_lock:
                        status['msg'] = 'dry-run — pick skip'
                    continue
                if pick_q.full():
                    with status_lock:
                        status['msg'] = '이미 픽업 진행 중'
                    continue
                if not dets:
                    with status_lock:
                        status['msg'] = '검출 없음 — pick skip'
                    continue
                pick_q.put(dets)
                with status_lock:
                    status['msg'] = '픽업 큐 등록 → worker 처리 중...'

        print(f'\n총 픽업: {pick_count_ref[0]}')
    finally:
        quit_evt.set()
        if pick_thr is not None:
            try: pick_q.put_nowait(None)
            except queue.Full: pass
            pick_thr.join(timeout=5.0)
        cap_thr.join(timeout=2.0)
        cv2.destroyAllWindows()
        detector.stop()
        if robot is not None:
            try:
                robot.gripper_open()
                robot.gripper_shutdown()
            except Exception:
                pass
            import rclpy
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    main()
