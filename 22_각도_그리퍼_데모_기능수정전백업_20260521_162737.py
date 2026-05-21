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
import importlib.util
import math
import os
import sys
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
        r = self.model.predict(bgr, conf=conf, imgsz=640, verbose=False,
                               agnostic_nms=True, iou=0.5)[0]
        if r.boxes is None or r.masks is None or len(r.boxes) == 0:
            return bgr, []

        boxes_xywh = r.boxes.xywh.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        masks = r.masks.data.cpu().numpy()
        H, W = bgr.shape[:2]

        out = []
        for i in range(len(boxes_xywh)):
            cx_px, cy_px, _, _ = boxes_xywh[i]
            mk = masks[i]
            if mk.shape != (H, W):
                mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)

            # 마스크 → 이미지 평면 minAreaRect → box 4점
            m8 = (mk > 0.5).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            rect_px = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
            box_px = cv2.boxPoints(rect_px)

            # 큐브 중심의 깊이
            z_m = self._depth_m_at(depth, cx_px, cy_px)
            if z_m < 0.05:
                continue

            # 카메라→베이스 center 변환
            c = rs.rs2_deproject_pixel_to_point(
                self.intr, [float(cx_px), float(cy_px)], z_m)
            ch = np.array([c[0], c[1], c[2], 1.0])
            b = (self.T_cam2base @ ch)[:3] * 1000.0

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
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, RS_W // 2, RS_H // 2)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='로봇 모션 SKIP, 검출/각도만 출력')
    ap.add_argument('--quick', action='store_true',
                    help='두산 드라이버 reset SKIP — bringup 이 이미 살아있을 때만')
    ap.add_argument('--conf', type=float, default=DETECT_CONF_THR)
    ap.add_argument('--limit', type=int, default=None,
                    help='몇 개 큐브 처리할지 (default 무제한 — 검출 + IK OK 인 모든 cube)')
    ap.add_argument('--auto-after', type=float, default=None,
                    help='preview N초 후 자동 확정 (ENTER 입력 없이)')
    ap.add_argument('--debug-img', type=str, default='/tmp/angle_demo_debug.png')
    args = ap.parse_args()

    if not args.dry_run and not args.quick:
        p12.reset_robot_driver()      # dsr_bringup2 새 터미널로 띄움 + 안정화 대기

    print('\n[1/4] 비전 init')
    detector = CubeAngleDetector()

    robot = None
    if not args.dry_run:
        print('[2/4] 로봇 + 그리퍼 init')
        import rclpy
        rclpy.init()
        robot = p12.PickAndPlace()
        robot.recover_safety()
        robot.activate_robot()
        robot.gripper_init()
        print('[3/4] HOME 자세로 이동')
        robot.move_joint_deg(HOME_JOINTS, duration=2.5)
    else:
        print('[2/4] dry-run — robot init skip')
        print('[3/4] dry-run — HOME skip')

    try:
        print('[4/4] 큐브 라이브 검출 — ENTER 로 그 시점 결과 확정')
        bgr, dets = detector.preview_live(conf=args.conf, save_debug=args.debug_img,
                                          auto_after=args.auto_after)
        if bgr is None:
            print('!! 사용자 취소 — 종료')
            return
        print(f'\n확정된 검출: {len(dets)}개')
        for i, d in enumerate(dets):
            print(f'  [{i}] px={d["pixel"]}  '
                  f'base=({d["base_xyz_mm"][0]:.1f},'
                  f'{d["base_xyz_mm"][1]:.1f},'
                  f'{d["base_xyz_mm"][2]:.1f})mm  '
                  f'class={d["class_name"]}°  '
                  f'yaw={d["cube_yaw_deg"]:+.1f}° (bin {d["yaw_bin_center_deg"]:.1f}°, '
                  f'mask {d["yaw_mask_deg"]:+.1f}°)  '
                  f'conf={d["conf"]:.2f}')

        draw_debug(bgr, dets, args.debug_img)

        if args.dry_run or not dets:
            if not dets:
                print('!! 검출 없음 — 종료')
            return

        # IK 사전 체크 — approach/pick/lift 3 pose 통과하는 큐브 + yaw 찾기
        print(f'\n[IK 사전 체크] {len(dets)}개 cube')
        candidates = []
        for d in dets:
            ok, yaw_use = assess_reachability(robot, d)
            mark = 'OK' if ok else 'FAIL'
            print(f'  base=({d["base_xyz_mm"][0]:.0f},{d["base_xyz_mm"][1]:.0f}) '
                  f'class={d["class_name"]}°  yaw={d["cube_yaw_deg"]:+.1f}° → '
                  f'use {yaw_use:+.1f}°  IK {mark}')
            if ok:
                candidates.append((d, yaw_use))

        if not candidates:
            print('\n!! 모든 cube 도달 불가 — HOME 복귀 후 종료')
            robot.move_joint_deg(HOME_JOINTS, duration=2.5)
            return

        candidates.sort(key=lambda x: -x[0]['conf'])
        picked = 0
        for d, yaw_use in candidates:
            if args.limit is not None and picked >= args.limit:
                break
            cell = PLACE_CELLS[picked % len(PLACE_CELLS)]
            # 15번 execute_one_cycle 재사용 — joint multi-WP 통합 pick&place 흐름.
            # 22번이 assess_reachability 로 결정한 yaw 를 cube_yaw_deg 로 덮어씀.
            target = {**d, 'cube_yaw_deg': yaw_use}
            try:
                p15.execute_one_cycle(robot, target, cell, args)
                picked += 1
                print(f'\n   ✓ {picked}/{args.limit} 완료\n')
            except RuntimeError as e:
                print(f'\n!! 픽업 실패 (yaw={yaw_use:+.1f}°) — '
                      f'HOME 복귀 후 다음 후보로\n   원인: {e}')
                try:
                    robot.move_joint_deg(HOME_JOINTS, duration=2.5)
                except Exception:
                    pass
                continue
        print(f'\n총 픽업: {picked}/{args.limit}')
        if picked == 0:
            print('!! 모든 후보 큐브 픽업 실패')

        print('\n[HOME 복귀]')
        robot.move_joint_deg(HOME_JOINTS, duration=2.5)
        print('\n=== 완료 ===')
    finally:
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
