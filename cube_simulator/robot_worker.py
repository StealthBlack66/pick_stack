"""로봇 실행 워커 (QThread).

GUI 메인 스레드와 분리하여 rclpy + p12.PickAndPlace + p16.execute_stack_pick_place
호출 흐름을 그대로 수행. 17_미술쌓기.build_art 로직을 GUI 가 만든 layout
으로 옮긴 것.
"""
from __future__ import annotations

import importlib.util
import io
import math
import sys
import threading
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

from . import MODULE_DIR, MODULE_PATHS
from .model import PlacedCube


_DIR = MODULE_DIR  # 하위 호환 — 기존 _DIR 참조 유지


def _load_module(name: str, filename: Optional[str] = None):
    """한글/숫자 시작 파일을 importlib 로 로드.

    name 만 주면 cube_simulator.MODULE_PATHS 에서 경로 lookup.
    filename 인자는 하위 호환 — 명시 시 그것을 우선 사용.
    """
    if name in sys.modules:
        return sys.modules[name]
    if filename is not None:
        path = MODULE_DIR / filename
    else:
        path = MODULE_PATHS.get(name)
        if path is None:
            raise KeyError(
                f'unknown module key {name!r} — MODULE_PATHS 에 등록 필요'
            )
    sys.path.insert(0, str(MODULE_DIR))
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _LogStream:
    """sys.stdout 대체 — 한 줄 단위로 콜백(emit_fn) 호출."""

    def __init__(self, emit_fn):
        self._emit = emit_fn
        self._buf = ''

    def write(self, s: str):
        if not isinstance(s, str):
            s = str(s)
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            try:
                self._emit(line.rstrip('\r'))
            except Exception:
                pass
        return len(s)

    def flush(self):
        if self._buf:
            try:
                self._emit(self._buf)
            except Exception:
                pass
            self._buf = ''

    def isatty(self) -> bool:
        return False


# --- 세션 (rclpy + PickAndPlace 노드 재사용) ---------------------------------
# 매 실행마다 노드를 만들고 부수면 'vision_pick_and_place' 같은 고정 이름의
# rosout publisher 가 반복 등록되어 [WARN] Publisher already registered ... 가
# 발생한다. GUI 한 세션 동안 노드 1개만 만들어 재사용.

@dataclass
class RobotSession:
    p12: object
    p15: object
    p16: object
    robot: object  # PickAndPlace | None (dry-run 시 None)
    activated: bool = False
    gripper_inited: bool = False

    def shutdown(self) -> None:
        if self.robot is not None:
            try:
                self.robot.destroy_node()
            except Exception:
                pass
        try:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


class SessionManager:
    """rclpy 세션 + dsr_bringup2 subprocess 한 묶음의 lifecycle 관리자.

    이전엔 모듈 전역 `_SESSION`/`_BRINGUP_PROC`/`_BRINGUP_LOG_FN` 였던 것을
    인스턴스 상태로. 앱 1회 실행 = manager 1개 (default_manager() 싱글톤).
    """

    def __init__(self) -> None:
        self._session: Optional[RobotSession] = None
        self._session_lock = threading.Lock()
        self.bringup_proc = None  # subprocess.Popen | None  (외부 read 필요)
        self._bringup_lock = threading.Lock()
        self.bringup_log_fn = None  # 가장 최근 워커가 설정한 emit 콜백

    @property
    def session_lock(self):
        return self._session_lock

    @property
    def bringup_lock(self):
        return self._bringup_lock

    @property
    def session(self) -> Optional[RobotSession]:
        return self._session

    def get_or_create_session(self, log_fn) -> RobotSession:
        """세션 없으면 생성, 있으면 그대로. caller 가 session_lock 안에서 호출.

        log_fn(str) — 진행 로그를 GUI 로 보낼 콜백.
        """
        if self._session is None:
            log_fn('세션 초기화 — PickAndPlace 노드 생성 중...')
            self._session = self._create_session()
        else:
            log_fn('세션 재사용 (노드 재생성 없음)')
        return self._session

    @staticmethod
    def _create_session() -> RobotSession:
        import rclpy
        if not rclpy.ok():
            rclpy.init()
        p12 = _load_module('p12')
        p15 = _load_module('p15')
        p16 = _load_module('p16')
        robot = p12.PickAndPlace()
        return RobotSession(p12=p12, p15=p15, p16=p16, robot=robot)

    @staticmethod
    def get_dry_session(log_fn) -> RobotSession:
        """dry-run 전용 — 모듈만 로드, rclpy 노드 안 만듬.

        실 모드 진입 시 stale session 으로 이어지는 것을 막기 위해 cache 안 함.
        호출자는 매번 새로 만들고 끝나면 폐기.
        """
        log_fn('(dry-run) 모듈 로드만 — rclpy 노드 미생성')
        p12 = _load_module('p12')
        p15 = _load_module('p15')
        p16 = _load_module('p16')
        return RobotSession(p12=p12, p15=p15, p16=p16, robot=None)

    def teardown_session(self) -> None:
        """앱 종료 / 정렬 모드 전 1회 호출."""
        with self._session_lock:
            if self._session is not None:
                self._session.shutdown()
                self._session = None

    def teardown_bringup(self) -> None:
        """앱 종료 시 dsr_bringup2 프로세스 그룹 종료."""
        with self._bringup_lock:
            proc = self.bringup_proc
            self.bringup_proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            import os as _os
            import signal as _signal
            _os.killpg(_os.getpgid(proc.pid), _signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


# --- module-level default singleton + backward-compat wrappers ---------------
# main_window.closeEvent 가 from .robot_worker import teardown_session,
# teardown_bringup 하는 기존 호출 그대로 유지.

_DEFAULT_MANAGER: Optional[SessionManager] = None
_DEFAULT_MANAGER_LOCK = threading.Lock()


def default_manager() -> SessionManager:
    """앱 한 개 = manager 한 개. 첫 호출 시 생성."""
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        with _DEFAULT_MANAGER_LOCK:
            if _DEFAULT_MANAGER is None:
                _DEFAULT_MANAGER = SessionManager()
    return _DEFAULT_MANAGER


def teardown_bringup() -> None:
    default_manager().teardown_bringup()


def teardown_session() -> None:
    default_manager().teardown_session()


@dataclass
class RunRequest:
    """[로봇 실행] 1회 요청.

    mode='stack' : 큐브 쌓기 (build_art 흐름)
    mode='align' : 바둑판 정렬 (15번 main 흐름)
    """
    layout: list[PlacedCube]
    base_xy: tuple[float, float]
    pitch_mm: float
    cube_width_mm: float
    dry_run: bool = False
    dets: Optional[list[dict]] = None
    vision_conf_thr: float = 0.40
    skip_activate: bool = False
    pre_open_width_mm: float = 40.0
    place_pre_open_width_mm: float = 27.0
    mode: str = 'stack'                  # 'stack' | 'align' | 'recover' | 'calibrate'
    align_limit: Optional[int] = None    # align 모드 — 처음 N개만 정렬
    align_quick: bool = True             # --quick (드라이버 리셋 SKIP)
    # 캘리브레이션 모드 옵션 — 09_원샷_캘리브레이션.py 의 CLI flag 와 매핑
    cal_skip_launch: bool = False        # 09 --skip-launch (bringup 이미 떠있으면)
    cal_no_cleanup: bool = False         # 09 --no-cleanup (위험)
    cal_keep_bringup: bool = True        # 09 --keep-bringup (RViz 계속 사용)
    # 사용자가 표에서 확정한 plan. None 이면 dets/grid_cells 매칭으로 자동.
    # 주어지면 src/tgt 의 x/y/yaw 는 plan 그대로, z 만 sample_z 기반 robot 프레임 계산.
    plan: Optional[list] = None          # list[motion_plan.PlanItem]
    # 큐브 내려놓을 때 target_z 에 더할 offset (mm). 음수 = 더 깊이 내려감.
    # 기본 -10.0 — 실측 결과 큐브 top 기준 sample_z 가 약 10mm 높게 잡혀
    # placement 후 큐브가 공중에 떠 있는 현상 보정.
    place_z_offset_mm: float = -10.0
    # 큐브를 집을 때 source z (sample_z) 에 더할 offset (mm).
    # 기본 0 = p16 의 원래 동작 (pick_z = sz - cube_w/2, cube 중심까지 = 12.5mm 깊이).
    # 음수 = 더 깊이 (큐브 미끄러질 때만 -3 ~ -5 정도). bottom 충돌 방지 clamp 됨.
    pick_z_offset_mm: float = 0.0
    # 실제 테이블 표면 z (mm, robot base frame). None 이면 sample_z - cube_w 로 자동 추정.
    # 비전 sample_z 가 불안정하거나 잘못 잡힐 때 사용자가 실측치 입력 (예: 캘리브).
    z_table_top_override_mm: Optional[float] = None
    # finger tip 이 테이블 표면 위에 유지해야 할 최소 여유 (mm). 안전 floor.
    pick_min_clearance_mm: float = 5.0


class RobotWorker(QThread):
    """1 회의 RunRequest 를 처리하고 종료."""

    progress = pyqtSignal(int, int, str)        # (current_idx, total, msg)
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    started_run = pyqtSignal()
    finished_run = pyqtSignal(bool)             # True = 정상 완료
    finished_align = pyqtSignal(bool, object)   # ok, dets (정렬 모드 전용)

    def __init__(self,
                 req: RunRequest,
                 session_manager: Optional[SessionManager] = None,
                 parent=None):
        super().__init__(parent)
        self._req = req
        self._estop = False
        self._mutex = QMutex()
        self._sm = session_manager if session_manager is not None else default_manager()

    def request_estop(self) -> None:
        with QMutexLocker(self._mutex):
            self._estop = True

    def _check_estop(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._estop

    # --- main -----------------------------------------------------------------
    def run(self) -> None:  # noqa: C901
        self.started_run.emit()
        try:
            self._run_impl()
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'예외: {e!r}')
            self.finished_run.emit(False)

    def _run_impl(self) -> None:
        # rclpy import 가능한지 (vision_env 가 --system-site-packages 인지) 확인
        try:
            import rclpy  # noqa: F401
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'rclpy import 실패: {e!r}  (vision_env --system-site-packages 확인)')
            self.finished_run.emit(False)
            return

        # 정렬 모드 — 15번 파이썬을 그대로 실행 (자체 rclpy.init/노드/HOME/비전/정렬/shutdown).
        # 세션이 살아 있으면 rclpy.init 충돌이라 미리 정리.
        if self._req.mode == 'align':
            self._run_p15_main()
            return

        # 복구 모드 — bringup ready + 세션 + activate + safe HOME 만. 비전/배치 X.
        if self._req.mode == 'recover':
            self._run_recover()
            return

        # 캘리브레이션 모드 — 09_원샷_캘리브레이션.py 를 subprocess 로 실행.
        # 09 가 자체적으로 bringup launch + servo + 08 호출. 우리 세션과 별 프로세스.
        if self._req.mode == 'calibrate':
            self._run_oneshot_calibration()
            return

        # 쌓기 모드 — 실 로봇이라면 사전에 bringup 검사 + 자동 시작
        if not self._req.dry_run:
            if not self._ensure_bringup_ready():
                self.finished_run.emit(False)
                return

        # dry-run 일 때는 rclpy 노드 안 만들고 모듈만 — stale session 회피.
        # 실 모드 일 때만 실제 세션 (캐시 가능) 생성.
        if self._req.dry_run:
            session = self._sm.get_dry_session(self.log.emit)
            was_existing = False
        else:
            with self._sm.session_lock:
                was_existing = self._sm.session is not None
                try:
                    session = self._sm.get_or_create_session(self.log.emit)
                except Exception as e:  # noqa: BLE001
                    self.error.emit(f'세션 생성 실패: {e!r}')
                    self.finished_run.emit(False)
                    return

        p12 = session.p12
        p15 = session.p15
        p16 = session.p16
        robot = session.robot

        # 신규 세션이면 DDS 서비스 발견 대기 (정렬 모드 이후 재초기화 시 필수)
        if (not self._req.dry_run) and (not was_existing):
            if not self._warmup_services(robot, timeout_sec=20.0):
                self.finished_run.emit(False)
                return

        try:
            if not self._req.dry_run and not self._req.skip_activate and not session.activated:
                self.log.emit('로봇 활성화...')
                self._activate_with_retry(robot)
                session.activated = True
            if not self._req.dry_run and not session.gripper_inited:
                self.log.emit('그리퍼 초기화 + 열기...')
                robot.gripper_init()
                robot.gripper_open()
                session.gripper_inited = True
            if not self._req.dry_run:
                # 초기 HOME — 로봇이 어떤 자세에서 시작할지 알 수 없으므로 5s 보다 길게.
                # state 검사 후 trajectory 보냄 (servo OFF 면 즉시 중단).
                ok_home = self._safe_move_home(
                    robot, p12, initial_duration=8.0, retry_duration=12.0,
                    label='초기 HOME',
                )
                if not ok_home:
                    self.finished_run.emit(False)
                    return

            # 모드 분기 — 정렬은 별도 시퀀스
            if self._req.mode == 'align':
                self._run_alignment(session)
                return

            # 비전 검출 — dets 가 비어 있으면 호출
            dets = self._req.dets
            if dets is None:
                if self._req.dry_run:
                    dets = self._make_fake_dets(p15)
                    self.log.emit(f'(dry-run) fake dets {len(dets)}개 사용')
                else:
                    self.log.emit('비전 검출 시작 — 카메라 창 ENTER 또는 q 로 확정')
                    try:
                        vision = p15.CubeDetector(conf_thr=self._req.vision_conf_thr)
                        dets = vision.preview_until_confirm()
                    except Exception as e:  # noqa: BLE001
                        self.error.emit(f'비전 실패: {e!r}')
                        return

            need = len(self._req.layout)
            if len(dets) < need:
                self.error.emit(
                    f'검출된 큐브 {len(dets)}개 < 필요 {need}개 — 실행 중단'
                )
                return

            # build_art 와 동일한 매칭 로직
            grid_cells = p15.make_grid_cells()
            MATCH_RADIUS_MM = 40.0
            matched_cubes = []
            matched_cell_indices = []
            for i in range(len(grid_cells)):
                if len(matched_cubes) >= need:
                    break
                cell = grid_cells[i]
                closest, best_d = None, MATCH_RADIUS_MM
                for d in dets:
                    bx, by, _ = d['base_xyz_mm']
                    dd = math.hypot(bx - cell[0], by - cell[1])
                    if dd < best_d:
                        best_d, closest = dd, d
                if closest is not None:
                    matched_cubes.append(closest)
                    matched_cell_indices.append(i)

            if len(matched_cubes) < need:
                self.error.emit(
                    f'그리드에서 매칭된 큐브 {len(matched_cubes)}개 < 필요 {need}개'
                )
                return

            import numpy as np
            sample_z = float(np.median([c['base_xyz_mm'][2] for c in matched_cubes]))
            # 테이블 표면 z — 사용자 override 우선, 아니면 sample_z - cube_w 자동 추정
            if self._req.z_table_top_override_mm is not None:
                z_table_top = float(self._req.z_table_top_override_mm)
                self.log.emit(
                    f'  → z_table_top override 사용: {z_table_top:.1f}mm '
                    f'(자동 추정값 {sample_z - p15.CUBE_WIDTH_MM:.1f}mm 무시)'
                )
            else:
                z_table_top = sample_z - p15.CUBE_WIDTH_MM
            base_xy = self._req.base_xy
            pitch = self._req.pitch_mm

            # 레이아웃 정렬 (layer asc, gy asc, gx asc)
            layout_sorted = sorted(
                enumerate(self._req.layout),
                key=lambda e: (e[1].layer, e[1].gy, e[1].gx),
            )

            cell_positions = [grid_cells[ci] for ci in matched_cell_indices]
            used_cells: set = set()

            class _Args:
                pass

            args = _Args()
            args.dry_run = self._req.dry_run

            self.log.emit(
                f'배치 시작 ({need} cubes, table z={z_table_top:.1f}, '
                f'base=({base_xy[0]:.0f},{base_xy[1]:.0f}), pitch={pitch:.1f}mm)'
            )

            # plan 이 주어졌으면 src/tgt x/y/yaw 는 plan 우선, z 만 robot frame 계산.
            # plan 없으면 기존 grid_cells / model 기반 자동 계산.
            user_plan = self._req.plan
            if user_plan is not None and len(user_plan) >= need:
                self.log.emit(f'  → 사용자 확정 plan {len(user_plan)}개 src/tgt 좌표 사용')

            for order_idx, (orig_i, cube) in enumerate(layout_sorted):
                if self._check_estop():
                    self.error.emit('비상정지 요청 — 중단')
                    self.finished_run.emit(False)
                    return
                cell_idx = matched_cell_indices[order_idx]
                cx_auto, cy_auto = grid_cells[cell_idx]
                pick_yaw_auto = p15.pick_yaw_for_grid_cell(
                    order_idx, cell_positions, 90.0, used_cells,
                )

                # plan 우선, 없으면 자동 — z 는 항상 robot-frame (sample_z + layer*cw)
                tgt_z = (sample_z
                         + cube.layer * self._req.cube_width_mm
                         + self._req.place_z_offset_mm)
                if user_plan is not None and order_idx < len(user_plan):
                    pi = user_plan[order_idx]
                    cx = float(pi.src_x)
                    cy = float(pi.src_y)
                    pick_yaw = float(pi.src_yaw)
                    target_x = float(pi.tgt_x)
                    target_y = float(pi.tgt_y)
                    target_yaw = float(pi.tgt_yaw)
                    src_tag = 'plan'
                else:
                    cx, cy = cx_auto, cy_auto
                    pick_yaw = pick_yaw_auto
                    target_x = base_xy[0] + cube.gx * pitch
                    target_y = base_xy[1] + cube.gy * pitch
                    target_yaw = cube.yaw_deg
                    src_tag = 'auto'

                # Doosan RPY [0,180,Y] — pitch=180 으로 TCP 가 뒤집혀 local Z 가
                # world -Z 방향이라 yaw 부호가 world 에서 반전됨. 시뮬 화면(world
                # +Z CCW 기준) 과 일치시키려면 robot 에 보낼 때 부호 flip.
                pick_yaw_rpy = -pick_yaw
                target_yaw_rpy = -target_yaw

                # 픽 깊이 — p16 안에서 pick_z = src_z - cube_w/2 (cube 중심).
                # 우리가 src_z 를 sample_z + offset 로 주면 pick_z 가 그만큼 더 깊어짐.
                pick_offset = self._req.pick_z_offset_mm
                cw_half = self._req.cube_width_mm / 2.0
                src_z_pick = sample_z + pick_offset
                tentative_pick_z = src_z_pick - cw_half

                # 절대 안전 floor — finger tip 이 추정 테이블 표면 + min_clearance 보다
                # 깊이 못 내려가게. 비전 sample_z 가 불안정해도 테이블 충돌 차단.
                min_clearance = self._req.pick_min_clearance_mm
                floor_pick_z = z_table_top + min_clearance
                if tentative_pick_z < floor_pick_z:
                    # src_z 를 올려서 pick_z = floor 맞춤
                    src_z_pick = floor_pick_z + cw_half
                    self.log.emit(
                        f'⚠ pick_z={tentative_pick_z:.1f}mm 가 floor {floor_pick_z:.1f}mm '
                        f'(table {z_table_top:.1f} + clearance {min_clearance:.1f}) 보다 '
                        f'아래 — src_z 를 {src_z_pick:.1f} 로 올려 clamp'
                    )

                src = (cx, cy, src_z_pick)
                target = (target_x, target_y, tgt_z)

                # p16 가 src_z 받아 pick_z = src_z - cw/2 로 계산
                actual_pick_z = src_z_pick - cw_half
                clearance = actual_pick_z - z_table_top
                msg = (
                    f'[{order_idx+1}/{need}] L{cube.layer} '
                    f'({cube.gx:+.1f},{cube.gy:+.1f}) → '
                    f'({target_x:.0f},{target_y:.0f},z={tgt_z:.0f}) '
                    f'pick({cx:.0f},{cy:.0f},z={actual_pick_z:.1f}, '
                    f'table_clr={clearance:.1f}mm) '
                    f'world_yaw={pick_yaw:+.0f}° tgt_world_yaw={target_yaw:+.0f}° '
                    f'(rpy_z {pick_yaw_rpy:+.0f}/{target_yaw_rpy:+.0f}) [{src_tag}]'
                )
                self.progress.emit(orig_i, need, msg)
                self.log.emit(msg)

                # p16 의 print 를 캡처해 log 시그널로.
                # 부호 flip 된 *_yaw_rpy 를 p16 에 전달 — p16 는 rpy=[0,180,Y] 형태로
                # 보내는데 TCP 가 뒤집힌 상태라 world 회전이 -Y 가 됨.
                # 따라서 -world_yaw 를 보내야 실제 cube 가 world_yaw 로 놓임.
                buf = io.StringIO()
                try:
                    with redirect_stdout(buf):
                        p16.execute_stack_pick_place(
                            robot, src, pick_yaw_rpy, target, target_yaw_rpy,
                            args, z_table_top=z_table_top,
                            pre_open_width_mm=self._req.pre_open_width_mm,
                        )
                except Exception as e:  # noqa: BLE001
                    captured = buf.getvalue()
                    if captured:
                        self.log.emit(captured)
                    self.error.emit(f'[#{order_idx+1}] 실행 실패: {e!r}')
                    self.finished_run.emit(False)
                    return
                captured = buf.getvalue()
                if captured:
                    self.log.emit(captured.rstrip())
                used_cells.add(order_idx)

            if not self._req.dry_run:
                ok_home_end = self._safe_move_home(
                    robot, p12, initial_duration=8.0, retry_duration=12.0,
                    label='HOME 복귀',
                )
                if not ok_home_end:
                    # 배치는 완료됐지만 마지막 HOME 만 실패 — 위험하지만 진행은 했음
                    self.log.emit('⚠ 배치는 완료, HOME 복귀 실패 — 펜던트로 수동 복귀')

            self.log.emit('=== 배치 완료 ===')
            self.finished_run.emit(True)
        finally:
            # robot.destroy_node() 호출하지 않음 — 세션이 보유, GUI 종료 시
            # teardown_session() 에서 한 번에 정리. (rosout publisher 중복 등록 방지)
            pass

    # 15번 stdout 에서 'z 통일: median X.Xmm' 라인을 잡아 sample_z 추출
    _SAMPLE_Z_RE = __import__('re').compile(
        r'z\s+통일:\s+median\s+(-?\d+(?:\.\d+)?)\s*mm'
    )

    # --- align 모드: 15번 파이썬 통째 실행 ----------------------------------
    def _run_p15_main(self) -> None:
        """15_바둑판_정렬.py 를 subprocess 로 실행.

        터미널에서 `python 15_바둑판_정렬.py --quick` 을 친 것과 완전 동일하게:
        - 자체 rclpy.init + PickAndPlace 노드 생성
        - 로봇 활성화 + 그리퍼 초기화 + HOME 이동
        - 비전 창 (cv2.imshow) — 별도 OpenCV 창으로 뜸
        - 큐브 검출 + 그리드 매핑 + 픽앤플레이스 시퀀스
        - HOME 복귀 + vision.stop + robot.destroy_node + rclpy.shutdown

        15번의 stdout/stderr 한 줄씩 그대로 GUI 로그 패널로 흐른다.
        우리 시뮬레이터 세션과 별 프로세스라 rclpy / 노드 이름 충돌 없음.
        """
        import os as _os
        import subprocess
        import sys as _sys

        # 우리 세션이 살아있으면 rosout publisher 이름 충돌 방지를 위해 정리
        if self._sm.session is not None:
            self.log.emit('세션 정리 (정렬은 15번이 자체 노드를 만듬)...')
            self._sm.teardown_session()

        script = MODULE_PATHS['p15']
        if not script.exists():
            self.error.emit(f'스크립트 없음: {script}')
            self.finished_align.emit(False, None)
            self.finished_run.emit(False)
            return

        # 우리 인터프리터(vision_env) 그대로 — rclpy / RealSense / YOLO 동일 환경
        cmd = [_sys.executable, '-u', str(script)]
        if self._req.align_quick:
            cmd.append('--quick')
        if self._req.dry_run:
            cmd.append('--dry-run')
        if self._req.align_limit is not None:
            cmd.extend(['--limit', str(self._req.align_limit)])

        # dry-run 아니면 dsr_bringup2 가 떠 있는지 검사 후, 없으면 자동 시작
        if not self._req.dry_run:
            if not self._ensure_bringup_ready():
                self.finished_align.emit(False, None)
                self.finished_run.emit(False)
                return

        self.log.emit('═' * 60)
        self.log.emit('▶ ' + ' '.join(cmd))
        self.log.emit('═' * 60)

        ok = False
        rc = -1
        try:
            # cwd 는 02_Doosan_Robot_제어 — 15번이 상대경로 (calibration_data 등) 사용
            env = _os.environ.copy()
            env.setdefault('PYTHONUNBUFFERED', '1')
            proc = subprocess.Popen(
                cmd,
                cwd=str(_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                env=env,
            )
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'15번 실행 실패: {e!r}')
            self.finished_align.emit(False, None)
            self.finished_run.emit(False)
            return

        sample_z_captured: Optional[float] = None
        try:
            assert proc.stdout is not None
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                clean = line.rstrip('\n')
                self.log.emit(clean)
                m = self._SAMPLE_Z_RE.search(clean)
                if m:
                    try:
                        sample_z_captured = float(m.group(1))
                    except ValueError:
                        pass
                if self._check_estop() and proc.poll() is None:
                    self.log.emit('▶ 비상정지 — 15번 프로세스 종료')
                    proc.terminate()
                    break
            rc = proc.wait()
            ok = (rc == 0)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'15번 출력 읽기 실패: {e!r}')
            try:
                proc.terminate()
            except Exception:
                pass
            rc = proc.wait() if proc.poll() is None else proc.returncode

        self.log.emit('═' * 60)
        self.log.emit(f'▶ 15번 종료 rc={rc} {"(성공)" if ok else "(실패/중단)"}')
        self.log.emit('═' * 60)

        # 정렬 후 — 실제 grid_cells 좌표 + 캡처된 sample_z 로 dets 구성
        # 그리고 정렬에 쓴 5×5 그리드 영역도 payload 로 함께 전달 → GUI 가
        # 그리드 영역을 화면 바닥에 시각화.
        payload = None
        if ok:
            try:
                p15 = _load_module('p15')
                grid_cells = p15.make_grid_cells()
                sample_z = sample_z_captured if sample_z_captured is not None else 0.0
                if sample_z_captured is not None:
                    self.log.emit(f'  → 정렬 z 캡처: sample_z={sample_z:.1f}mm')
                else:
                    self.log.emit('  → sample_z 캡처 실패 (z 통일 라인 미발견) — 0 으로 대체')
                dets = [
                    {'base_xyz_mm': (float(x), float(y), float(sample_z)),
                     'cube_yaw_deg': 0.0, 'conf': 1.0}
                    for x, y in grid_cells
                ]
                # 5×5 그리드 중심 = (시작셀 + 끝셀) / 2
                xs = [c[0] for c in grid_cells]
                ys = [c[1] for c in grid_cells]
                grid_center = (
                    (min(xs) + max(xs)) / 2.0,
                    (min(ys) + max(ys)) / 2.0,
                )
                payload = {
                    'dets': dets,
                    'sample_z': float(sample_z),
                    'grid_cells': [(float(x), float(y)) for x, y in grid_cells],
                    'grid_center': grid_center,
                    'grid_spacing': float(p15.GRID_SPACING),
                    'grid_rows': int(p15.GRID_ROWS),
                    'grid_cols': int(p15.GRID_COLS),
                    'cube_width_mm': float(p15.CUBE_WIDTH_MM),
                }
            except Exception as e:  # noqa: BLE001
                self.log.emit(f'  → payload 구성 실패: {e!r}')
                payload = None
        self.finished_align.emit(ok, payload)
        self.finished_run.emit(ok)

    # --- 원샷 캘리브레이션 모드 ----------------------------------------------
    def _run_oneshot_calibration(self) -> None:
        """09_원샷_캘리브레이션.py 를 subprocess 로 실행.

        15번 정렬 모드와 동일 패턴 — 09 가 자체 rclpy.init / bringup 관리 /
        servo on / 08 호출. 우리 세션은 충돌 방지를 위해 미리 정리.
        09 자체가 cv2.imshow 창을 띄움 (ArUco 마커 보기 등). 사용자는 GUI
        로그 + cv2 창 둘 다 보며 's' / ENTER 키 입력.
        """
        import os as _os
        import subprocess
        import sys as _sys

        # 우리 세션이 살아있으면 rosout publisher 이름 충돌 방지를 위해 정리
        if self._sm.session is not None:
            self.log.emit('세션 정리 (캘리브는 09 가 자체 노드를 만듬)...')
            self._sm.teardown_session()

        script = MODULE_PATHS['p09']
        if not script.exists():
            self.error.emit(f'스크립트 없음: {script}')
            self.finished_run.emit(False)
            return

        cmd = [_sys.executable, '-u', str(script)]
        if self._req.cal_skip_launch:
            cmd.append('--skip-launch')
        if self._req.cal_no_cleanup:
            cmd.append('--no-cleanup')
        if self._req.cal_keep_bringup:
            cmd.append('--keep-bringup')

        self.log.emit('═' * 60)
        self.log.emit('🎯 원샷 Hand-Eye 캘리브레이션 시작')
        self.log.emit('  ' + ' '.join(cmd))
        self.log.emit('  사용 안내:')
        self.log.emit('    1. ArUco 마커가 그리퍼에 부착돼 있어야 합니다 (DICT_6X6_50, 50mm)')
        self.log.emit('    2. cv2 창이 별도로 떠 마커 검출 화면 표시')
        self.log.emit("    3. 마커 보이는 안전 자세로 RViz/펜던트로 이동 후 's' 키로 base 자세 저장")
        self.log.emit('    4. 자동으로 20개 포즈 순회 + Hand-Eye 캘리브 계산')
        self.log.emit('═' * 60)

        try:
            env = _os.environ.copy()
            env.setdefault('PYTHONUNBUFFERED', '1')
            proc = subprocess.Popen(
                cmd,
                cwd=str(_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                env=env,
            )
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'09 실행 실패: {e!r}')
            self.finished_run.emit(False)
            return

        rc = -1
        try:
            assert proc.stdout is not None
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                self.log.emit(line.rstrip('\n'))
                if self._check_estop() and proc.poll() is None:
                    self.log.emit('▶ 비상정지 — 09 프로세스 종료')
                    proc.terminate()
                    break
            rc = proc.wait()
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'09 출력 읽기 실패: {e!r}')
            try:
                proc.terminate()
            except Exception:
                pass
            rc = proc.wait() if proc.poll() is None else proc.returncode

        ok = (rc == 0)
        self.log.emit('═' * 60)
        self.log.emit(f'🎯 캘리브레이션 종료 rc={rc} {"(성공)" if ok else "(실패/중단)"}')
        if ok:
            self.log.emit('  → calibration_data/ 폴더에 결과 저장됨 (translation/rotation)')
        self.log.emit('═' * 60)
        self.finished_run.emit(ok)

    # --- bringup 자동 라이프사이클 --------------------------------------------
    def _ensure_bringup_ready(self, wait_timeout: float = 90.0) -> bool:
        """dsr_bringup2 가 떠 있는지 확인, 없으면 자동 시작 후 대기.

        - 이미 서비스 등록돼 있으면 즉시 True
        - 없으면 ros2 launch 를 백그라운드로 실행 + 서비스 등록될 때까지 polling
        - 등록되면 True, timeout/실패면 False
        - 띄운 프로세스는 모듈 전역 _BRINGUP_PROC 로 보관 → 다음 작업 재사용,
          GUI 종료 시 teardown_bringup() 으로 정리
        """
        # 1) 이미 떠 있나?
        self.log.emit('[bringup] 서비스 가용성 확인...')
        available, _ = self._check_doosan_bringup(timeout_sec=3.0)
        if available:
            self.log.emit('[bringup] ✓ 이미 떠 있음 — 자동 시작 생략')
            return True

        # 2) 자동 시작
        self.log.emit('[bringup] dsr_bringup2 가 안 떠 있음 — 자동으로 띄움')
        if not self._auto_start_bringup():
            self.error.emit('dsr_bringup2 자동 시작 실패 — 로그 확인')
            return False

        # 3) 서비스 등록 polling
        import time
        deadline = time.monotonic() + wait_timeout
        last_log = 0.0
        while time.monotonic() < deadline:
            if self._check_estop():
                self.log.emit('[bringup] 비상정지 — 대기 중단')
                return False
            with self._sm.bringup_lock:
                proc = self._sm.bringup_proc
            if proc is None or proc.poll() is not None:
                rc = proc.returncode if proc is not None else 'N/A'
                self.error.emit(f'[bringup] 프로세스 예기치 못한 종료 rc={rc}')
                return False
            available, _ = self._check_doosan_bringup(timeout_sec=2.0)
            if available:
                self.log.emit('[bringup] ✓ 서비스 등록 완료 — 정렬/쌓기 진행')
                # 컨트롤러 안정 위해 짧게 대기
                time.sleep(1.5)
                return True
            now = time.monotonic()
            if now - last_log > 5.0:
                remaining = int(deadline - now)
                self.log.emit(
                    f'[bringup] 서비스 등록 대기... (남은 {remaining}s, '
                    f'두산 컨트롤러 연결 시도 중)'
                )
                last_log = now
            time.sleep(1.0)
        self.error.emit(
            '[bringup] 타임아웃 — 컨트롤러 IP/네트워크/모델명 확인 필요'
        )
        return False

    def _auto_start_bringup(self) -> bool:
        """dsr_bringup2 launch 를 백그라운드 subprocess 로 시작.

        bash -c 로 source ros + ws + ros2 launch — GUI 가 ros 환경 source 없이
        실행돼도 동작. start_new_session=True 로 process group 분리하여
        teardown 시 그룹 통째로 SIGTERM.
        """
        import os as _os
        import subprocess
        import threading as _threading
        from pathlib import Path as _Path

        sm = self._sm
        with sm.bringup_lock:
            if sm.bringup_proc is not None and sm.bringup_proc.poll() is None:
                self.log.emit('[bringup] 이미 시작됨 — wait 만 진행')
                sm.bringup_log_fn = self.log.emit
                return True

        # doosan_config 에서 namespace / model / host / ws / distro
        try:
            cfg = _load_module('doosan_config')
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'doosan_config 로드 실패: {e!r}')
            return False
        ns = cfg.NAMESPACE
        model = cfg.ROBOT_MODEL
        host = cfg.ROBOT_IP
        ws = cfg.DOOSAN_WS
        distro = cfg.ROS_DISTRO
        pkg = cfg.BRINGUP_PKG
        launch = cfg.BRINGUP_LAUNCH

        # ros + ws source 후 launch
        shell_lines = []
        ros_setup = f'/opt/ros/{distro}/setup.bash'
        if _Path(ros_setup).exists():
            shell_lines.append(f'source {ros_setup}')
        else:
            # 다른 distro 자동 감지
            for d in ('humble', 'iron', 'jazzy', 'rolling', 'foxy'):
                p = f'/opt/ros/{d}/setup.bash'
                if _Path(p).exists():
                    shell_lines.append(f'source {p}')
                    distro = d
                    break
        ws_setup = _Path(ws) / 'install' / 'setup.bash'
        if ws_setup.exists():
            shell_lines.append(f'source {ws_setup}')
        else:
            # 후보 위치 자동 감지
            for cand in (
                _Path.home() / 'doosan_ws/install/setup.bash',
                _Path.home() / 'ros2_ws/install/setup.bash',
                _Path.home() / 'dsr_ws/install/setup.bash',
            ):
                if cand.exists():
                    shell_lines.append(f'source {cand}')
                    break
        shell_lines.append(
            f'exec ros2 launch {pkg} {launch} '
            f'name:={ns} model:={model} host:={host} mode:=real'
        )
        bash_cmd = ' && '.join(shell_lines)
        self.log.emit(f'[bringup] $ {bash_cmd}')

        try:
            proc = subprocess.Popen(
                ['bash', '-c', bash_cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,  # process group 분리
                env=_os.environ.copy(),
            )
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'[bringup] subprocess 실패: {e!r}')
            return False

        with sm.bringup_lock:
            sm.bringup_proc = proc
            sm.bringup_log_fn = self.log.emit

        # stdout reader — bringup 로그를 GUI 로 흘려보냄
        def _reader():
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    fn = sm.bringup_log_fn
                    if fn is None:
                        continue
                    try:
                        fn('[bringup] ' + line.rstrip())
                    except Exception:
                        return
            except Exception:
                pass

        _threading.Thread(target=_reader, daemon=True).start()
        return True

    # --- helpers --------------------------------------------------------------
    def _check_doosan_bringup(self,
                              ns: Optional[str] = None,
                              probe: str = 'system/set_robot_mode',
                              timeout_sec: float = 5.0
                              ) -> tuple[bool, str]:
        """`ros2 service list` 로 dsr_bringup2 서비스 등록 여부 빠른 검사.

        Returns:
            (available, hint) — available=False 면 hint 에 다음 조치 안내.
        """
        import subprocess
        if ns is None:
            try:
                cfg = _load_module('doosan_config')
                ns = cfg.NAMESPACE
            except Exception:
                ns = 'dsr01'
        target = f'/{ns}/{probe}'
        try:
            out = subprocess.check_output(
                ['ros2', 'service', 'list'],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
            )
        except FileNotFoundError:
            return (False,
                    "ros2 명령을 찾을 수 없습니다. "
                    "터미널에서 'source /opt/ros/humble/setup.bash' 후 재실행하세요.")
        except subprocess.TimeoutExpired:
            return (False,
                    "ros2 service list 가 응답하지 않음. "
                    "ROS_DOMAIN_ID / DDS 설정 확인.")
        except subprocess.CalledProcessError as e:
            return (False, f'ros2 service list 실패: {e!r}')
        if target in out:
            return (True, '')
        return (
            False,
            f"'{target}' 미등록. 별도 터미널에서 dsr_bringup2 를 띄우세요:\n"
            f"    source /opt/ros/humble/setup.bash\n"
            f"    source ~/doosan_ws/install/setup.bash\n"
            f"    ros2 launch dsr_bringup2 dsr_bringup2_moveit.launch.py "
            f"name:=dsr01 model:=e0509 host:=110.120.1.18 mode:=real",
        )

    # 두산 robot_state 코드 — p12._STATE_NAMES 와 동일.
    # 1=STANDBY 2=MOVING 이면 servo ON, 그 외 (3=SAFE_OFF 등) 는 trajectory 불가.
    _STATE_SERVO_ON = (1, 2)
    _STATE_HINTS = {
        3: ('SAFE_OFF',
            '펜던트 모드 스위치가 AUTO(또는 RUN) 위치인지 확인 / '
            'EtherCAT·모터드라이버 통신 / 안전 회로 (door, light curtain)'),
        10: ('SAFE_OFF2', '펜던트 모드/안전 회로 확인'),
        5: ('SAFE_STOP', '보호 정지 — 펜던트에서 reset 또는 충돌 해제'),
        9: ('SAFE_STOP2', '보호 정지 — 펜던트에서 reset'),
        6: ('EMERGENCY_STOP',
            '펜던트의 빨간 비상정지 버튼을 풀고 재시도 (코드로 reset 불가)'),
        4: ('TEACHING', '펜던트가 티칭 모드 — AUTO 모드로 전환'),
    }

    def _activate_with_retry(self, robot, retries: int = 1) -> None:
        """robot.activate_robot() — '서비스 미응답' 만 한정해 워밍업 후 재시도.

        DDS discovery 가 첫 호출 직전에 완료될 때도 가끔 1차 wait_for_service
        가 살짝 부족해 timeout. 워밍업 추가 + 재시도.
        실패가 '서비스 미응답' 외 (활성화 거부 등) 면 즉시 raise — 호출자가 처리.
        """
        last_exc = None
        for attempt in range(retries + 1):
            try:
                robot.activate_robot()
                return
            except RuntimeError as e:
                msg = str(e)
                last_exc = e
                if '서비스 미응답' not in msg:
                    raise
                self.log.emit(
                    f'  activate 1차 시도 timeout ({msg}) — 워밍업 후 재시도'
                )
                if not self._warmup_services(robot, timeout_sec=15.0):
                    raise
        if last_exc is not None:
            raise last_exc

    def _warmup_services(self, robot, timeout_sec: float = 20.0) -> bool:
        """새 세션 생성 직후 DDS 서비스 발견 대기.

        teardown_session(rclpy.shutdown) 후 다시 rclpy.init + 새 노드 만들면
        DDS 가 서비스를 재발견할 때까지 5초 (`_wait` 기본) 보다 오래 걸릴 수
        있어 'set_robot_mode 미응답' 으로 깨짐. critical 서비스에 한해 더 긴
        timeout 으로 wait_for_service 호출 + rclpy spin 으로 discovery 진행.
        """
        import rclpy
        critical = [
            ('cli_mode', 'set_robot_mode'),
            ('cli_ctrl', 'set_robot_control'),
            ('cli_get_state', 'get_robot_state'),
        ]
        self.log.emit(f'세션 워밍업 — critical 서비스 발견 대기 (최대 {timeout_sec:.0f}s)')
        import time
        deadline = time.monotonic() + timeout_sec
        for attr, name in critical:
            cli = getattr(robot, attr, None)
            if cli is None:
                continue
            remaining = max(1.0, deadline - time.monotonic())
            try:
                # spin_once 로 discovery 메시지 흘리면서 wait_for_service 분할 호출
                slice_sec = 1.0
                got = False
                while remaining > 0:
                    if cli.wait_for_service(timeout_sec=min(slice_sec, remaining)):
                        got = True
                        break
                    try:
                        rclpy.spin_once(robot, timeout_sec=0.1)
                    except Exception:
                        pass
                    remaining = deadline - time.monotonic()
                if not got:
                    self.error.emit(
                        f'워밍업 timeout: {name} 서비스 발견 실패 — '
                        '컨트롤러/네트워크 확인'
                    )
                    return False
            except Exception as e:  # noqa: BLE001
                self.error.emit(f'워밍업 예외 ({name}): {e!r}')
                return False
        self.log.emit('세션 워밍업 완료')
        return True

    def _hard_reset_driver(self, wait_after_kill: float = 3.0,
                           bringup_timeout: float = 90.0) -> bool:
        """15번의 reset_robot_driver() 와 동일 원리 — 드라이버를 통째로 재시작.

        pendant 없이도 SAFE_OFF / SAFE_STOP / 컨트롤러 stale 상태를 한 번에 해소.
        흐름:
          1) dsr_* / DRCF / 기존 bringup 프로세스 강제 kill
          2) 우리 SessionManager 의 bringup_proc 핸들 클리어
          3) 짧게 대기 (3s)
          4) 우리 자체 _auto_start_bringup 로 새 bringup launch
          5) 서비스 등록 polling (최대 90s) — 25~30s 보통 소요

        15번과 차이점: gnome-terminal 안 쓰고 우리 인프라(_auto_start_bringup)
        사용. 한글 강의 스크립트나 GUI 자체는 죽이지 않음.
        """
        import os
        import time as _time
        self.log.emit('=== 드라이버 강제 리셋 (≈30초) — 15번 reset_robot_driver 원리 ===')

        # 1) 컨트롤러/드라이버 측 프로세스 강제 종료. GUI 자체는 안 죽이려고
        #    범위 좁힘 (dsr_bringup / DRCF / dsr_controller / dsr_hardware 만).
        kill_patterns = [
            'dsr_bringup',
            'DRCF',
            'dsr_controller',
            'dsr_hardware',
            'dsr_dual_controller',
        ]
        for pat in kill_patterns:
            os.system(f'pkill -9 -f {pat} 2>/dev/null')
        self.log.emit('  → dsr/DRCF 프로세스 정리 완료')

        # 2) 우리 자체 bringup_proc 핸들 클리어 (kill 됐으니 이미 죽었지만 핸들은 남음)
        with self._sm.bringup_lock:
            if self._sm.bringup_proc is not None:
                try:
                    self._sm.bringup_proc.terminate()
                except Exception:
                    pass
                try:
                    self._sm.bringup_proc.wait(timeout=2)
                except Exception:
                    pass
                self._sm.bringup_proc = None

        # 3) 안정화 대기
        self.log.emit(f'  → {wait_after_kill:.1f}s 대기 후 bringup 재시작')
        _time.sleep(wait_after_kill)
        if self._check_estop():
            return False

        # 4) 우리 자체 bringup auto-start
        if not self._auto_start_bringup():
            self.error.emit('드라이버 재시작 실패 — dsr_bringup2 launch 불가')
            return False

        # 5) 서비스 등록 polling — 컨트롤러 + bringup 안정화까지 최대 bringup_timeout 초
        self.log.emit('  → 드라이버 안정화 대기 (보통 25~30s)')
        deadline = _time.monotonic() + bringup_timeout
        last_log = 0.0
        while _time.monotonic() < deadline:
            if self._check_estop():
                return False
            with self._sm.bringup_lock:
                proc = self._sm.bringup_proc
            if proc is None or proc.poll() is not None:
                rc = proc.returncode if proc is not None else 'N/A'
                self.error.emit(f'  bringup 프로세스 예기치 못한 종료 rc={rc}')
                return False
            available, _ = self._check_doosan_bringup(timeout_sec=2.0)
            if available:
                self.log.emit('  → 컨트롤러 + bringup 안정화 완료')
                _time.sleep(2.0)  # 추가 안정화
                return True
            now = _time.monotonic()
            if now - last_log > 5.0:
                remaining = int(deadline - now)
                self.log.emit(f'  ... 서비스 등록 대기 (남은 {remaining}s)')
                last_log = now
            _time.sleep(1.0)

        self.error.emit('드라이버 재시작 후 서비스 등록 timeout')
        return False

    def _ensure_servo_on(self, robot, label: str) -> bool:
        """state 검사 + 필요 시 recover_safety + activate_robot 으로 servo ON 보장.

        True 면 trajectory 보낼 수 있는 상태. False 면 사용자 개입 필요 — error.emit
        을 한 번 호출해 두고 호출자는 finished_run(False) 로 빠짐.

        SAFE_OFF 등 코드로 못 풀리는 상태는 1회만 시도하고 즉시 명확한 가이드 +
        중단 — 반복 시도는 사용자에게 혼란만 줌.
        """
        try:
            state = robot.get_robot_state()
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'{label}: robot_state 조회 실패 — {e!r}')
            return False
        if state in self._STATE_SERVO_ON:
            return True
        name, hint = self._STATE_HINTS.get(state, (f'UNKNOWN({state})', ''))
        self.log.emit(f'{label}: robot_state={state} ({name}) — 1회 자동 복구 시도')
        try:
            robot.recover_safety(verbose=False)
            robot.activate_robot()
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'{label}: 활성화 실패 — {e!r}')
            return False
        state2 = robot.get_robot_state()
        if state2 in self._STATE_SERVO_ON:
            return True
        name2, _ = self._STATE_HINTS.get(state2, (f'UNKNOWN({state2})', ''))
        self.error.emit(
            f'{label} 중단 — robot_state={state2} ({name2}) 가 코드로 풀리지 않음.\n'
            f'  조치: {hint or "두산 매뉴얼 참고"}\n'
            f'  ① 펜던트 모드 키 스위치 = AUTO 위치인지 확인\n'
            f'  ② 펜던트의 "Servo ON" 또는 "Reset" 버튼 직접 누르기\n'
            f'  ③ EtherCAT/안전 회로 (door, light curtain) 점검\n'
            f'  ④ 위 ②까지 한 뒤 [🏠 원점복귀] 다시 시도'
        )
        return False

    def _safe_move_home(self,
                        robot,
                        p12,
                        initial_duration: float = 8.0,
                        retry_duration: float = 12.0,
                        label: str = 'HOME 이동') -> bool:
        """HOME 으로 안전 이동 — state 사전 검사 + trajectory 재시도.

        리턴: True=성공, False=중단 (error.emit 호출됨).
        두산 controller 는 시작 자세가 멀거나 path tolerance 넘으면 trajectory
        를 status=6 (ABORTED) 로 종료. 초기 HOME 은 8s + 1회 12s 재시도.
        servo 가 안 켜진 상태(SAFE_OFF 등) 면 trajectory 자체 시도하지 않고
        명확한 가이드와 함께 중단.
        """
        if not self._ensure_servo_on(robot, label):
            return False
        self.log.emit(f'{label} 이동 중 (duration={initial_duration:.1f}s)...')
        try:
            robot.move_joint_deg(p12.HOME_JOINT_DEG, duration=initial_duration)
            return True
        except RuntimeError as e:
            msg = str(e)
            self.log.emit(f'⚠ {label} 1차 실패: {msg}')
            if 'status=6' in msg or 'ABORTED' in msg.upper():
                if not self._ensure_servo_on(robot, label + ' (복구)'):
                    return False
            self.log.emit(
                f'{label} 재시도 (duration={retry_duration:.1f}s) — '
                '시작 자세가 멀면 더 길게 잡음'
            )
            try:
                robot.move_joint_deg(p12.HOME_JOINT_DEG, duration=retry_duration)
                return True
            except RuntimeError as e2:
                self.error.emit(f'{label} 재시도도 실패: {e2!r}')
                return False

    def _run_recover(self) -> None:
        """복구 모드 — 15번 main 의 reset_robot_driver 와 동일 원리로 강제 reset.

        펜던트 없이도 SAFE_OFF / SAFE_STOP / 비상정지 해제 직후 등 모든 상태에서
        복구. controller 가 미리 teardown_session 호출 가정.
        """
        if self._req.dry_run:
            self.log.emit('(dry-run) 복구 모드 시뮬 — 실 모션 없이 통과')
            self.finished_run.emit(True)
            return

        # 1) 15번 reset_robot_driver 원리 — dsr/DRCF/bringup 통째 kill 후 새로 launch.
        #    SAFE_OFF 도 이걸로 풀림 (펜던트 조작 불필요).
        if not self._hard_reset_driver():
            self.finished_run.emit(False)
            return

        with self._sm.session_lock:
            try:
                # controller.start_recover_home 가 teardown 했으니 항상 신규 생성
                session = self._sm.get_or_create_session(self.log.emit)
            except Exception as e:  # noqa: BLE001
                self.error.emit(f'복구: 세션 생성 실패 — {e!r}')
                self.finished_run.emit(False)
                return

        p12 = session.p12
        robot = session.robot
        self.log.emit('=== 원점복귀 시작 ===')

        # 새 노드라 DDS service discovery 보장 — 길게 (30s)
        if not self._warmup_services(robot, timeout_sec=30.0):
            self.finished_run.emit(False)
            return

        # 1) recover_safety + activate (재시도 포함)
        try:
            self.log.emit('1/2  recover_safety + activate_robot')
            try:
                robot.recover_safety(verbose=True)
            except Exception as re:  # noqa: BLE001
                self.log.emit(f'  → recover_safety 무시: {re!r}')
            self._activate_with_retry(robot, retries=2)
            session.activated = True
        except Exception as e:  # noqa: BLE001
            self.error.emit(f'복구: activate 실패 — {e!r}')
            self.finished_run.emit(False)
            return

        # 활성화 후 그리퍼도 초기화 — 비상정지 해제 직후엔 그리퍼 토크가 풀려 있음
        try:
            self.log.emit('1.5/2  그리퍼 재초기화 + 열기')
            robot.gripper_init()
            robot.gripper_open()
            session.gripper_inited = True
        except Exception as e:  # noqa: BLE001
            self.log.emit(f'  → 그리퍼 init 실패 (무시, HOME 이동만 진행): {e!r}')

        # 2) state 검사 + HOME 이동 (긴 duration, 1회 재시도)
        self.log.emit('2/2  HOME 자세 이동')
        if self._check_estop():
            self.error.emit('복구 중 비상정지 요청 — 중단')
            self.finished_run.emit(False)
            return
        ok = self._safe_move_home(
            robot, p12, initial_duration=10.0, retry_duration=15.0,
            label='원점복귀',
        )
        if ok:
            self.log.emit('=== 원점복귀 완료 ===')
            self.finished_run.emit(True)
        else:
            self.finished_run.emit(False)

    def _make_fake_dets(self, p15) -> list[dict]:
        """비전 없이 dry-run 할 때, grid_cells 를 그대로 source 로 사용."""
        cells = p15.make_grid_cells()
        sample_z = 50.0
        dets = []
        for x, y in cells:
            dets.append({
                'base_xyz_mm': (float(x), float(y), sample_z),
                'cube_yaw_deg': 0.0,
                'conf': 1.0,
            })
        return dets
