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
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

from .model import PlacedCube


_DIR = Path(__file__).resolve().parent.parent  # 02_Doosan_Robot_제어/


def _load_module(name: str, filename: str):
    """한글/숫자 시작 파일을 importlib 로 로드."""
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(_DIR))
    spec = importlib.util.spec_from_file_location(name, str(_DIR / filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- 세션 (rclpy + PickAndPlace 노드 재사용) ---------------------------------
# 매 실행마다 노드를 만들고 부수면 'vision_pick_and_place' 같은 고정 이름의
# rosout publisher 가 반복 등록되어 [WARN] Publisher already registered ... 가
# 발생한다. GUI 한 세션 동안 노드 1개만 만들어 재사용.

@dataclass
class RobotSession:
    p12: object
    p15: object
    p16: object
    robot: object
    activated: bool = False
    gripper_inited: bool = False

    def shutdown(self) -> None:
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


_SESSION: Optional[RobotSession] = None
_SESSION_LOCK = threading.Lock()


def _create_session() -> RobotSession:
    import rclpy
    if not rclpy.ok():
        rclpy.init()
    p12 = _load_module('p12', '12_비전_피크앤플레이스.py')
    p15 = _load_module('p15', '15_바둑판_정렬.py')
    p16 = _load_module('p16', '16_탑쌓기.py')
    robot = p12.PickAndPlace()
    return RobotSession(p12=p12, p15=p15, p16=p16, robot=robot)


def teardown_session() -> None:
    """앱 종료 시 (MainWindow.closeEvent) 에서 1회 호출."""
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            _SESSION.shutdown()
            _SESSION = None


@dataclass
class RunRequest:
    """[로봇 실행] 1회 요청."""
    layout: list[PlacedCube]
    base_xy: tuple[float, float]
    pitch_mm: float
    cube_width_mm: float
    dry_run: bool = False
    # dets: build_art 와 동일한 형식 [{'base_xyz_mm': (x,y,z), 'cube_yaw_deg': float}, ...]
    # None 이면 worker 가 비전을 호출.
    dets: Optional[list[dict]] = None
    # 비전 호출 시 conf threshold
    vision_conf_thr: float = 0.40
    skip_activate: bool = False  # 이미 활성화된 robot 재사용 시
    pre_open_width_mm: float = 40.0
    place_pre_open_width_mm: float = 27.0


class RobotWorker(QThread):
    """1 회의 RunRequest 를 처리하고 종료."""

    progress = pyqtSignal(int, int, str)        # (current_idx, total, msg)
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    started_run = pyqtSignal()
    finished_run = pyqtSignal(bool)             # True = 정상 완료

    def __init__(self, req: RunRequest, parent=None):
        super().__init__(parent)
        self._req = req
        self._estop = False
        self._mutex = QMutex()

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

        # 세션 (PickAndPlace 노드 1개) 생성 또는 재사용
        global _SESSION
        with _SESSION_LOCK:
            if _SESSION is None:
                try:
                    self.log.emit('세션 초기화 — PickAndPlace 노드 생성 중...')
                    _SESSION = _create_session()
                except Exception as e:  # noqa: BLE001
                    self.error.emit(f'세션 생성 실패: {e!r}')
                    self.finished_run.emit(False)
                    return
            else:
                self.log.emit('세션 재사용 (노드 재생성 없음)')
            session = _SESSION

        p12 = session.p12
        p15 = session.p15
        p16 = session.p16
        robot = session.robot

        try:
            if not self._req.dry_run and not self._req.skip_activate and not session.activated:
                self.log.emit('로봇 활성화...')
                robot.activate_robot()
                session.activated = True
            if not self._req.dry_run and not session.gripper_inited:
                self.log.emit('그리퍼 초기화 + 열기...')
                robot.gripper_init()
                robot.gripper_open()
                session.gripper_inited = True
            if not self._req.dry_run:
                robot.move_joint_deg(p12.HOME_JOINT_DEG, duration=p15.MOVE_DURATION_SEC)

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

            for order_idx, (orig_i, cube) in enumerate(layout_sorted):
                if self._check_estop():
                    self.error.emit('비상정지 요청 — 중단')
                    self.finished_run.emit(False)
                    return
                cell_idx = matched_cell_indices[order_idx]
                cx, cy = grid_cells[cell_idx]
                pick_yaw = p15.pick_yaw_for_grid_cell(
                    order_idx, cell_positions, 90.0, used_cells,
                )
                src = (cx, cy, sample_z)
                target_x = base_xy[0] + cube.gx * pitch
                target_y = base_xy[1] + cube.gy * pitch
                target_z = sample_z + cube.layer * self._req.cube_width_mm
                target = (target_x, target_y, target_z)

                msg = (
                    f'[{order_idx+1}/{need}] L{cube.layer} '
                    f'({cube.gx:+.1f},{cube.gy:+.1f}) → '
                    f'({target_x:.0f},{target_y:.0f},z={target_z:.0f}) '
                    f'pick({cx:.0f},{cy:.0f}) yaw={pick_yaw:+.0f}°'
                )
                self.progress.emit(orig_i, need, msg)
                self.log.emit(msg)

                # p16 의 print 를 캡처해 log 시그널로
                buf = io.StringIO()
                try:
                    with redirect_stdout(buf):
                        p16.execute_stack_pick_place(
                            robot, src, pick_yaw, target, cube.yaw_deg,
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
                self.log.emit('HOME 복귀...')
                robot.move_joint_deg(p12.HOME_JOINT_DEG, duration=p15.MOVE_DURATION_SEC)

            self.log.emit('=== 배치 완료 ===')
            self.finished_run.emit(True)
        finally:
            # robot.destroy_node() 호출하지 않음 — 세션이 보유, GUI 종료 시
            # teardown_session() 에서 한 번에 정리. (rosout publisher 중복 등록 방지)
            pass

    # --- helpers --------------------------------------------------------------
    def _make_fake_dets(self, p15) -> list[dict]:
        """비전 없이 dry-run 할 때, grid_cells 를 그대로 source 로 사용."""
        cells = p15.make_grid_cells()
        sample_z = 50.0  # 임의 — dry-run 에선 모션 안 함
        dets = []
        for x, y in cells:
            dets.append({
                'base_xyz_mm': (float(x), float(y), sample_z),
                'cube_yaw_deg': 0.0,
                'conf': 1.0,
            })
        return dets
