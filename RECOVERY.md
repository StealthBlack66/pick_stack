# 두산 로봇 충돌 신호 리커버리

## 문제 상황

두산 e0509(혹은 다른 시리즈)를 ROS2로 처음 연결하면, 컨트롤러에 **이전 세션에서 발생한 SAFE_STOP(충돌·protective stop) 신호가 남아있는 경우**가 있다. 이 상태에서는 `set_robot_mode`, `set_robot_control`, 모션 명령을 아무리 보내도 로봇이 움직이지 않는다.

펜던트에서 손으로 풀어도 되지만, 스크립트 실행 때마다 반복해야 하므로 **ROS2 서비스로 자동 복구**하는 코드를 활성화 루틴에 넣었다.

---

## 로봇 상태 코드

`/dsr01/system/get_robot_state` 가 반환하는 정수.

| 코드 | 이름 | 비고 |
|----|----|----|
| 0 | INITIALIZING | |
| 1 | STANDBY | 정상 — 명령 가능 |
| 2 | MOVING | 정상 — 이동 중 |
| 3 | SAFE_OFF | Servo OFF — `SetRobotControl(1)` 으로 복구 |
| 4 | TEACHING | |
| **5** | **SAFE_STOP** | **충돌·protective stop — 자동 복구 대상** |
| 6 | EMERGENCY_STOP | 펜던트 비상정지 버튼 — **코드로 못 풂** |
| 7 | HOMING | |
| 8 | RECOVERY | 복구 모드 진입 상태 |
| **9** | **SAFE_STOP2** | **2차 보호정지 — 자동 복구 대상** |
| 10 | SAFE_OFF2 | Servo OFF — `SetRobotControl(1)` 으로 복구 |
| 15 | NOT_READY | |

---

## 복구 시퀀스 (SAFE_STOP / SAFE_STOP2)

3단계 ROS2 서비스 호출. 각 단계 사이 0.5초 sleep 은 컨트롤러 내부 상태 전이가 비동기로 끝나기 때문에 필요.

```
1) RECOVERY 진입       set_safety_mode            safety_mode=2
2) safe-stop 리셋       set_safe_stop_reset_type   reset_type=0
3) AUTONOMOUS 복귀     set_safety_mode            safety_mode=1
```

최종 상태가 1(STANDBY) 또는 2(MOVING) 이면 복구 성공.

---

## 코드

### import / 서비스 클라이언트

```python
import time
import rclpy
from rclpy.node import Node

from dsr_msgs2.srv import (
    SetRobotMode, SetRobotControl,
    GetRobotState, SetSafetyMode, SetSafeStopResetType,
)

NS = 'dsr01'   # 본인 네임스페이스에 맞게

class MyRobot(Node):
    def __init__(self):
        super().__init__('my_robot')
        self.cli_mode        = self.create_client(SetRobotMode,         f'/{NS}/system/set_robot_mode')
        self.cli_ctrl        = self.create_client(SetRobotControl,      f'/{NS}/system/set_robot_control')
        self.cli_get_state   = self.create_client(GetRobotState,        f'/{NS}/system/get_robot_state')
        self.cli_safety_mode = self.create_client(SetSafetyMode,        f'/{NS}/system/set_safety_mode')
        self.cli_safe_reset  = self.create_client(SetSafeStopResetType, f'/{NS}/system/set_safe_stop_reset_type')

    def _wait(self, cli, name, timeout=5.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'서비스 미응답: {name}')

    def _call(self, cli, req, timeout=30.0):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()
```

### 상태 조회 + 리커버리

```python
    _STATE_NAMES = {
        0: 'INITIALIZING', 1: 'STANDBY', 2: 'MOVING',
        3: 'SAFE_OFF', 4: 'TEACHING', 5: 'SAFE_STOP',
        6: 'EMERGENCY_STOP', 7: 'HOMING', 8: 'RECOVERY',
        9: 'SAFE_STOP2', 10: 'SAFE_OFF2', 15: 'NOT_READY',
    }

    def get_robot_state(self):
        """현재 robot_state 정수 반환. 실패 시 -1."""
        if not self.cli_get_state.wait_for_service(timeout_sec=2.0):
            return -1
        r = self._call(self.cli_get_state, GetRobotState.Request(), timeout=3.0)
        return r.robot_state if r else -1

    def recover_safety(self, verbose=True):
        """
        충돌·safe-stop 후 fault 복구.
          - SAFE_STOP / SAFE_STOP2 : RECOVERY 진입 → safe-stop reset → AUTONOMOUS 복귀
          - SAFE_OFF / SAFE_OFF2   : Servo OFF 상태 → SetRobotControl(1) 로 재 ON
          - EMERGENCY_STOP         : 펜던트의 비상정지 버튼 — 코드로 reset 불가, 사용자 안내
          - 그 외(STANDBY 등)      : 그대로 통과
        """
        s = self.get_robot_state()
        name = self._STATE_NAMES.get(s, f'UNKNOWN({s})')
        if verbose:
            print(f'   robot_state={s} ({name})')

        if s in (5, 9):   # SAFE_STOP / SAFE_STOP2 — 충돌·protective stop
            if verbose:
                print('   safe-stop 감지 → recovery 시퀀스 실행')
            for cli, name_ in [(self.cli_safety_mode, 'safety_mode'),
                               (self.cli_safe_reset, 'safe_stop_reset')]:
                if not cli.wait_for_service(timeout_sec=2.0):
                    print(f'   서비스 미응답: {name_}')
                    return False
            # 1) RECOVERY 진입
            m = SetSafetyMode.Request(); m.safety_mode = 2; m.safety_event = 0
            self._call(self.cli_safety_mode, m, timeout=3.0)
            time.sleep(0.5)
            # 2) safe-stop reset (program stop)
            rs = SetSafeStopResetType.Request(); rs.reset_type = 0
            self._call(self.cli_safe_reset, rs, timeout=3.0)
            time.sleep(0.5)
            # 3) AUTONOMOUS 복귀
            m2 = SetSafetyMode.Request(); m2.safety_mode = 1; m2.safety_event = 0
            self._call(self.cli_safety_mode, m2, timeout=3.0)
            time.sleep(0.5)
            new_s = self.get_robot_state()
            if verbose:
                print(f'   recovery 후 robot_state={new_s} ({self._STATE_NAMES.get(new_s, "?")})')
            return new_s in (1, 2)   # STANDBY/MOVING 이면 OK
        if s == 6:    # EMERGENCY_STOP — 펜던트 비상정지
            print('   !! EMERGENCY_STOP 상태 — 펜던트의 비상정지 버튼을 풀고 재시도하세요.')
            return False
        if s in (3, 10):   # SAFE_OFF / SAFE_OFF2
            if verbose:
                print('   Servo OFF 감지 → 재활성 시도')
            return True   # 다음 단계의 SetRobotControl(1) 가 처리
        return True   # 정상 (STANDBY 등)
```

### 활성화 루틴 (리커버리 → AUTONOMOUS → SERVO ON)

```python
    def activate_robot(self):
        """
        충돌·safe-stop 후 자동 복구 + AUTONOMOUS + SERVO_ON.
        매 실행 시작 때 호출되며 idempotent (이미 정상이면 빠르게 통과).
        """
        self._wait(self.cli_mode, 'set_robot_mode')
        self._wait(self.cli_ctrl, 'set_robot_control')

        # 0) 충돌·safe-stop 자동 복구
        self.recover_safety(verbose=True)

        # 1) AUTONOMOUS
        m = SetRobotMode.Request()
        m.robot_mode = 1
        r = self._call(self.cli_mode, m)
        if not (r and r.success):
            print('   (이미 AUTONOMOUS 모드일 가능성 — 계속 진행)')

        # 2) SERVO ON
        c = SetRobotControl.Request()
        c.robot_control = 1
        r = self._call(self.cli_ctrl, c)
        if not (r and r.success):
            print('   (이미 Servo ON 상태일 가능성 — 계속 진행)')

        # 3) 활성화 후 상태 한 번 더 확인
        final_s = self.get_robot_state()
        print(f'   활성화 후 robot_state={final_s} ({self._STATE_NAMES.get(final_s,"?")})')
```

---

## 적용 방법

1. **import 추가** — `SetSafetyMode`, `SetSafeStopResetType`, `GetRobotState` 를 `dsr_msgs2.srv` 에서 추가.
2. **`__init__` 에 클라이언트 3개 생성** — `cli_get_state`, `cli_safety_mode`, `cli_safe_reset`.
3. **`_STATE_NAMES`, `get_robot_state()`, `recover_safety()` 메서드를 클래스에 복사**.
4. **메인 진입점에서 한 줄 호출** — `set_robot_mode(1)` / `set_robot_control(1)` 호출 **직전에** `self.recover_safety()` 를 부르거나, 아예 위의 `activate_robot()` 로 통째로 대체.

```python
def main():
    rclpy.init()
    robot = MyRobot()
    robot.activate_robot()     # ← 충돌 신호 있어도 자동으로 풀고 시작
    # ... 이후 모션 명령 ...
```

---

## 확인 방법

1. 로봇을 손으로 살짝 막아 충돌 발생시키기 → 펜던트에 **SAFE_STOP** 표시 확인.
2. 스크립트 종료 후 재실행.
3. 콘솔에 `robot_state=5 (SAFE_STOP) → safe-stop 감지 → recovery 시퀀스 실행 → recovery 후 robot_state=1 (STANDBY)` 가 찍히고, 이후 모션 명령이 정상으로 들어가면 성공.

EMERGENCY_STOP(코드 6) 이 찍히면 **펜던트 비상정지 버튼이 눌려있는 상태** 이므로 손으로 풀어야 한다 — ROS2 서비스로는 복구 불가.
