# 다른 환경에서 강의 예제 실행하기

이 폴더의 두산 강의 예제는 자기 환경(다른 namespace, 다른 IP, 다른 fork 등) 에 맞춰
**한 군데에서만** 설정하면 모두 작동하도록 설계됐다.

## 30초 셋업

```bash
cd 로봇강의_예제/02_Doosan_Robot_제어

# 1) 템플릿 복사
cp .env.example .env

# 2) .env 열어서 자기 값으로 수정 (예: namespace, IP)
$EDITOR .env

# 3) (선택) python-dotenv 설치 — 미설치여도 동작 (아래 "대안" 참고)
pip install python-dotenv

# 4) 적용 확인
python3 doosan_config.py
```

`python3 doosan_config.py` 실행 결과에서 `NAMESPACE`, `ROBOT_IP` 등이
자기가 설정한 값으로 보이면 끝. 강의 예제 실행 시 자동으로 반영된다.

## 본인 환경 정보 확인하는 방법

자기 두산 셋업의 namespace / 패키지 이름을 모르면:

```bash
# namespace 확인 (예: dsr01, dsr01e0509)
ros2 node list | head

# 서비스 prefix 확인
ros2 service list | grep -E "set_robot_mode|set_robot_control" | head

# launch 패키지 확인
ros2 pkg list | grep -i "bringup\|dsr"
```

## `.env` 에 넣을 수 있는 값

| 환경변수 | 기본값 | 의미 |
| --- | --- | --- |
| `DSR_NAME` | `dsr01` | ROS namespace |
| `DSR_MODEL` | `e0509` | 로봇 모델 |
| `DSR_HOST` | `110.120.1.18` | 컨트롤러 IP |
| `DSR_RT_HOST` | `110.120.1.5` | PC 측 RT 호스트 IP |
| `DSR_SUBNET` | `110.120.1` | IP 자동 탐색용 서브넷 |
| `DSR_BRINGUP_PKG` | `dsr_bringup2` | bringup 패키지 이름 (fork 다른 경우) |
| `DSR_BRINGUP_LAUNCH` | `dsr_bringup2_moveit.launch.py` | launch 파일 이름 |
| `DSR_MOVEIT_CONTROLLER` | `dsr_moveit_controller` | MoveIt 컨트롤러 이름 |
| `DOOSAN_WS` | `~/doosan_ws` | 워크스페이스 경로 |
| `ROS_DISTRO` | `humble` | ROS distro |

## 대안: `.env` 안 쓰고 셸 export 만

`python-dotenv` 설치가 부담스러우면 셸에서 직접:

```bash
export DSR_NAME=dsr01e0509
export DSR_HOST=192.168.137.100
python3 09_원샷_캘리브레이션.py
```

`doosan_config.py` 가 환경변수를 우선 인식하므로 동일하게 동작한다.

## 충돌 진단 체크리스트

스크립트가 "서비스 미응답" 또는 "namespace 못 찾음" 류 오류로 실패할 때:

```bash
# 1) 현재 적용된 설정 확인
python3 doosan_config.py

# 2) 실제 떠있는 서비스 prefix 확인
ros2 service list | head

# 3) 두 값을 비교 — namespace 가 다르면 .env 의 DSR_NAME 수정
```

`srv('system/set_robot_mode')` 결과(예: `/dsr01/...`)가 실제 서비스 prefix와
일치해야 한다. 안 맞으면 `DSR_NAME` 만 고치면 끝.

## 한계

이 프레임워크가 **자동 처리하는 것**:
- namespace, IP, RT host, 모델, 패키지 이름, launch 파일 이름, MoveIt 컨트롤러 이름

이 프레임워크가 **자동 처리 못 하는 것**:
- 메시지 패키지 이름이 `dsr_msgs2` 가 아닌 다른 fork (예: `doosan_msgs`):
  Python `import` 문 자체가 깨지므로 코드 수정 필요. 강의 예제는
  `dsr_msgs2` 를 가정한다.
- 메시지 타입의 필드/이름이 다른 fork: 호환되지 않으면 직접 수정 필요.

위 두 경우는 fork 차이가 크다는 의미라 프레임워크 한 곳을 고친다고 풀리지 않는다.
