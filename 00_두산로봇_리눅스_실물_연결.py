import os
import subprocess
import time
import sys

from doosan_config import (
    NAMESPACE, ROBOT_MODEL, ROBOT_IP, RT_HOST,
    BRINGUP_PKG, BRINGUP_LAUNCH, DOOSAN_WS, ROS_DISTRO,
)


def connect_doosan_linux():
    """
    [리눅스 전용 두산 로봇 실물 연결 스크립트]
    1. 모든 기존 ROS 프로세스를 초기화합니다.
    2. 실물 로봇(E0509)의 IP로 드라이버를 실행합니다.
    3. 새로운 터미널 세션을 열어 백그라운드에서 로봇 엔진이 돌아가도록 합니다.
    """
    
    print("\n" + "="*70)
    print("  [Doosan Robot Linux Real-Connect System]")
    print("  리눅스 환경에서 로봇과의 연결을 초기화하고 새롭게 시작합니다.")
    print("="*70)

    # 1. 이전 프로세스 및 메모리 정리
    print("\n[단계 1] 기존 로봇 통신 프로세스 정리 중...")
    # 강제 종료 명령어 리스트
    cmd_cleanup = "pkill -9 -f dsr; pkill -9 -f rviz2; pkill -9 -f DRCF; pkill -9 -f ros2"
    os.system(cmd_cleanup)
    os.system("docker stop $(docker ps -a -q) 2>/dev/null")
    os.system("docker rm $(docker ps -a -q) 2>/dev/null")
    time.sleep(3)
    print(">> 정리 완료.")

    # 2. 로봇 연결 설정 — doosan_config 또는 .env 에서 가져옴
    print(f"\n[단계 2] 로봇 연결 설정 확인")
    print(f"   - Namespace: {NAMESPACE}")
    print(f"   - 모델: {ROBOT_MODEL}")
    print(f"   - IP 주소: {ROBOT_IP}")
    print(f"   - RT host: {RT_HOST}")

    # 3. ROS 2 실행 명령어 구성
    # 환경 변수 소싱(Sourcing) 및 런치 파일 실행
    ros_cmd = (
        f"source /opt/ros/{ROS_DISTRO}/setup.bash && "
        f"source {DOOSAN_WS}/install/setup.bash && "
        f"ros2 launch {BRINGUP_PKG} {BRINGUP_LAUNCH} "
        f"name:={NAMESPACE} model:={ROBOT_MODEL} mode:=real "
        f"host:={ROBOT_IP} rt_host:={RT_HOST}"
    )

    # 리눅스 터미널(gnome-terminal)을 새로 열어 명령어 실행
    print("\n[단계 3] 새로운 터미널에서 로봇 엔진 실행 중...")
    terminal_cmd = f'gnome-terminal -- bash -c "{ros_cmd}; exec bash"'
    
    try:
        subprocess.Popen(terminal_cmd, shell=True)
        print(">> 로봇 연결용 새 터미널 창이 성공적으로 열렸습니다.")
    except Exception as e:
        print(f"!! 터미널 실행 중 오류 발생: {e}")
        return

    # 4. 연결 대기 루프
    print("\n[단계 4] 시스템 안정화 대기 (약 20초)")
    print(">> 이 시간 동안 로봇 컨트롤러와 하드웨어 엔진이 동기화됩니다.")
    
    for i in range(20, 0, -1):
        sys.stdout.write(f"\r연결 대기 중... {i}초 ")
        sys.stdout.flush()
        time.sleep(1)

    print("\n\n" + "="*70)
    print("  [연결 초기화 완료]")
    print("  1. 새로 열린 터미널에 에러 메시지가 없는지 확인하세요.")
    print("  2. RViz 창에 로봇 모델이 정상적으로 나타났다면 성공입니다.")
    print("  3. 이제 '00_로봇_활성화하기.py'를 실행하여 모터를 켜주세요.")
    print("="*70 + "\n")

if __name__ == "__main__":
    connect_doosan_linux()
