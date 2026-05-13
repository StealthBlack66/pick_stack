import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# 방금 만든 설정 파일 임포트
from doosan_config import get_calibrated_radians, moveit_action


class DoosanCalibratedMover(Node):
    def __init__(self):
        super().__init__('doosan_calibrated_mover')
        # namespace + MoveIt 컨트롤러 이름 자동 prefix
        # (예전 버전은 '/dsr_moveit_controller/...' 처럼 namespace 누락이라 다른 환경에서 깨짐)
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            moveit_action('follow_joint_trajectory'),
        )

    def move_to_angles(self, user_angles):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            print("!! 서버를 찾을 수 없습니다. (dsr_moveit_controller가 실행 중인지 확인하세요)")
            return None

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        point = JointTrajectoryPoint()
        # 사용자의 0도를 로봇의 특정 오프셋으로 자동 보정하여 전송
        point.positions = get_calibrated_radians(user_angles)
        point.time_from_start = Duration(sec=5, nanosec=0)
        
        goal_msg.trajectory.points = [point]
        print(f">> 전송되는 라디안 값: {[round(p, 4) for p in point.positions]}")
        
        return self._action_client.send_goal_async(goal_msg)

def main():
    rclpy.init()
    node = DoosanCalibratedMover()

    print("\n" + "="*60)
    print("  [두산 로봇 보정 제어 - 결과 확인형]")
    print("  각도를 입력하면 보정된 값으로 이동하며 성공 여부를 출력합니다.")
    print("="*60)

    try:
        line = input("\n>> 각도 입력 (예: 0 0 0 0 0 0): ")
        angles = [float(x) for x in line.split()]
        if len(angles) == 6:
            send_goal_future = node.move_to_angles(angles)
            if send_goal_future:
                rclpy.spin_until_future_complete(node, send_goal_future)
                goal_handle = send_goal_future.result()

                if not goal_handle.accepted:
                    print("!! 로봇이 명령을 거절했습니다. (Servo ON 상태인지 확인하세요)")
                else:
                    print(">> 명령 수락됨. 이동 중...")
                    result_future = goal_handle.get_result_async()
                    rclpy.spin_until_future_complete(node, result_future)
                    
                    status = result_future.result().status
                    if status == 4: # GoalStatus.STATUS_SUCCEEDED
                        print(">> 이동 성공!")
                    else:
                        print(f"!! 이동 실패 (상태 코드: {status})")
    except Exception as e:
        print(f"!! 오류 발생: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
