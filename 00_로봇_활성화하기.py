import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import SetRobotMode, SetRobotControl

from doosan_config import srv


class DoosanActivator(Node):
    def __init__(self):
        super().__init__('doosan_activator')
        self.mode_client = self.create_client(SetRobotMode, srv('system/set_robot_mode'))
        self.control_client = self.create_client(SetRobotControl, srv('system/set_robot_control'))

    def activate(self):
        print("\n[로봇 활성화 시작] 모터 및 제어 권한을 설정합니다...")
        
        # 1. 로봇 모드 설정 (1: AUTONOMOUS)
        while not self.mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('서비스 대기 중: /system/set_robot_mode')
        
        req_mode = SetRobotMode.Request()
        req_mode.robot_mode = 1 # AUTONOMOUS
        future = self.mode_client.call_async(req_mode)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() and future.result().success:
            print(">> 로봇 모드 설정 완료: AUTONOMOUS")
        else:
            print("!! 로봇 모드 설정 실패")
            return

        # 2. 제어 권한 설정
        while not self.control_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('서비스 대기 중: /system/set_robot_control')
        
        req_ctrl = SetRobotControl.Request()
        req_ctrl.robot_control = 1 # CONTROL_SERVO_ON
        future = self.control_client.call_async(req_ctrl)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() and future.result().success:
            print(">> 제어 권한(Servo ON) 설정 완료")
        else:
            print("!! 제어 권한 설정 실패")

def main():
    rclpy.init()
    node = DoosanActivator()
    node.activate()
    print("\n[활성화 완료] 이제 로봇을 조작할 수 있습니다.\n")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
