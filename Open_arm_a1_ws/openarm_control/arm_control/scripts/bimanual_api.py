#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import time

class BimanualArmAPI(Node):
    """
    A simple API for the UI team to control the two arms in simulation.
    """
    def __init__(self):
        super().__init__('bimanual_arm_api')
        self.left_client = ActionClient(self, FollowJointTrajectory, '/left_arm_controller/follow_joint_trajectory')
        self.right_client = ActionClient(self, FollowJointTrajectory, '/right_arm_controller/follow_joint_trajectory')
        self.left_gripper_client = ActionClient(self, FollowJointTrajectory, '/left_gripper_controller/follow_joint_trajectory')
        self.right_gripper_client = ActionClient(self, FollowJointTrajectory, '/right_gripper_controller/follow_joint_trajectory')
        
        self.get_logger().info('Bimanual Arm & Gripper API Node Started')

    def move_arm(self, side, positions, duration=2.0):
        """
        side: 'left' or 'right'
        positions: list of 7 floats (radians)
        duration: time in seconds to reach the goal
        """
        client = self.left_client if side == 'left' else self.right_client
        
        if not client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(f'{side} arm controller not available!')
            return False

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [f'openarm_{side}_joint{i}' for i in range(1, 8)]
        
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        
        goal_msg.trajectory.points = [point]
        
        self.get_logger().info(f'Sending goal to {side} arm: {positions}')
        return client.send_goal_async(goal_msg)

    def move_gripper(self, side, position, duration=1.0):
        """
        side: 'left' or 'right'
        position: float (0.0 to 0.044)
        """
        client = self.left_gripper_client if side == 'left' else self.right_gripper_client
        
        if not client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(f'{side} gripper controller not available!')
            return False

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [f'openarm_{side}_finger_joint1']
        
        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        
        goal_msg.trajectory.points = [point]
        
        self.get_logger().info(f'Sending goal to {side} gripper: {position}')
        return client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    api = BimanualArmAPI()
    
    # Example: Move both arms to a 'ready' position
    # This is just for testing if run directly
    ready_pos = [0.0, -0.5, 0.0, 1.0, 0.0, 0.5, 0.0]
    
    print("Moving left arm to ready position...")
    api.move_arm('left', ready_pos)
    
    time.sleep(1)
    
    print("Moving right arm to ready position...")
    api.move_arm('right', ready_pos)

    rclpy.spin(api)
    api.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
