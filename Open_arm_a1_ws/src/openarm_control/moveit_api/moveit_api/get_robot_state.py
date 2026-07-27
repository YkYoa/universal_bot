#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
import math
import time

class RobotStatePrinter(Node):
    def __init__(self):
        super().__init__('robot_state_printer')
        
        self.joint_state = None
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.left_joints = [
            'openarm_left_joint1', 'openarm_left_joint2', 'openarm_left_joint3',
            'openarm_left_joint4', 'openarm_left_joint5', 'openarm_left_joint6', 'openarm_left_joint7'
        ]
        self.right_joints = [
            'openarm_right_joint1', 'openarm_right_joint2', 'openarm_right_joint3',
            'openarm_right_joint4', 'openarm_right_joint5', 'openarm_right_joint6', 'openarm_right_joint7'
        ]

    def joint_callback(self, msg):
        self.joint_state = msg

    def print_state(self):
        self.get_logger().info("Fetching robot state...")
        
        # Wait for joint states
        timeout = 5.0
        start = time.time()
        while self.joint_state is None and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            
        if self.joint_state is None:
            self.get_logger().error("Could not get /joint_states")
            return
            
        # Parse joint states
        try:
            la_joints_rad = [self.joint_state.position[self.joint_state.name.index(j)] for j in self.left_joints]
            ra_joints_rad = [self.joint_state.position[self.joint_state.name.index(j)] for j in self.right_joints]
            
            la_joints_deg = [math.degrees(r) for r in la_joints_rad]
            ra_joints_deg = [math.degrees(r) for r in ra_joints_rad]
        except ValueError as e:
            self.get_logger().error(f"Missing joint in state: {e}")
            return

        # Get Cartesian Poses
        la_pose = None
        ra_pose = None
        
        try:
            t = self.tf_buffer.lookup_transform('world', 'openarm_left_hand_tcp', rclpy.time.Time())
            la_pose = [
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z,
                t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w
            ]
        except Exception as e:
            self.get_logger().warn(f"Could not get left arm pose: {e}")
            
        try:
            t = self.tf_buffer.lookup_transform('world', 'openarm_right_hand_tcp', rclpy.time.Time())
            ra_pose = [
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z,
                t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w
            ]
        except Exception as e:
            self.get_logger().warn(f"Could not get right arm pose: {e}")

        # Print formatted
        print("\n" + "="*60)
        print(" 📋 CURRENT ROBOT STATE FOR sequences.yaml")
        print("="*60)
        
        print("\n--- JOINTS (Degrees) ---")
        print(f"laJoints: [{', '.join(f'{x:.2f}' for x in la_joints_deg)}]")
        print(f"raJoints: [{', '.join(f'{x:.2f}' for x in ra_joints_deg)}]")

        print("\n--- POSES [x, y, z, qx, qy, qz, qw] ---")
        if la_pose:
            print(f"laPose: [{', '.join(f'{x:.4f}' for x in la_pose)}]")
        if ra_pose:
            print(f"raPose: [{', '.join(f'{x:.4f}' for x in ra_pose)}]")
            
        print("\n" + "="*60 + "\n")

def main(args=None):
    rclpy.init(args=args)
    node = RobotStatePrinter()
    
    # Give TF buffer time to fill
    time.sleep(1.0)
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
        
    node.print_state()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
