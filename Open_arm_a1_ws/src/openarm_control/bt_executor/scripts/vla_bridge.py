#!/usr/bin/env python3
import sys
import os
import time
import base64
import io
import json
import requests
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped, Quaternion
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.action import ActionServer, GoalResponse, CancelResponse, ActionClient
from openarm_messages.action import QueryVLA, ExecuteSkill
from rclpy.callback_groups import ReentrantCallbackGroup

# Try loading cv2 and cv_bridge for camera subscriptions
try:
    import cv2
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False

class VLABridgeNode(Node):
    def __init__(self):
        super().__init__('vla_bridge_node')
        
        # Parameters
        self.declare_parameter('vla_host', 'localhost')
        self.declare_parameter('vla_port', 10000)
        self.declare_parameter('mode', 'real')  # 'real' or 'mock'
        self.declare_parameter('arm', 'left')   # 'left' or 'right'
        self.declare_parameter('instruction', 'point to the white bottle, avoid the table')
        self.declare_parameter('pos_scale', 0.1) # scale normalized translation delta to meters
        self.declare_parameter('rot_scale', 1.0) # scale rotation axis-angle
        self.declare_parameter('predict_mode', 'batch') # 'batch' or 'stream'
        self.declare_parameter('front_camera_topic', '/camera_front/image_raw')
        self.declare_parameter('left_camera_topic', '/camera_left/image_raw')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('planner_profile', 'linear_approach')
        self.declare_parameter('planning_mode', 'normal')
        self.declare_parameter('position_only', False)
        self.declare_parameter('velocity_override', 0.0)
        self.declare_parameter('planning_frame', 'openarm_base_link')
        self.declare_parameter('prepare_vla_ready', True)
        self.declare_parameter('vla_ready_once', True)
        self.declare_parameter('vla_ready_named_pose', 'vla_ready')
        self.declare_parameter('vla_ready_planner_profile', 'safe_rrt')
        self.declare_parameter('save_plan', False)
        
        self.vla_host = self.get_parameter('vla_host').value
        self.vla_port = self.get_parameter('vla_port').value
        self.mode = self.get_parameter('mode').value
        self.arm_side = self.get_parameter('arm').value
        self.instruction = self.get_parameter('instruction').value
        self.pos_scale = self.get_parameter('pos_scale').value
        self.rot_scale = self.get_parameter('rot_scale').value
        self.predict_mode = self.get_parameter('predict_mode').value
        self.front_camera_topic = self.get_parameter('front_camera_topic').value
        self.left_camera_topic = self.get_parameter('left_camera_topic').value
        self.joint_states_topic = self.get_parameter('joint_states_topic').value
        self.planner_profile = self.get_parameter('planner_profile').value
        self.planning_mode = self.get_parameter('planning_mode').value
        self.position_only = self.get_parameter('position_only').value
        self.velocity_override = self.get_parameter('velocity_override').value
        self.planning_frame = self.get_parameter('planning_frame').value
        self.prepare_vla_ready = self.get_parameter('prepare_vla_ready').value
        self.vla_ready_once = self.get_parameter('vla_ready_once').value
        self.vla_ready_named_pose = self.get_parameter('vla_ready_named_pose').value
        self.vla_ready_planner_profile = self.get_parameter('vla_ready_planner_profile').value
        self.save_plan = self.get_parameter('save_plan').value
        
        # TF2 buffer and listener to track end effector pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.get_logger().info(f"VLA Bridge Node started in '{self.mode}' mode with '{self.predict_mode}' predict mode.")
        self.get_logger().info(f"Target VLA Model URL: http://{self.vla_host}:{self.vla_port}")
        
        # CV Bridge
        self.bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None
        if not CV_BRIDGE_AVAILABLE:
            self.get_logger().warn("cv_bridge or cv2 is not available. Live camera frames will fallback to dummy images.")
            
        # Image buffers
        self.front_image_msg = None
        self.left_image_msg = None
        
        self.cb_group = ReentrantCallbackGroup()

        # Subscriptions & Publishers
        # Isaac Sim publishes joint_states with BEST_EFFORT QoS.
        # Using RELIABLE (depth=10) causes zero DDS matches → current_joints is always None.
        joint_states_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.joint_states_sub = self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_states_callback,
            joint_states_qos,
            callback_group=self.cb_group
        )
        
        # Isaac Sim publishes camera images with BEST_EFFORT reliability and VOLATILE durability.
        # Using RELIABLE QoS (default depth=10) would cause zero matches → fallback to dummy.
        # Must use BEST_EFFORT to match Isaac Sim's publisher QoS.
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.front_image_sub = self.create_subscription(
            Image,
            self.front_camera_topic,
            self.front_image_callback,
            camera_qos,
            callback_group=self.cb_group
        )
        
        self.left_image_sub = self.create_subscription(
            Image,
            self.left_camera_topic,
            self.left_image_callback,
            camera_qos,
            callback_group=self.cb_group
        )
        self.get_logger().info(f"Subscribed to front camera: '{self.front_camera_topic}' (BEST_EFFORT QoS)")
        self.get_logger().info(f"Subscribed to left camera:  '{self.left_camera_topic}' (BEST_EFFORT QoS)")
        
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/bt_executor/vla_grasp_pose',
            10
        )
        
        # Planning scene publisher to modify the Allowed Collision Matrix (ACM)
        from moveit_msgs.msg import PlanningScene
        self.planning_scene_pub = self.create_publisher(
            PlanningScene,
            '/planning_scene',
            10
        )
        # Timer to continuously publish the Allowed Collision Matrix at 1Hz
        self.acm_timer = self.create_timer(1.0, self.publish_acm)
        
        # Action Server
        self._action_server = ActionServer(
            self,
            QueryVLA,
            '/vla_bridge/query',
            execute_callback=self.execute_action_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group
        )
        self.get_logger().info("QueryVLA Action Server started on topic '/vla_bridge/query'")

        # Action Client for executing skills
        self._execute_skill_client = ActionClient(
            self,
            ExecuteSkill,
            'robot_skills_server/execute_skill',
            callback_group=self.cb_group
        )
        
        # State variables
        self.current_joints = None
        self.joint_names = []
        self.query_active = False
        self.last_gripper_state = None
        self.vla_ready_prepared = False

        # Timer to trigger query for testing (or you can trigger it via a service) has been removed.

    def joint_states_callback(self, msg):
        # Store latest joints for the selected arm
        self.joint_names = msg.name
        self.current_joints = msg.position

    def front_image_callback(self, msg):
        if self.front_image_msg is None:
            self.get_logger().info(f"First front camera frame received! Encoding: {msg.encoding}, size: {msg.width}x{msg.height}")
        self.front_image_msg = msg

    def left_image_callback(self, msg):
        if self.left_image_msg is None:
            self.get_logger().info(f"First left camera frame received! Encoding: {msg.encoding}, size: {msg.width}x{msg.height}")
        self.left_image_msg = msg

    def get_arm_joints(self):
        if self.current_joints is None:
            return None
            
        prefix = f"openarm_{self.arm_side}_joint"
        arm_positions = []
        for i in range(1, 8):
            name = f"{prefix}{i}"
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                arm_positions.append(self.current_joints[idx])
        
        # Define limits: (lower, upper)
        if self.arm_side == "left":
            limits = [
                (-3.490659, 1.396263),
                (-3.316125, 0.174532),
                (-1.570796, 1.570796),
                (0.0, 2.443461),
                (-1.570796, 1.570796),
                (-0.785398, 0.785398),
                (-1.570796, 1.570796),
            ]
        else: # right arm
            limits = [
                (-1.396263, 3.490659),
                (-0.174532, 3.316125),
                (-1.570796, 1.570796),
                (0.0, 2.443461),
                (-1.570796, 1.570796),
                (-0.785398, 0.785398),
                (-1.570796, 1.570796),
            ]

        # Apply a small gap/buffer (e.g., 0.02 radians) to keep joint states strictly within limits
        gap = 0.02
        clipped_positions = []
        for i, pos in enumerate(arm_positions):
            if i < len(limits):
                low, high = limits[i]
                clipped_val = max(low + gap, min(high - gap, pos))
                clipped_positions.append(clipped_val)
            else:
                clipped_positions.append(pos)
        arm_positions = clipped_positions

        # If gripper finger joint exists, read it too
        gripper_name = f"openarm_{self.arm_side}_finger_joint1"
        gripper_pos = 0.0
        if gripper_name in self.joint_names:
            idx = self.joint_names.index(gripper_name)
            gripper_pos = self.current_joints[idx]
            # Clip gripper joint with a small gap (limits are 0.0 to 0.044)
            gripper_pos = max(0.001, min(0.043, gripper_pos))
            
        return arm_positions, gripper_pos

    def get_latest_images_b64(self):
        """
        Retrieves the latest front and left images, converts them using cv_bridge,
        resizes to 224x224, and encodes as base64.
        Falls back to dummy images if they are not available.
        """
        img1_b64 = None
        img2_b64 = None
        
        if self.bridge is not None:
            # Front camera
            if self.front_image_msg is not None:
                try:
                    # Convert to RGB
                    cv_img = self.bridge.imgmsg_to_cv2(self.front_image_msg, desired_encoding='rgb8')
                    # Resize to 224x224
                    cv_img_resized = cv2.resize(cv_img, (224, 224))
                    
                    buf = io.BytesIO()
                    np.save(buf, cv_img_resized)
                    img1_b64 = base64.b64encode(buf.getvalue()).decode()
                    self.get_logger().info("Successfully encoded live front camera frame.")
                except Exception as e:
                    self.get_logger().error(f"Failed to process front image: {e}")
            
            # Left camera
            if self.left_image_msg is not None:
                try:
                    # Convert to RGB
                    cv_img = self.bridge.imgmsg_to_cv2(self.left_image_msg, desired_encoding='rgb8')
                    # Resize to 224x224
                    cv_img_resized = cv2.resize(cv_img, (224, 224))
                    
                    buf = io.BytesIO()
                    np.save(buf, cv_img_resized)
                    img2_b64 = base64.b64encode(buf.getvalue()).decode()
                    self.get_logger().info("Successfully encoded live left camera frame.")
                except Exception as e:
                    self.get_logger().error(f"Failed to process left image: {e}")
                    
        if img1_b64 is None:
            self.get_logger().error("Front camera frame not available.")
            return None
            
        if img2_b64 is None:
            self.get_logger().error("Left camera frame not available.")
            return None
            
        return [img1_b64, img2_b64]

    def axis_angle_to_quaternion(self, r):
        angle = np.linalg.norm(r)
        if angle < 1e-6:
            return [0.0, 0.0, 0.0, 1.0]
        axis = r / angle
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        return [
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            np.cos(half_angle)
        ]

    def multiply_quaternions(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ]

    def compute_target_pose(self, action, curr_pos, curr_q):
        # Rotate by 180 degrees around Z axis to match body orientation
        delta_pos = np.array([-action[0], -action[1], action[2]]) * self.pos_scale
        delta_rot_vec = np.array([-action[3], -action[4], action[5]]) * self.rot_scale
        
        target_pos = curr_pos + delta_pos
        q_delta = self.axis_angle_to_quaternion(delta_rot_vec)
        target_q_list = self.multiply_quaternions(curr_q, q_delta)
        
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = target_pos[0]
        msg.pose.position.y = target_pos[1]
        msg.pose.position.z = target_pos[2]
        msg.pose.orientation.x = target_q_list[0]
        msg.pose.orientation.y = target_q_list[1]
        msg.pose.orientation.z = target_q_list[2]
        msg.pose.orientation.w = target_q_list[3]
        return msg, target_pos, target_q_list

    def transform_pose(self, pose_msg, target_frame):
        if pose_msg.header.frame_id == target_frame:
            return pose_msg
            
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                pose_msg.header.frame_id,
                rclpy.time.Time()
            )
        except Exception as ex:
            self.get_logger().error(f"Failed to lookup transform from {pose_msg.header.frame_id} to {target_frame}: {ex}")
            return None
            
        # Extract transform translation and rotation
        t_x = trans.transform.translation.x
        t_y = trans.transform.translation.y
        t_z = trans.transform.translation.z
        
        rx = trans.transform.rotation.x
        ry = trans.transform.rotation.y
        rz = trans.transform.rotation.z
        rw = trans.transform.rotation.w
        
        # Original pose
        px = pose_msg.pose.position.x
        py = pose_msg.pose.position.y
        pz = pose_msg.pose.position.z
        
        qx = pose_msg.pose.orientation.x
        qy = pose_msg.pose.orientation.y
        qz = pose_msg.pose.orientation.z
        qw = pose_msg.pose.orientation.w
        
        # Apply rotation then translation: target = R * source + T
        v = np.array([px, py, pz])
        q_xyz = np.array([rx, ry, rz])
        q_w = rw
        
        tmp = np.cross(q_xyz, v) + q_w * v
        v_rot = v + 2.0 * np.cross(q_xyz, tmp)
        
        # Translate
        transformed_pos = v_rot + np.array([t_x, t_y, t_z])
        
        # Multiply quaternions for orientation: target_q = trans_q * pose_q
        transformed_q = self.multiply_quaternions([rx, ry, rz, rw], [qx, qy, qz, qw])
        
        msg = PoseStamped()
        msg.header.stamp = pose_msg.header.stamp
        msg.header.frame_id = target_frame
        msg.pose.position.x = transformed_pos[0]
        msg.pose.position.y = transformed_pos[1]
        msg.pose.position.z = transformed_pos[2]
        msg.pose.orientation.x = transformed_q[0]
        msg.pose.orientation.y = transformed_q[1]
        msg.pose.orientation.z = transformed_q[2]
        msg.pose.orientation.w = transformed_q[3]
        return msg

    def publish_acm(self):
        try:
            from moveit_msgs.msg import PlanningScene, AllowedCollisionMatrix, AllowedCollisionEntry
            
            scene = PlanningScene()
            scene.is_diff = True
            
            acm = AllowedCollisionMatrix()
            links = [
                "openarm_base_link",
                "openarm_base_bl_caster_link",
                "openarm_base_bl_wheel_link",
                "openarm_base_br_caster_link",
                "openarm_base_br_wheel_link",
                "openarm_base_fl_caster_link",
                "openarm_base_fl_wheel_link",
                "openarm_base_fr_caster_link",
                "openarm_base_fr_wheel_link",
                "openarm_base_left_drive_wheel_link",
                "openarm_base_right_drive_wheel_link",
                "openarm_body_link0",
                "openarm_left_link0",
                "openarm_left_link1",
                "openarm_left_link2",
                "openarm_left_link3",
                "openarm_left_link4",
                "openarm_left_link5",
                "openarm_left_link6",
                "openarm_left_link7",
                "openarm_left_hand_tcp",
                "openarm_left_left_finger",
                "openarm_left_right_finger",
                "openarm_right_link0",
                "openarm_right_link1",
                "openarm_right_link2",
                "openarm_right_link3",
                "openarm_right_link4",
                "openarm_right_link5",
                "openarm_right_link6",
                "openarm_right_link7",
                "openarm_right_hand_tcp",
                "openarm_right_left_finger",
                "openarm_right_right_finger"
            ]
            acm.entry_names = links
            for i in range(len(links)):
                entry = AllowedCollisionEntry()
                entry.enabled = [True] * len(links)
                acm.entry_values.append(entry)
                
            scene.allowed_collision_matrix = acm
            self.planning_scene_pub.publish(scene)
        except Exception as e:
            self.get_logger().error(f"Failed to publish ACM: {e}")

    def print_action_step(self, step_idx, action, source):
        joint_names = [
            "EE delta x", 
            "EE delta y", 
            "EE delta z", 
            "EE delta rot x", 
            "EE delta rot y", 
            "EE delta rot z", 
            "Gripper"
        ]
        self.get_logger().info(f"[{source}] Step {step_idx}:")
        for j_idx, val in enumerate(action):
            name = joint_names[j_idx] if j_idx < len(joint_names) else f"joint_{j_idx}"
            if j_idx < 3:
                unit = "(pos delta)"
            elif j_idx < 6:
                unit = "rad"
            else:
                unit = "[-1=open, 1=close]"
            self.get_logger().info(f"    {name:<14}: {val:>8.3f} {unit}")

    def red_text(self, text):
        return f"\033[1;31m{text}\033[0m"

    def format_pose(self, pose_stamped):
        p = pose_stamped.pose.position
        q = pose_stamped.pose.orientation
        return (
            f"frame={pose_stamped.header.frame_id}, "
            f"pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f}), "
            f"quat=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})"
        )

    def log_step_failure(self, step_idx, total_steps, pose_stamped, reason):
        if total_steps is None:
            step_label = f"step {step_idx}"
        else:
            step_label = f"step {step_idx + 1}/{total_steps} (model index {step_idx})"
        self.get_logger().error(self.red_text(
            f"VLA waypoint planning failed at {step_label}: {reason}. "
            f"Target {self.format_pose(pose_stamped)}"
        ))

    def execute_cartesian_move(self, waypoints, step_idx=None, total_steps=None):
        if not self._execute_skill_client.wait_for_server(timeout_sec=5.0):
            msg = "ExecuteSkill action server not available!"
            if step_idx is not None and waypoints:
                self.log_step_failure(step_idx, total_steps, waypoints[-1], msg)
            else:
                self.get_logger().error(self.red_text(msg))
            return False
            
        goal_msg = ExecuteSkill.Goal()
        goal_msg.skill_name = "cartesian_move"
        goal_msg.arm = f"{self.arm_side}_arm"
        profile = self.planner_profile
        if not self.save_plan:
            profile += "_no_save"
        goal_msg.planner_profile = profile
        goal_msg.planning_mode = self.planning_mode
        goal_msg.waypoints = waypoints
        goal_msg.velocity_override = self.velocity_override
        goal_msg.position_only = self.position_only
        
        self.get_logger().info(f"Sending cartesian_move skill with {len(waypoints)} waypoints...")
        
        # Send goal
        send_goal_future = self._execute_skill_client.send_goal_async(goal_msg)
        
        # Wait for goal acceptance
        while not send_goal_future.done():
            time.sleep(0.02)
            
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            msg = "Cartesian move goal rejected by server!"
            if step_idx is not None and waypoints:
                self.log_step_failure(step_idx, total_steps, waypoints[-1], msg)
            else:
                self.get_logger().error(self.red_text(msg))
            return False
            
        self.get_logger().info("Cartesian move goal accepted. Waiting for execution result...")
        
        # Get result
        get_result_future = goal_handle.get_result_async()
        while not get_result_future.done():
            time.sleep(0.02)
            
        wrapped_result = get_result_future.result()
        result = wrapped_result.result
        if not result.success:
            reason = result.error_message or f"action status={wrapped_result.status}"
            if step_idx is not None and waypoints:
                self.log_step_failure(step_idx, total_steps, waypoints[-1], reason)
            else:
                self.get_logger().error(self.red_text(f"Cartesian move failed: {reason}"))
            return False

        return True

    def execute_vla_ready_skill(self, goal_msg, label):
        if not self._execute_skill_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"ExecuteSkill action server not available for {label}!")
            return False

        self.get_logger().info(f"Sending {label} skill...")

        send_goal_future = self._execute_skill_client.send_goal_async(goal_msg)
        while not send_goal_future.done():
            time.sleep(0.02)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"{label} goal rejected by server!")
            return False

        get_result_future = goal_handle.get_result_async()
        while not get_result_future.done():
            time.sleep(0.02)

        wrapped_result = get_result_future.result()
        result = wrapped_result.result
        if not result.success:
            reason = result.error_message or f"action status={wrapped_result.status}"
            self.get_logger().error(f"{label} failed: {reason}")
            return False

        return True

    def prepare_vla_ready_pose(self):
        if not self.prepare_vla_ready:
            return True
        if self.vla_ready_once and self.vla_ready_prepared:
            return True

        arm = f"{self.arm_side}_arm"
        self.get_logger().info("Preparing VLA ready pose (Pose 1 -> Pose 2)...")

        if self.arm_side == "left":
            ready_1_joint_targets = [
                0.0017,
                -0.6105,
                -0.0579,
                0.1424,
                -0.0420,
                -0.6117,
                0.1721,
            ]

            profile = self.vla_ready_planner_profile
            if not self.save_plan:
                profile += "_no_save"

            ready_1_goal = ExecuteSkill.Goal()
            ready_1_goal.skill_name = "move_to_joint"
            ready_1_goal.arm = arm
            ready_1_goal.planner_profile = profile
            ready_1_goal.planning_mode = self.planning_mode
            ready_1_goal.joint_targets = ready_1_joint_targets
            ready_1_goal.velocity_override = self.velocity_override
            ready_1_goal.position_only = False

            if not self.execute_vla_ready_skill(ready_1_goal, "VLA ready 1"):
                return False
        else:
            self.get_logger().warn(
                f"VLA ready 1 joint targets are only configured for the left arm; "
                f"skipping ready 1 for arm='{self.arm_side}'."
            )

        profile = self.vla_ready_planner_profile
        if not self.save_plan:
            profile += "_no_save"

        ready_2_goal = ExecuteSkill.Goal()
        ready_2_goal.skill_name = "move_to_named_pose"
        ready_2_goal.arm = arm
        ready_2_goal.planner_profile = profile
        ready_2_goal.planning_mode = self.planning_mode
        ready_2_goal.named_pose = self.vla_ready_named_pose
        ready_2_goal.velocity_override = self.velocity_override
        ready_2_goal.position_only = False

        if not self.execute_vla_ready_skill(
            ready_2_goal,
            f"VLA ready 2 named pose '{self.vla_ready_named_pose}'"
        ):
            return False

        self.vla_ready_prepared = True
        self.get_logger().info("VLA ready pose preparation completed.")
        return True

    def execute_gripper_skill(self, action_name):
        if not self._execute_skill_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"ExecuteSkill action server not available for gripper {action_name}!")
            return False
            
        goal_msg = ExecuteSkill.Goal()
        goal_msg.skill_name = action_name  # "open_gripper" or "close_gripper"
        goal_msg.arm = f"{self.arm_side}_arm"
        
        self.get_logger().info(f"Sending {action_name} skill to gripper...")
        
        send_goal_future = self._execute_skill_client.send_goal_async(goal_msg)
        while not send_goal_future.done():
            time.sleep(0.02)
            
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"Gripper skill {action_name} goal rejected!")
            return False
            
        get_result_future = goal_handle.get_result_async()
        while not get_result_future.done():
            time.sleep(0.02)
            
        return get_result_future.result().result.success

    def execute_cartesian_move_stepwise(self, waypoints, actions=None, start_pose_msg=None):
        total_steps = len(waypoints)

        # --- Gripper majority vote ---
        # Determine gripper intent ONCE across all steps instead of toggling per step.
        # VLA often returns hard ±1 values that alternate; voting avoids rapid open/close.
        gripper_command = None
        if actions is not None and len(actions) > 0:
            gripper_vals = [a[6] for a in actions if len(a) > 6]
            closes = sum(1 for v in gripper_vals if v > 0.5)
            opens  = sum(1 for v in gripper_vals if v < -0.5)
            self.get_logger().info(
                f"Gripper vote: {closes} close vs {opens} open out of {len(gripper_vals)} steps. "
                f"Values: {[round(v, 2) for v in gripper_vals]}"
            )
            if closes > opens:
                gripper_command = "close_gripper"
            elif opens > closes:
                gripper_command = "open_gripper"
            # else: tie → no gripper command this round

        for step_idx, waypoint in enumerate(waypoints):
            self.get_logger().info(
                f"Planning/executing VLA waypoint {step_idx + 1}/{total_steps}: "
                f"{self.format_pose(waypoint)}"
            )
            if not self.execute_cartesian_move([waypoint], step_idx=step_idx, total_steps=total_steps):
                self.get_logger().warn("Execution failed! Returning to previous successful stage...")
                if step_idx > 0:
                    prev_waypoint = waypoints[step_idx - 1]
                    self.get_logger().info(f"Returning to previous successful waypoint {step_idx}: {self.format_pose(prev_waypoint)}")
                    self.execute_cartesian_move([prev_waypoint])
                elif start_pose_msg is not None:
                    self.get_logger().info(f"Returning to starting pose: {self.format_pose(start_pose_msg)}")
                    self.execute_cartesian_move([start_pose_msg])
                return False, step_idx

        # Execute gripper command once, after all arm waypoints succeed, only if state changed
        if gripper_command is not None and gripper_command != self.last_gripper_state:
            self.get_logger().info(f"Executing gripper command (majority vote): {gripper_command}")
            if self.execute_gripper_skill(gripper_command):
                self.last_gripper_state = gripper_command
        elif gripper_command == self.last_gripper_state:
            self.get_logger().info(f"Gripper already in target state '{gripper_command}', skipping.")
        else:
            self.get_logger().info("Gripper vote was a tie or no data — no gripper command sent.")

        return True, None

    def query_vla_model(self, instruction_override=None):
        current_instr = self.get_parameter('instruction').value
        instr = instruction_override if (instruction_override is not None and instruction_override != "") else current_instr
        
        if self.query_active:
            return False, None, "Query is already active."
        self.query_active = True
        
        # Publish ACM to ensure collisions are allowed for the demo
        self.publish_acm()

        if not self.prepare_vla_ready_pose():
            self.query_active = False
            return False, None, "Failed to prepare VLA ready pose."
        
        # Read current joints just for logging/compatibility state
        joint_data = self.get_arm_joints()
        if joint_data is None:
            self.get_logger().warn("Joint states not available yet.")
            self.query_active = False
            return False, None, "Joint states not available yet."
            
        arm_positions, gripper_pos = joint_data
        
        # Compatibility state vector (6-DoF: 5 joints in degrees + 1 gripper)
        import math
        state_deg = [
            math.degrees(arm_positions[0]), # shoulder_pan
            math.degrees(arm_positions[1]), # shoulder_lift
            math.degrees(arm_positions[2]), # elbow_flex
            math.degrees(arm_positions[3]), # wrist_flex
            math.degrees(arm_positions[4]), # wrist_roll
            gripper_pos * 100               # gripper opening scale
        ]
        
        self.get_logger().info(f"Querying model with state (deg): {state_deg} and instruction: '{instr}'")

        # Query initial TF pose at start of query
        try:
            trans = self.tf_buffer.lookup_transform(
                'world',
                f'openarm_{self.arm_side}_hand_tcp',
                rclpy.time.Time()
            )
            curr_pos = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            curr_q = [
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ]
            self.get_logger().info(f"Current TCP pose from TF: pos={curr_pos}, q={curr_q}")
        except Exception as ex:
            self.get_logger().error(f"TF lookup failed: {ex}. Cannot proceed without current pose.")
            self.query_active = False
            return False, None, f"TF lookup failed: {ex}"

        images = self.get_latest_images_b64()
        if images is None:
            self.query_active = False
            return False, None, "Camera images not available."
        img1_b64, img2_b64 = images

        if self.predict_mode == 'stream':
            url = f"http://{self.vla_host}:{self.vla_port}/predict_base64_stream"
            try:
                payload = {
                    "base64_rgb": [img1_b64, img2_b64],
                    "state": state_deg,
                    "instr": instr
                }
                
                self.get_logger().info("Sending streaming HTTP request to real VLA model...")
                response = requests.post(url, json=payload, stream=True, timeout=15.0)
                if response.status_code == 200:
                    last_step_pose = None
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line.decode('utf-8'))
                            if "value" in data:
                                action = np.array(data["value"][0])
                                step_idx = data["index"]
                                self.print_action_step(step_idx, action, source="Real VLA Model Stream")
                                
                                # Execute this step immediately!
                                step_pose, _, _ = self.compute_target_pose(action, curr_pos, curr_q)
                                transformed_step_pose = self.transform_pose(step_pose, self.planning_frame)
                                if transformed_step_pose is None:
                                    self.query_active = False
                                    return False, None, f"Failed to transform VLA step to planning frame {self.planning_frame}"
                                if not self.execute_cartesian_move([transformed_step_pose], step_idx=step_idx):
                                    self.query_active = False
                                    return False, None, f"Trajectory execution failed at stream step {step_idx}."
                                last_step_pose = transformed_step_pose
                            elif "time_taken" in data:
                                self.get_logger().info(f"Stream generation finished. Time taken: {data['time_taken']}s")
                    
                    self.query_active = False
                    if last_step_pose is not None:
                        self.pose_pub.publish(last_step_pose)
                        return True, last_step_pose, ""
                    else:
                        return False, None, "No action received in stream."
                else:
                    err = f"HTTP Error {response.status_code}: {response.text}"
                    self.get_logger().error(err)
                    self.query_active = False
                    return False, None, err
            except Exception as e:
                err = f"Failed to query VLA model API stream: {str(e)}"
                self.get_logger().error(err)
                self.query_active = False
                return False, None, err
        else:
            # Batch mode
            url = f"http://{self.vla_host}:{self.vla_port}/predict_base64"
            try:
                payload = {
                    "base64_rgb": [img1_b64, img2_b64],
                    "state": state_deg,
                    "instr": instr
                }
                
                self.get_logger().info("Sending HTTP request to real VLA model...")
                response = requests.post(url, json=payload, timeout=8.0)
                if response.status_code == 200:
                    actions = response.json()
                    self.print_actions_sequence(actions, source="Real VLA Model API")
                    
                    # Generate waypoints sequence
                    waypoints = []
                    for act in actions:
                        wp, _, _ = self.compute_target_pose(act, curr_pos, curr_q)
                        transformed_wp = self.transform_pose(wp, self.planning_frame)
                        if transformed_wp is None:
                            self.query_active = False
                            return False, None, f"Failed to transform VLA waypoint to planning frame {self.planning_frame}"
                        waypoints.append(transformed_wp)
                    
                    # Execute trajectory on robot
                    self.get_logger().info("Executing VLA trajectory waypoints on robot...")
                    
                    # Create start pose message in planning frame for fallback recovery
                    start_pose_world = PoseStamped()
                    start_pose_world.header.stamp = self.get_clock().now().to_msg()
                    start_pose_world.header.frame_id = 'world'
                    start_pose_world.pose.position.x = curr_pos[0]
                    start_pose_world.pose.position.y = curr_pos[1]
                    start_pose_world.pose.position.z = curr_pos[2]
                    start_pose_world.pose.orientation.x = curr_q[0]
                    start_pose_world.pose.orientation.y = curr_q[1]
                    start_pose_world.pose.orientation.z = curr_q[2]
                    start_pose_world.pose.orientation.w = curr_q[3]
                    start_pose_planning = self.transform_pose(start_pose_world, self.planning_frame)
                    
                    execution_success, failed_step = self.execute_cartesian_move_stepwise(
                        waypoints, actions=actions, start_pose_msg=start_pose_planning
                    )
                    
                    self.query_active = False
                    if execution_success:
                        self.get_logger().info("VLA trajectory execution succeeded!")
                        target_pose = waypoints[-1]
                        self.pose_pub.publish(target_pose)
                        return True, target_pose, ""
                    else:
                        self.get_logger().error("VLA trajectory execution failed!")
                        return False, None, f"Trajectory execution failed at step {failed_step}."
                else:
                    err = f"HTTP Error {response.status_code}: {response.text}"
                    self.get_logger().error(err)
                    self.query_active = False
                    return False, None, err
                    
            except Exception as e:
                err = f"Failed to query VLA model API: {str(e)}"
                self.get_logger().error(err)
                self.query_active = False
                return False, None, err

    def print_actions_sequence(self, actions, source="VLA Model API"):
        self.get_logger().info(f"--- Action Sequence from {source} (timesteps={len(actions)}) ---")
        joint_names = [
            "EE delta x", 
            "EE delta y", 
            "EE delta z", 
            "EE delta rot x", 
            "EE delta rot y", 
            "EE delta rot z", 
            "Gripper"
        ]
        for idx, act in enumerate(actions):
            self.get_logger().info(f"  Step {idx}:")
            for j_idx, val in enumerate(act):
                if j_idx < len(joint_names):
                    name = joint_names[j_idx]
                else:
                    name = f"joint_{j_idx}"
                
                if j_idx < 3:
                    unit = "(pos delta)"
                elif j_idx < 6:
                    unit = "rad"
                else:
                    unit = "[-1=open, 1=close]"
                self.get_logger().info(f"    {name:<14}: {val:>8.3f} {unit}")
        self.get_logger().info(f"----------------------------------------------------------")

    def goal_callback(self, goal_request):
        self.get_logger().info('Received action goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received action cancel request')
        return CancelResponse.ACCEPT

    def execute_action_callback(self, goal_handle):
        self.get_logger().info('Executing VLA query action...')
        task = goal_handle.request.task
        
        # Publish feedback
        feedback_msg = QueryVLA.Feedback()
        feedback_msg.status = "Querying VLA model..."
        goal_handle.publish_feedback(feedback_msg)
        
        # Run query
        success, target_pose, error_msg = self.query_vla_model(instruction_override=task)
        
        result = QueryVLA.Result()
        result.success = success
        result.error_message = error_msg
        if success and target_pose is not None:
            result.target_pose = target_pose
            goal_handle.succeed()
            self.get_logger().info('Goal succeeded!')
        else:
            goal_handle.abort()
            self.get_logger().warn(f'Goal failed: {error_msg}')
            
        return result

def main(args=None):
    from rclpy.executors import MultiThreadedExecutor
    rclpy.init(args=args)
    node = VLABridgeNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
