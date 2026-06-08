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
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from openarm_messages.action import QueryVLA

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
        self.declare_parameter('mode', 'mock')  # 'real' or 'mock'
        self.declare_parameter('arm', 'left')   # 'left' or 'right'
        self.declare_parameter('instruction', 'Push the apple to the block')
        self.declare_parameter('pos_scale', 0.1) # scale normalized translation delta to meters
        self.declare_parameter('rot_scale', 1.0) # scale rotation axis-angle
        self.declare_parameter('predict_mode', 'batch') # 'batch' or 'stream'
        self.declare_parameter('front_camera_topic', '/camera_front/image_raw')
        self.declare_parameter('left_camera_topic', '/camera_left/image_raw')
        self.declare_parameter('joint_states_topic', '/joint_states')
        
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
            joint_states_qos
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
            camera_qos
        )
        
        self.left_image_sub = self.create_subscription(
            Image,
            self.left_camera_topic,
            self.left_image_callback,
            camera_qos
        )
        self.get_logger().info(f"Subscribed to front camera: '{self.front_camera_topic}' (BEST_EFFORT QoS)")
        self.get_logger().info(f"Subscribed to left camera:  '{self.left_camera_topic}' (BEST_EFFORT QoS)")
        
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/bt_executor/vla_grasp_pose',
            10
        )
        
        # Action Server
        self._action_server = ActionServer(
            self,
            QueryVLA,
            '/vla_bridge/query',
            execute_callback=self.execute_action_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        self.get_logger().info("QueryVLA Action Server started on topic '/vla_bridge/query'")
        
        # State variables
        self.current_joints = None
        self.joint_names = []
        self.query_active = False

        # Timer to trigger query for testing (or you can trigger it via a service)
        # Retries every 5 seconds until camera images are available, then fires once.
        self.trigger_timer = self.create_timer(5.0, self.test_trigger_callback)
        self.trigger_timer_fired = False

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
        
        # If gripper finger joint exists, read it too
        gripper_name = f"openarm_{self.arm_side}_finger_joint1"
        gripper_pos = 0.0
        if gripper_name in self.joint_names:
            idx = self.joint_names.index(gripper_name)
            gripper_pos = self.current_joints[idx]
            
        return arm_positions, gripper_pos

    def forward_kinematics_so_arm100(self, joints_deg):
        """
        Simple forward kinematics solver for SO-ARM100 dimensions to map 
        predicted joint angles to Cartesian coordinates in meters.
        """
        theta = np.radians(joints_deg)
        
        # SO-ARM100 Link lengths (approximate in meters)
        L0 = 0.12  # Base height
        L1 = 0.15  # Upper arm
        L2 = 0.15  # Lower arm
        L3 = 0.08  # Wrist length
        L4 = 0.05  # Hand offset
        
        # Transformation matrices
        def rz(a):
            return np.array([
                [np.cos(a), -np.sin(a), 0, 0],
                [np.sin(a), np.cos(a), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
        def ry(a):
            return np.array([
                [np.cos(a), 0, np.sin(a), 0],
                [0, 1, 0, 0],
                [-np.sin(a), 0, np.cos(a), 0],
                [0, 0, 0, 1]
            ])
            
        def tz(d):
            T = np.eye(4)
            T[2, 3] = d
            return T
            
        # Kinematic Chain
        T = tz(L0)
        T = T @ rz(theta[0])          # Shoulder Pan
        T = T @ ry(theta[1])          # Shoulder Lift
        T = T @ tz(L1)
        T = T @ ry(theta[2])          # Elbow Flex
        T = T @ tz(L2)
        T = T @ ry(theta[3])          # Wrist Flex
        T = T @ tz(L3)
        T = T @ rz(theta[4])          # Wrist Roll
        T = T @ tz(L4)                # Tool tip
        
        # Position
        pos = T[:3, 3]
        
        # Rotation Matrix to Quaternion
        R = T[:3, :3]
        q = self.matrix_to_quaternion(R)
        
        return pos, q

    def matrix_to_quaternion(self, R):
        tr = np.trace(R)
        q = Quaternion()
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            q.w = 0.25 * S
            q.x = (R[2, 1] - R[1, 2]) / S
            q.y = (R[0, 2] - R[2, 0]) / S
            q.z = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q.w = (R[2, 1] - R[1, 2]) / S
            q.x = 0.25 * S
            q.y = (R[0, 1] + R[1, 0]) / S
            q.z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q.w = (R[0, 2] - R[2, 0]) / S
            q.x = (R[0, 1] + R[1, 0]) / S
            q.y = 0.25 * S
            q.z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q.w = (R[1, 0] - R[0, 1]) / S
            q.x = (R[0, 2] + R[2, 0]) / S
            q.y = (R[1, 2] + R[2, 1]) / S
            q.z = 0.25 * S
        return q

    def encode_dummy_image(self):
        # Creates a mock 224x224 RGB image (solid grey)
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        buf = io.BytesIO()
        np.save(buf, img)
        return base64.b64encode(buf.getvalue()).decode()

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
            self.get_logger().warn("Front camera frame not available. Falling back to dummy image.")
            img1_b64 = self.encode_dummy_image()
            
        if img2_b64 is None:
            self.get_logger().warn("Left camera frame not available. Falling back to dummy image.")
            img2_b64 = self.encode_dummy_image()
            
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
        delta_pos = np.array(action[:3]) * self.pos_scale
        delta_rot_vec = np.array(action[3:6]) * self.rot_scale
        
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
        return msg

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

    def query_vla_model(self, instruction_override=None):
        instr = instruction_override if instruction_override is not None else self.instruction
        
        if self.query_active:
            return False, None, "Query is already active."
        self.query_active = True
        
        # Read current joints just for logging/compatibility state
        joint_data = self.get_arm_joints()
        if joint_data is None:
            self.get_logger().warn("Joint states not available yet.")
            self.query_active = False
            return False, None, "Joint states not available yet."
            
        arm_positions, gripper_pos = joint_data
        
        # Compatibility state vector (6-DoF, degrees)
        state_deg = [
            np.degrees(arm_positions[0]), # shoulder_pan
            np.degrees(arm_positions[1]), # shoulder_lift
            np.degrees(arm_positions[2]), # elbow_flex
            np.degrees(arm_positions[3]), # wrist_flex
            np.degrees(arm_positions[4]), # wrist_roll
            gripper_pos * 100.0           # gripper opening scale
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
            self.get_logger().warning(f"TF lookup failed: {ex}. Using default home reference.")
            curr_pos = np.array([-0.3, 0.15, 0.8])
            curr_q = [0.0, 0.0, 0.0, 1.0]

        img1_b64, img2_b64 = self.get_latest_images_b64()

        if self.predict_mode == 'stream':
            if self.mode == 'real':
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
                        last_action = None
                        for line in response.iter_lines():
                            if line:
                                data = json.loads(line.decode('utf-8'))
                                if "value" in data:
                                    action = np.array(data["value"][0])
                                    step_idx = data["index"]
                                    self.print_action_step(step_idx, action, source="Real VLA Model Stream")
                                    last_action = action
                                elif "time_taken" in data:
                                    self.get_logger().info(f"Stream generation finished. Time taken: {data['time_taken']}s")
                        
                        self.query_active = False
                        if last_action is not None:
                            target_pose = self.compute_target_pose(last_action, curr_pos, curr_q)
                            self.pose_pub.publish(target_pose)
                            return True, target_pose, ""
                        else:
                            return False, None, "No action received in stream."
                    else:
                        err = f"HTTP Error {response.status_code}: {response.text}"
                        self.get_logger().error(err)
                        self.get_logger().info("Falling back to simulated mockup stream...")
                        target_pose = self.trigger_mock_stream(curr_pos, curr_q)
                        self.query_active = False
                        return True, target_pose, ""
                except Exception as e:
                    err = f"Failed to query VLA model API stream: {str(e)}"
                    self.get_logger().error(err)
                    self.get_logger().info("Falling back to simulated mockup stream...")
                    target_pose = self.trigger_mock_stream(curr_pos, curr_q)
                    self.query_active = False
                    return True, target_pose, ""
            else:
                target_pose = self.trigger_mock_stream(curr_pos, curr_q)
                self.query_active = False
                return True, target_pose, ""
        else:
            # Batch mode
            if self.mode == 'real':
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
                        
                        # Take the final predicted pose of the sequence (8th step)
                        target_joints = actions[-1]
                        target_pose = self.compute_target_pose(target_joints, curr_pos, curr_q)
                        self.pose_pub.publish(target_pose)
                        self.query_active = False
                        return True, target_pose, ""
                    else:
                        err = f"HTTP Error {response.status_code}: {response.text}"
                        self.get_logger().error(err)
                        self.get_logger().info("Falling back to simulated mockup pose...")
                        target_pose = self.trigger_mock_pose(curr_pos, curr_q)
                        self.query_active = False
                        return True, target_pose, ""
                        
                except Exception as e:
                    err = f"Failed to query VLA model API: {str(e)}"
                    self.get_logger().error(err)
                    self.get_logger().info("Falling back to simulated mockup pose...")
                    target_pose = self.trigger_mock_pose(curr_pos, curr_q)
                    self.query_active = False
                    return True, target_pose, ""
            else:
                target_pose = self.trigger_mock_pose(curr_pos, curr_q)
                self.query_active = False
                return True, target_pose, ""

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

    def trigger_mock_pose(self, curr_pos, curr_q):
        # 7-DoF LIBERO EE delta action mockup: [dx, dy, dz, ax, ay, az, gripper]
        mock_actions = [
            [ 0.12, -0.10,  0.06,  0.00,  0.09, -0.08, -1.00],
            [ 0.13, -0.11,  0.07,  0.00,  0.09, -0.08, -1.00],
            [ 0.14, -0.11,  0.07,  0.00,  0.09, -0.08, -1.00],
            [ 0.15, -0.12,  0.08,  0.00,  0.10, -0.08, -1.00],
            [ 0.16, -0.12,  0.08,  0.00,  0.10, -0.08, -1.00],
            [ 0.17, -0.13,  0.09,  0.00,  0.10, -0.08, -1.00],
            [ 0.18, -0.13,  0.09,  0.00,  0.10, -0.08, -1.00],
            [ 0.19, -0.14,  0.10,  0.00,  0.11, -0.08,  1.00]
        ]
        self.print_actions_sequence(mock_actions, source="Mock Simulator (LIBERO EE delta)")
        
        # Take the final predicted pose of the sequence (8th step)
        target_joints = mock_actions[-1]
        target_pose = self.compute_target_pose(target_joints, curr_pos, curr_q)
        self.pose_pub.publish(target_pose)
        self.get_logger().info(f"Published mockup target grasp pose: Position ({target_pose.pose.position.x:.3f}, {target_pose.pose.position.y:.3f}, {target_pose.pose.position.z:.3f})")
        return target_pose

    def trigger_mock_stream(self, curr_pos, curr_q):
        mock_actions = [
            [ 0.12, -0.10,  0.06,  0.00,  0.09, -0.08, -1.00],
            [ 0.13, -0.11,  0.07,  0.00,  0.09, -0.08, -1.00],
            [ 0.14, -0.11,  0.07,  0.00,  0.09, -0.08, -1.00],
            [ 0.15, -0.12,  0.08,  0.00,  0.10, -0.08, -1.00],
            [ 0.16, -0.12,  0.08,  0.00,  0.10, -0.08, -1.00],
            [ 0.17, -0.13,  0.09,  0.00,  0.10, -0.08, -1.00],
            [ 0.18, -0.13,  0.09,  0.00,  0.10, -0.08, -1.00],
            [ 0.19, -0.14,  0.10,  0.00,  0.11, -0.08,  1.00]
        ]
        self.get_logger().info("Starting simulated mockup stream...")
        target_pose = None
        for idx, action in enumerate(mock_actions):
            self.print_action_step(idx, action, source="Mock Simulator Stream")
            target_pose = self.compute_target_pose(action, curr_pos, curr_q)
            self.pose_pub.publish(target_pose)
            self.get_logger().info(f"Published mockup target grasp pose stream: Position ({target_pose.pose.position.x:.3f}, {target_pose.pose.position.y:.3f}, {target_pose.pose.position.z:.3f})")
            time.sleep(0.3)
        return target_pose

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

    def test_trigger_callback(self):
        if not self.trigger_timer_fired:
            # Wait until both cameras have published at least one frame
            cameras_ready = (self.front_image_msg is not None and self.left_image_msg is not None)
            if not cameras_ready:
                front_ok = "✓" if self.front_image_msg is not None else "✗"
                left_ok  = "✓" if self.left_image_msg  is not None else "✗"
                self.get_logger().info(
                    f"Waiting for cameras... front={front_ok}, left={left_ok}. Retrying in 5s."
                )
                return  # Timer is recurring, will retry in 5 seconds
            self.trigger_timer_fired = True
            self.get_logger().info("Both cameras ready! Starting demo query...")
            self.query_vla_model()

def main(args=None):
    rclpy.init(args=args)
    node = VLABridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
