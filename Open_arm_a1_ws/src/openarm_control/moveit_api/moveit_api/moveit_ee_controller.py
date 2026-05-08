#!/usr/bin/env python3
"""
MoveIt End-Effector Controller Node

A ROS 2 node that accepts end-effector (EE) pose commands via ROS 2 services
and uses MoveIt2's MoveGroupInterface to plan and execute motions.

This node acts as the bridge between the HTTP API server and MoveIt2.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    RobotState,
    WorkspaceParameters,
)
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from std_srvs.srv import Trigger
import json
import threading
import time
import math

# MoveIt error codes → human-readable messages
MOVEIT_ERROR_CODES = {
    1: 'SUCCESS',
    -1: 'PLANNING_FAILED',
    -2: 'INVALID_MOTION_PLAN',
    -3: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE',
    -4: 'CONTROL_FAILED',
    -5: 'UNABLE_TO_AQUIRE_SENSOR_DATA',
    -6: 'TIMED_OUT',
    -7: 'PREEMPTED',
    -10: 'START_STATE_IN_COLLISION',
    -11: 'START_STATE_VIOLATES_PATH_CONSTRAINTS',
    -12: 'GOAL_IN_COLLISION',
    -13: 'GOAL_VIOLATES_PATH_CONSTRAINTS',
    -14: 'GOAL_CONSTRAINTS_VIOLATED',
    -15: 'INVALID_GROUP_NAME',
    -16: 'INVALID_GOAL_CONSTRAINTS',
    -17: 'INVALID_ROBOT_STATE',
    -18: 'INVALID_LINK_NAME',
    -19: 'INVALID_OBJECT_NAME',
    -21: 'FRAME_TRANSFORM_FAILURE',
    -22: 'COLLISION_CHECKING_UNAVAILABLE',
    -23: 'ROBOT_STATE_STALE',
    -24: 'SENSOR_INFO_STALE',
    -25: 'COMMUNICATION_FAILURE',
    -31: 'NO_IK_SOLUTION',
    99999: 'FAILURE',
}


class MoveItEEController(Node):
    """
    ROS 2 Node that provides end-effector pose control via MoveIt2.
    
    Planning Groups (from SRDF):
      - left_arm:  chain from openarm_body_link0 → openarm_left_hand_tcp
      - right_arm: chain from openarm_body_link0 → openarm_right_hand_tcp
      - both_arms: combined group
    
    Grippers:
      - left_gripper:  openarm_left_finger_joint1  (0.0 - 0.044)
      - right_gripper: openarm_right_finger_joint1 (0.0 - 0.044)
    """

    # Joint names per arm
    LEFT_ARM_JOINTS = [f'openarm_left_joint{i}' for i in range(1, 8)]
    RIGHT_ARM_JOINTS = [f'openarm_right_joint{i}' for i in range(1, 8)]

    def __init__(self):
        super().__init__('moveit_ee_controller')
        
        self._cb_group = ReentrantCallbackGroup()
        
        # ── MoveGroup Action Clients ──
        self._move_group_client = ActionClient(
            self, MoveGroup, 'move_action',
            callback_group=self._cb_group,
        )
        
        # ── Joint trajectory action clients (for gripper & direct joint) ──
        self._left_arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/left_arm_controller/follow_joint_trajectory',
            callback_group=self._cb_group,
        )
        self._right_arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/right_arm_controller/follow_joint_trajectory',
            callback_group=self._cb_group,
        )
        self._left_gripper_client = ActionClient(
            self, FollowJointTrajectory,
            '/left_gripper_controller/follow_joint_trajectory',
            callback_group=self._cb_group,
        )
        self._right_gripper_client = ActionClient(
            self, FollowJointTrajectory,
            '/right_gripper_controller/follow_joint_trajectory',
            callback_group=self._cb_group,
        )
        
        # ── FK/IK service clients ──
        self._fk_client = self.create_client(
            GetPositionFK, '/compute_fk',
            callback_group=self._cb_group,
        )
        self._ik_client = self.create_client(
            GetPositionIK, '/compute_ik',
            callback_group=self._cb_group,
        )
        
        # ── Joint state subscriber ──
        self._current_joint_state = None
        self._joint_state_lock = threading.Lock()
        self.create_subscription(
            JointState, '/joint_states',
            self._joint_state_cb, 10,
            callback_group=self._cb_group,
        )
        
        # ── State tracking ──
        self._is_moving = {'left_arm': False, 'right_arm': False}
        self._last_result = {'left_arm': None, 'right_arm': None}
        self._state_lock = threading.Lock()
        
        self.get_logger().info('MoveIt EE Controller node initialized')
        self.get_logger().info('Waiting for MoveGroup action server...')
        
    def _joint_state_cb(self, msg: JointState):
        with self._joint_state_lock:
            self._current_joint_state = msg
    
    def get_current_joint_state(self) -> JointState:
        with self._joint_state_lock:
            return self._current_joint_state

    # ──────────────────────────────────────────────
    # End-Effector Pose Control via MoveGroup Action
    # ──────────────────────────────────────────────
    
    def move_to_pose(self, group_name: str, target_pose: dict,
                     velocity_scaling: float = 0.3,
                     acceleration_scaling: float = 0.3,
                     planning_time: float = 10.0,
                     num_planning_attempts: int = 10,
                     position_tolerance: float = 0.01,
                     orientation_tolerance: float = 0.5,
                     position_only: bool = False) -> dict:
        """
        Plan and execute a motion to a target end-effector pose.
        
        Args:
            group_name: 'left_arm' or 'right_arm'
            target_pose: dict with keys:
                - position: {x, y, z}
                - orientation: {x, y, z, w}  (quaternion)
            velocity_scaling: 0.0 - 1.0
            acceleration_scaling: 0.0 - 1.0
            planning_time: seconds allowed for planning
            num_planning_attempts: number of planning attempts
            position_tolerance: meters (default 0.01 = 1cm)
            orientation_tolerance: radians (default 0.5 ≈ 29°)
            position_only: if True, ignore orientation constraint
            
        Returns:
            dict with 'success', 'message', 'planning_time'
        """
        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            return {'success': False, 'message': 'MoveGroup action server not available'}
        
        # Determine the end-effector link
        if group_name == 'left_arm':
            ee_link = 'openarm_left_hand_tcp'
        elif group_name == 'right_arm':
            ee_link = 'openarm_right_hand_tcp'
        else:
            return {'success': False, 'message': f'Unknown group: {group_name}'}
        
        with self._state_lock:
            self._is_moving[group_name] = True
        
        try:
            # Build the MoveGroup goal
            goal = MoveGroup.Goal()
            
            # ── Planning options ──
            goal.planning_options.plan_only = False
            goal.planning_options.look_around = False
            goal.planning_options.replan = True
            goal.planning_options.replan_attempts = 3
            
            # ── Motion plan request ──
            req = goal.request
            req.group_name = group_name
            req.num_planning_attempts = num_planning_attempts
            req.allowed_planning_time = planning_time
            req.max_velocity_scaling_factor = velocity_scaling
            req.max_acceleration_scaling_factor = acceleration_scaling
            
            # Workspace parameters
            req.workspace_parameters.header.frame_id = 'world'
            req.workspace_parameters.min_corner.x = -2.0
            req.workspace_parameters.min_corner.y = -2.0
            req.workspace_parameters.min_corner.z = -2.0
            req.workspace_parameters.max_corner.x = 2.0
            req.workspace_parameters.max_corner.y = 2.0
            req.workspace_parameters.max_corner.z = 2.0
            
            # ── Goal constraints (target EE pose) ──
            constraints = Constraints()
            
            # Position constraint
            pos_constraint = PositionConstraint()
            pos_constraint.header.frame_id = 'world'
            pos_constraint.link_name = ee_link
            pos_constraint.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
            
            # Bounding volume (sphere around target)
            bounding_volume = BoundingVolume()
            solid = SolidPrimitive()
            solid.type = SolidPrimitive.SPHERE
            solid.dimensions = [position_tolerance]
            bounding_volume.primitives.append(solid)
            
            target_pose_msg = Pose()
            target_pose_msg.position.x = float(target_pose['position']['x'])
            target_pose_msg.position.y = float(target_pose['position']['y'])
            target_pose_msg.position.z = float(target_pose['position']['z'])
            bounding_volume.primitive_poses.append(target_pose_msg)
            
            pos_constraint.constraint_region = bounding_volume
            pos_constraint.weight = 1.0
            constraints.position_constraints.append(pos_constraint)
            
            # Orientation constraint (skip if position_only mode)
            if not position_only:
                orient_constraint = OrientationConstraint()
                orient_constraint.header.frame_id = 'world'
                orient_constraint.link_name = ee_link
                orient_constraint.orientation.x = float(target_pose['orientation']['x'])
                orient_constraint.orientation.y = float(target_pose['orientation']['y'])
                orient_constraint.orientation.z = float(target_pose['orientation']['z'])
                orient_constraint.orientation.w = float(target_pose['orientation']['w'])
                orient_constraint.absolute_x_axis_tolerance = orientation_tolerance
                orient_constraint.absolute_y_axis_tolerance = orientation_tolerance
                orient_constraint.absolute_z_axis_tolerance = orientation_tolerance
                orient_constraint.weight = 1.0
                constraints.orientation_constraints.append(orient_constraint)
            
            req.goal_constraints.append(constraints)
            
            # ── Send goal ──
            self.get_logger().info(
                f'Planning motion for {group_name} to '
                f'pos=({target_pose["position"]["x"]:.3f}, '
                f'{target_pose["position"]["y"]:.3f}, '
                f'{target_pose["position"]["z"]:.3f})'
            )
            
            send_goal_future = self._move_group_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=10.0)
            
            goal_handle = send_goal_future.result()
            if not goal_handle or not goal_handle.accepted:
                return {'success': False, 'message': 'Goal rejected by MoveGroup'}
            
            # Wait for result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
            
            result = result_future.result()
            if result is None:
                return {'success': False, 'message': 'Execution timed out'}
            
            move_result = result.result
            error_code = move_result.error_code.val
            
            if error_code == 1:  # SUCCESS
                return {
                    'success': True,
                    'message': 'Motion executed successfully',
                    'planning_time': move_result.planning_time,
                }
            else:
                error_name = MOVEIT_ERROR_CODES.get(error_code, f'UNKNOWN({error_code})')
                hint = ''
                if error_code in (99999, -1):
                    hint = (' Hint: The target pose may be unreachable. '
                            'Try a position closer to the current EE pose, '
                            'use position_only=true, or increase tolerances.')
                elif error_code == -31:
                    hint = ' Hint: No IK solution found for the target pose.'
                return {
                    'success': False,
                    'message': f'{error_name}: Planning failed for {group_name}.{hint}',
                    'error_code': error_code,
                    'planning_time': move_result.planning_time,
                }
                
        except Exception as e:
            self.get_logger().error(f'Exception during move_to_pose: {e}')
            return {'success': False, 'message': str(e)}
        finally:
            with self._state_lock:
                self._is_moving[group_name] = False

    # ──────────────────────────────────────────────
    # Named Pose (SRDF group_state) via MoveGroup
    # ──────────────────────────────────────────────
    
    def move_to_named_pose(self, group_name: str, pose_name: str,
                           velocity_scaling: float = 0.3,
                           acceleration_scaling: float = 0.3) -> dict:
        """
        Move to a named pose defined in SRDF (e.g. 'home', 'ready').
        
        Available poses: home, ready (for left_arm/right_arm), open/close (for grippers)
        """
        # Named poses from SRDF
        named_poses = {
            'left_arm': {
                'home':  [0.0, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0],
                'ready': [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0],
            },
            'right_arm': {
                'home':  [0.0, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0],
                'ready': [0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0],
            },
        }
        
        if group_name not in named_poses:
            return {'success': False, 'message': f'Unknown group: {group_name}'}
        if pose_name not in named_poses[group_name]:
            return {'success': False, 'message': f'Unknown pose: {pose_name} for {group_name}'}
        
        positions = named_poses[group_name][pose_name]
        return self.move_to_joint_positions(group_name, positions,
                                            velocity_scaling=velocity_scaling,
                                            acceleration_scaling=acceleration_scaling)

    # ──────────────────────────────────────────────
    # Direct Joint Position Control (via trajectory)
    # ──────────────────────────────────────────────

    def move_to_joint_positions(self, group_name: str, positions: list,
                                duration: float = 3.0,
                                velocity_scaling: float = 0.3,
                                acceleration_scaling: float = 0.3) -> dict:
        """
        Move arm joints directly to target positions.
        
        Args:
            group_name: 'left_arm' or 'right_arm'
            positions: list of 7 joint angles in radians
            duration: time to reach goal (seconds)
        """
        if group_name == 'left_arm':
            client = self._left_arm_client
            joint_names = self.LEFT_ARM_JOINTS
        elif group_name == 'right_arm':
            client = self._right_arm_client
            joint_names = self.RIGHT_ARM_JOINTS
        else:
            return {'success': False, 'message': f'Unknown group: {group_name}'}
        
        if len(positions) != 7:
            return {'success': False, 'message': f'Expected 7 joint positions, got {len(positions)}'}
        
        if not client.wait_for_server(timeout_sec=5.0):
            return {'success': False, 'message': f'{group_name} trajectory controller not available'}
        
        # Scale duration by velocity
        scaled_duration = duration / max(velocity_scaling, 0.05)
        
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names
        
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start.sec = int(scaled_duration)
        point.time_from_start.nanosec = int((scaled_duration - int(scaled_duration)) * 1e9)
        goal_msg.trajectory.points = [point]
        
        self.get_logger().info(f'Sending joint goal to {group_name}: {positions}')
        
        send_future = client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        
        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            return {'success': False, 'message': 'Joint trajectory goal rejected'}
        
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=scaled_duration + 10.0)
        
        result = result_future.result()
        if result is None:
            return {'success': False, 'message': 'Joint trajectory execution timed out'}
        
        error_code = result.result.error_code
        if error_code == 0:  # SUCCESSFUL
            return {'success': True, 'message': f'{group_name} reached target joint positions'}
        else:
            return {'success': False, 'message': f'Trajectory error code: {error_code}'}
    
    # ──────────────────────────────────────────────
    # Gripper Control
    # ──────────────────────────────────────────────
    
    def move_gripper(self, side: str, position: float, duration: float = 1.0) -> dict:
        """
        Control gripper opening.
        
        Args:
            side: 'left' or 'right'
            position: 0.0 (closed) to 0.044 (open)
            duration: execution time in seconds
        """
        position = max(0.0, min(0.044, float(position)))
        
        if side == 'left':
            client = self._left_gripper_client
            joint_name = 'openarm_left_finger_joint1'
        elif side == 'right':
            client = self._right_gripper_client
            joint_name = 'openarm_right_finger_joint1'
        else:
            return {'success': False, 'message': f'Unknown side: {side}'}
        
        if not client.wait_for_server(timeout_sec=5.0):
            return {'success': False, 'message': f'{side} gripper controller not available'}
        
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [joint_name]
        
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        goal_msg.trajectory.points = [point]
        
        self.get_logger().info(f'Moving {side} gripper to {position:.3f}')
        
        send_future = client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        
        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            return {'success': False, 'message': 'Gripper goal rejected'}
        
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration + 5.0)
        
        return {'success': True, 'message': f'{side} gripper moved to {position:.3f}'}

    # ──────────────────────────────────────────────
    # Get Current EE Pose (via FK)
    # ──────────────────────────────────────────────
    
    def get_ee_pose(self, group_name: str) -> dict:
        """
        Get current end-effector pose using Forward Kinematics.
        
        Returns:
            dict with position {x,y,z} and orientation {x,y,z,w}
        """
        js = self.get_current_joint_state()
        if js is None:
            return {'success': False, 'message': 'No joint states received yet'}
        
        if group_name == 'left_arm':
            ee_link = 'openarm_left_hand_tcp'
        elif group_name == 'right_arm':
            ee_link = 'openarm_right_hand_tcp'
        else:
            return {'success': False, 'message': f'Unknown group: {group_name}'}
        
        if not self._fk_client.wait_for_service(timeout_sec=5.0):
            # Fallback: return joint positions instead
            return self._get_joint_positions_fallback(group_name, js)
        
        fk_req = GetPositionFK.Request()
        fk_req.header.frame_id = 'world'
        fk_req.fk_link_names = [ee_link]
        fk_req.robot_state.joint_state = js
        
        future = self._fk_client.call_async(fk_req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        result = future.result()
        if result is None or result.error_code.val != 1:
            return self._get_joint_positions_fallback(group_name, js)
        
        pose = result.pose_stamped[0].pose
        return {
            'success': True,
            'group': group_name,
            'ee_link': ee_link,
            'position': {
                'x': round(pose.position.x, 6),
                'y': round(pose.position.y, 6),
                'z': round(pose.position.z, 6),
            },
            'orientation': {
                'x': round(pose.orientation.x, 6),
                'y': round(pose.orientation.y, 6),
                'z': round(pose.orientation.z, 6),
                'w': round(pose.orientation.w, 6),
            },
        }
    
    def _get_joint_positions_fallback(self, group_name: str, js: JointState) -> dict:
        """Return joint positions when FK service is unavailable."""
        if group_name == 'left_arm':
            target_joints = self.LEFT_ARM_JOINTS
        else:
            target_joints = self.RIGHT_ARM_JOINTS
        
        positions = {}
        for jname in target_joints:
            if jname in js.name:
                idx = js.name.index(jname)
                positions[jname] = round(js.position[idx], 6)
        
        return {
            'success': True,
            'group': group_name,
            'joint_positions': positions,
            'message': 'FK service unavailable, returning joint positions',
        }

    # ──────────────────────────────────────────────
    # Status
    # ──────────────────────────────────────────────
    
    def get_status(self) -> dict:
        js = self.get_current_joint_state()
        with self._state_lock:
            moving = dict(self._is_moving)
        
        result = {
            'success': True,
            'is_moving': moving,
            'joint_states_available': js is not None,
        }
        
        if js is not None:
            # Extract relevant joint positions
            joint_data = {}
            for name, pos in zip(js.name, js.position):
                joint_data[name] = round(pos, 6)
            result['joints'] = joint_data
        
        return result

