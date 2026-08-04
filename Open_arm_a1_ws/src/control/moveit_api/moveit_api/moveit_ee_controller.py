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
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

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
from std_msgs.msg import Header, String
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
    BOTH_ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

    # amazing_hand finger aliases (see openarm_bimanual.srdf left_hand_fingers/
    # right_hand_fingers groups): j11-j14 = yaw/abduction, j21-j24 = flex, per
    # finger 1-4 (thumb is finger 4 - see hand_kinematics_node.py's finger
    # reordering). Order here must match the SRDF group's joint order.
    LEFT_HAND_JOINTS = [f'openarm_left_j{n}' for n in (11, 12, 13, 14, 21, 22, 23, 24)]
    RIGHT_HAND_JOINTS = [f'openarm_right_j{n}' for n in (11, 12, 13, 14, 21, 22, 23, 24)]

    # body_type:=v2 articulated neck/head (see openarm_body.xacro). Group is
    # injected into the SRDF at launch time (robot_api.launch.py), not static.
    HEAD_JOINTS = ['openarm_body_neck_joint', 'openarm_body_head_joint']

    # Every SRDF planning group this controller can send joint-space goals to,
    # via the collision-checked MoveGroup path (never bypass MoveIt directly -
    # that's how a single-arm move used to skip collision checking entirely).
    GROUP_JOINTS = {
        'left_arm': LEFT_ARM_JOINTS,
        'right_arm': RIGHT_ARM_JOINTS,
        'both_arms': BOTH_ARM_JOINTS,
        'left_hand_fingers': LEFT_HAND_JOINTS,
        'right_hand_fingers': RIGHT_HAND_JOINTS,
        'head': HEAD_JOINTS,
    }

    def __init__(self):
        super().__init__('moveit_ee_controller')
        
        self._cb_group = ReentrantCallbackGroup()
        
        # ── MoveGroup Action Clients ──
        self._move_group_client = ActionClient(
            self, MoveGroup, 'move_action',
            callback_group=self._cb_group,
        )
        
        # ── Joint trajectory action clients (gripper only - arm/hand joint
        # moves always go through MoveGroup now, see _move_to_joints_moveit,
        # so they get collision checking instead of bypassing it) ──
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

        # ── Robot description (URDF) subscriber ──
        # robot_state_publisher latches /robot_description with TRANSIENT_LOCAL
        # durability so late subscribers still get it - cache it here for the
        # web visualizer's GET /api/urdf instead of re-reading/re-xacro'ing files.
        self._urdf_string = None
        self._urdf_lock = threading.Lock()
        urdf_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            String, '/robot_description',
            self._urdf_cb, urdf_qos,
            callback_group=self._cb_group,
        )

        # ── State tracking ──
        self._is_moving = {'left_arm': False, 'right_arm': False}
        self._last_result = {'left_arm': None, 'right_arm': None}
        self._state_lock = threading.Lock()
        
        self.get_logger().info('MoveIt EE Controller node initialized')
        self.get_logger().info('Waiting for MoveGroup action server...')

    def _wait_for_future(self, future, timeout_sec=None):
        """
        Thread-safe wait for a future when an executor is already spinning 
        the node in another thread.
        """
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout=timeout_sec):
            return None
        return future.result()
        
    def _joint_state_cb(self, msg: JointState):
        with self._joint_state_lock:
            self._current_joint_state = msg

    def get_current_joint_state(self) -> JointState:
        with self._joint_state_lock:
            return self._current_joint_state

    def _urdf_cb(self, msg: String):
        with self._urdf_lock:
            self._urdf_string = msg.data

    def get_urdf(self) -> str:
        with self._urdf_lock:
            return self._urdf_string

    # ──────────────────────────────────────────────
    # End-Effector Pose Control via MoveGroup Action
    # ──────────────────────────────────────────────
    
    def move_to_pose(self, group_name: str, target_pose: dict,
                     velocity_scaling: float = 0.3,
                     acceleration_scaling: float = 0.3,
                     planning_time: float = 5.0,
                     num_planning_attempts: int = 10,
                     position_tolerance: float = 0.01,
                     orientation_tolerance: float = 0.5,
                     position_only: bool = False,
                     pipeline_id: str = 'ompl',
                     planner_id: str = None) -> dict:
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
        if not rclpy.ok():
            return {'success': False, 'message': 'ROS 2 context is invalid or shut down'}

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
            req.pipeline_id = pipeline_id
            if planner_id:
                req.planner_id = planner_id
            
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
            goal_handle = self._wait_for_future(send_goal_future, timeout_sec=10.0)
            
            if not goal_handle or not goal_handle.accepted:
                return {'success': False, 'message': 'Goal rejected by MoveGroup'}
            
            # Wait for result
            result_future = goal_handle.get_result_async()
            result = self._wait_for_future(result_future, timeout_sec=60.0)
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
        Move to a named pose defined in SRDF (openarm_bimanual.srdf group_states).

        Available poses: home/ready (left_arm/right_arm/both_arms),
        open/close/home (left_hand_fingers/right_hand_fingers - amazing_hand).
        """
        # Named poses, kept in sync with openarm_bimanual.srdf's group_states
        named_poses = {
            'left_arm': {
                'home':  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'ready': [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0],
            },
            'right_arm': {
                'home':  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'ready': [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0],
            },
            'both_arms': {
                'home':  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'ready': [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0],
            },
            # order: j11,j12,j13,j14, j21,j22,j23,j24 (thumb = finger 4 = j14/j24)
            'left_hand_fingers': {
                'open':  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'close': [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                'home':  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.308997],
            },
            'right_hand_fingers': {
                'open':  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'close': [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                'home':  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.308997],
            },
        }

        if group_name not in named_poses:
            return {'success': False, 'message': f'Unknown group: {group_name}. '
                    f'Valid groups: {list(named_poses)}'}
        if pose_name not in named_poses[group_name]:
            return {'success': False, 'message': f'Unknown pose: {pose_name} for {group_name}. '
                    f'Valid poses: {list(named_poses[group_name])}'}
        
        positions = named_poses[group_name][pose_name]
        return self.move_to_joint_positions(group_name, positions,
                                            velocity_scaling=velocity_scaling,
                                            acceleration_scaling=acceleration_scaling)

    # ──────────────────────────────────────────────
    # Direct Joint Position Control (via trajectory)
    # ──────────────────────────────────────────────

    def _resolve_joint_positions(self, group_name: str, positions: list, unit: str):
        """Shared validation for move/plan-to-joint-positions. Returns
        (joint_names, positions_in_radians, None) on success, or
        (None, None, error_dict) on failure."""
        joint_names = self.GROUP_JOINTS.get(group_name)
        if joint_names is None:
            return None, None, {'success': False, 'message': f'Unknown group: {group_name}. '
                                 f'Valid groups: {list(self.GROUP_JOINTS)}'}

        if len(positions) != len(joint_names):
            return None, None, {'success': False, 'message': f'Expected {len(joint_names)} joint '
                                 f'positions for {group_name}, got {len(positions)}'}

        if unit == 'deg':
            positions = [math.radians(p) for p in positions]
        elif unit != 'rad':
            return None, None, {'success': False, 'message': f'Unknown unit: {unit}. Use "rad" or "deg"'}

        return joint_names, positions, None

    def move_to_joint_positions(self, group_name: str, positions: list,
                                duration: float = 3.0,
                                velocity_scaling: float = 0.3,
                                acceleration_scaling: float = 0.3,
                                unit: str = 'rad') -> dict:
        """Move joints to target positions via MoveGroup (collision-checked).

        `duration` is kept as a parameter for API backward-compatibility but
        is unused now - MoveIt's own trajectory time parameterization handles
        timing, scaled by velocity_scaling/acceleration_scaling.

        `unit`: 'rad' (default) or 'deg' - unit of every value in `positions`.
        """
        joint_names, positions, err = self._resolve_joint_positions(group_name, positions, unit)
        if err:
            return err

        return self._move_to_joints_moveit(group_name, joint_names, positions,
                                            velocity_scaling, acceleration_scaling)

    def plan_to_joint_positions(self, group_name: str, positions: list,
                                velocity_scaling: float = 0.3,
                                acceleration_scaling: float = 0.3,
                                unit: str = 'rad') -> dict:
        """Dry-run version of move_to_joint_positions: asks MoveGroup to plan
        the same joint-space goal WITHOUT executing it (planning_options.plan_only),
        so the robot never actually moves. Same success/failure/error_code
        shape as move_to_joint_positions - `success: false` means MoveIt
        couldn't find a valid (collision-free, reachable) plan.

        This is the RViz2 MotionPlanning-panel "Plan" button: call this
        first to validate a target before committing to move_to_joint_positions
        ("Execute"). Since it re-plans rather than replaying a stored
        trajectory, a Plan success is not an absolute guarantee Execute will
        also succeed if the scene changes in between - but for a human
        clicking Plan then Execute a moment later, that's not a real risk.
        """
        joint_names, positions, err = self._resolve_joint_positions(group_name, positions, unit)
        if err:
            return err

        return self._move_to_joints_moveit(group_name, joint_names, positions,
                                            velocity_scaling, acceleration_scaling,
                                            plan_only=True)

    def move_single_joint(self, group_name: str, joint, value: float,
                          unit: str = 'rad',
                          velocity_scaling: float = 0.3,
                          acceleration_scaling: float = 0.3) -> dict:
        """Move exactly one joint in a group to a target value, holding every
        other joint in that group at its current measured position.

        Args:
            group_name: any key in GROUP_JOINTS (left_arm, right_arm, both_arms,
                left_hand_fingers, right_hand_fingers, head)
            joint: joint name (e.g. 'openarm_left_joint4') or 0-based index
                within the group's joint order
            value: target value for that joint
            unit: 'rad' (default) or 'deg'
        """
        joint_names = self.GROUP_JOINTS.get(group_name)
        if joint_names is None:
            return {'success': False, 'message': f'Unknown group: {group_name}. '
                    f'Valid groups: {list(self.GROUP_JOINTS)}'}

        if isinstance(joint, int) or (isinstance(joint, str) and joint.lstrip('-').isdigit()):
            idx = int(joint)
            if idx < 0 or idx >= len(joint_names):
                return {'success': False, 'message': f'joint index out of range for '
                        f'{group_name} (0-{len(joint_names) - 1})'}
        else:
            if joint not in joint_names:
                return {'success': False, 'message': f'Unknown joint {joint!r} for '
                        f'{group_name}. Valid joints: {joint_names}'}
            idx = joint_names.index(joint)

        js = self.get_current_joint_state()
        if js is None:
            return {'success': False, 'message': 'No joint state available yet'}
        current = dict(zip(js.name, js.position))

        positions = [current.get(name, 0.0) for name in joint_names]
        positions[idx] = math.radians(value) if unit == 'deg' else float(value)

        # Every other joint's "target" here is just its current *measured*
        # position, re-sent as an explicit goal constraint (MoveGroup needs a
        # full goal for the whole group, there's no way to say "don't care").
        # With the default tight tolerance, ordinary sensor noise on that
        # reading (it will essentially never match the last commanded value
        # to within 0.001 rad) makes MoveIt think that joint needs correcting
        # too - so it generates a small but real, physically executed move on
        # a motor you never asked to touch. Give every joint EXCEPT the one
        # actually being commanded a much looser tolerance so noise on its
        # reading can't trigger real motion.
        hold_tolerance = 0.05  # ~2.9 deg - absorbs sensor noise, not a real move
        tolerances = [hold_tolerance] * len(joint_names)
        tolerances[idx] = 0.001

        return self._move_to_joints_moveit(group_name, joint_names, positions,
                                            velocity_scaling, acceleration_scaling,
                                            tolerances=tolerances)

    def _move_to_joints_moveit(self, group_name: str, joint_names: list, positions: list,
                               velocity: float, acceleration: float,
                               plan_only: bool = False,
                               tolerances: list = None) -> dict:
        """Plan (and, unless plan_only, execute) a joint-space goal via
        MoveGroup (collision-checked, reports MoveIt error codes the same
        way move_to_pose does)."""
        if not rclpy.ok():
            return {'success': False, 'message': 'ROS 2 context is invalid or shut down'}

        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            return {'success': False, 'message': 'MoveGroup action server not available'}

        goal = MoveGroup.Goal()
        goal.planning_options.plan_only = plan_only
        goal.planning_options.replan = not plan_only
        goal.planning_options.replan_attempts = 3

        req = goal.request
        req.group_name = group_name
        req.max_velocity_scaling_factor = velocity
        req.max_acceleration_scaling_factor = acceleration
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0

        if tolerances is None:
            tolerances = [0.001] * len(joint_names)

        constraints = Constraints()
        from moveit_msgs.msg import JointConstraint
        for name, pos, tol in zip(joint_names, positions, tolerances):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = tol
            jc.tolerance_below = tol
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints.append(constraints)

        self.get_logger().info(f'Planning joint motion for {group_name}...')
        send_future = self._move_group_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, timeout_sec=10.0)

        if not goal_handle or not goal_handle.accepted:
            return {'success': False, 'message': 'Joint goal rejected by MoveGroup'}

        result_future = goal_handle.get_result_async()
        result = self._wait_for_future(result_future, timeout_sec=60.0)
        if result is None:
            return {'success': False, 'message': 'Execution timed out'}

        error_code = result.result.error_code.val
        if error_code == 1:
            return {
                'success': True,
                'message': f'{group_name} plan succeeded' if plan_only else f'{group_name} reached target',
                'planning_time': result.result.planning_time,
            }
        error_name = MOVEIT_ERROR_CODES.get(error_code, f'UNKNOWN({error_code})')
        action = 'Planning' if plan_only else 'Planning/execution'
        return {
            'success': False,
            'message': f'{error_name}: {action} failed for {group_name}.',
            'error_code': error_code,
        }
    
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
        goal_handle = self._wait_for_future(send_future, timeout_sec=5.0)
        
        if not goal_handle or not goal_handle.accepted:
            return {'success': False, 'message': 'Gripper goal rejected'}
        
        result_future = goal_handle.get_result_async()
        self._wait_for_future(result_future, timeout_sec=duration + 5.0)
        
        return {'success': True, 'message': f'{side} gripper moved to {position:.3f}'}

    # ──────────────────────────────────────────────
    # amazing_hand Control (5-finger hand, ee_type:=amazing_hand)
    # ──────────────────────────────────────────────

    def move_hand(self, side: str, positions: list = None, action: str = None,
                  velocity_scaling: float = 0.3) -> dict:
        """
        Control the amazing_hand via MoveGroup (collision-checked, same
        left_hand_fingers/right_hand_fingers SRDF groups used everywhere else).

        Args:
            side: 'left' or 'right'
            positions: 8 joint values [j11,j12,j13,j14, j21,j22,j23,j24]
                (yaw then flex per finger 1-4; thumb = finger 4 = j14/j24)
            action: shortcut name - 'open', 'close', or 'home' (SRDF group_states)
        """
        if side not in ('left', 'right'):
            return {'success': False, 'message': f'Unknown side: {side}'}
        group_name = f'{side}_hand_fingers'

        if action is not None:
            return self.move_to_named_pose(group_name, action, velocity_scaling=velocity_scaling)

        if positions is None:
            return {'success': False, 'message': 'positions (8 values) or '
                    'action ("open"/"close"/"home") required'}

        return self.move_to_joint_positions(group_name, positions, velocity_scaling=velocity_scaling)

    # ──────────────────────────────────────────────
    # Get Current EE Pose (via FK)
    # ──────────────────────────────────────────────
    
    def get_ee_pose(self, group_name: str) -> dict:
        """
        Get current end-effector pose using Forward Kinematics.
        
        Returns:
            dict with position {x,y,z} and orientation {x,y,z,w}
        """
        if not rclpy.ok():
            return {'success': False, 'message': 'ROS 2 context is invalid or shut down'}

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
        result = self._wait_for_future(future, timeout_sec=5.0)
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

