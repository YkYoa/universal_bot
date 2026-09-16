#!/usr/bin/env python3
"""
OpenArm Robot REST API Server

A Flask-based HTTP REST API that the UI team can call over the internet
to control the OpenArm bimanual robot via MoveIt2.

Architecture:
  UI (browser/app)  ──HTTP──>  Flask API  ──ROS 2──>  MoveIt EE Controller  ──>  MoveGroup
  
Endpoints:
  GET  /api/status                    - Robot status & joint positions
  GET  /api/pose/<group>              - Current EE pose (left_arm / right_arm)
  POST /api/move/pose                 - Move EE to target pose (x,y,z,qx,qy,qz,qw)
  POST /api/move/joints               - Move joints to target positions
  POST /api/plan/joints               - Dry-run /api/move/joints (no motion) - reachability/collision check
  POST /api/move/named                - Move to named pose (home, ready)
  POST /api/gripper                   - Open/close gripper
  POST /api/stop                      - Emergency stop (cancel current motion)
"""

import os
import sys
import json
import threading
import time
import math
import yaml
import os
import signal
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

def euler_to_quaternion(roll, pitch, yaw):
    """Converts intrinsic roll/pitch/yaw (radians) to a {w,x,y,z} quaternion dict."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return {
        'w': cr * cp * cy + sr * sp * sy,
        'x': sr * cp * cy - cr * sp * sy,
        'y': cr * sp * cy + sr * cp * sy,
        'z': cr * cp * sy - sr * sp * cy
    }

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from flask_compress import Compress

# Import our ROS 2 node
from moveit_api.moveit_ee_controller import MoveItEEController
from moveit_api.fsm_bridge import FsmBridge


# ─────────────────────────────────────────────
# Flask App Setup
# ─────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY'] = 'openarm-robot-api-2026'

# Gzips the dashboard's JS/HTML (three.js alone is ~650KB minified) - real
# win over a slow/lossy link to a phone/tablet. Covers text/javascript,
# application/json, text/html, etc by default; binary mesh files (STL/DAE)
# aren't in the default mimetype list and are skipped (already dense binary
# data, gzip wouldn't help there anyway).
Compress(app)

# Enable CORS for UI team access from any origin. manage_session=False: this
# API has no login/Flask-session state to preserve across socket events, and
# Flask-SocketIO's own session-copying path (flask_socketio's _handle_event
# assigning to RequestContext.session) throws
# `AttributeError: property 'session' of 'RequestContext' object has no setter`
# against Flask 3's read-only session property - on Flask-SocketIO 5.3.6 that
# fires on every connect/disconnect/emit. Skipping session management avoids
# that code path entirely rather than pinning a library version.
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading',
                     manage_session=False)

OPENARM_API_VERSION = "2026.09.15"

# Global reference to the ROS 2 controller node
controller: MoveItEEController = None

# Bridge to sequence_executor's state machine. Everything sequence- and
# scene-related goes through it; None until main() builds it.
fsm: FsmBridge = None


def _require_controller():
    """Returns True if controller is initialized, or False with an error message."""
    if controller is None:
        return False, 'MoveIt controller is not initialized or still starting up'
    return True, ''


# ─────────────────────────────────────────────
# Middleware: CORS headers for REST API
# ─────────────────────────────────────────────

@app.after_request
def add_cors_headers(response):
    """Allows any origin to call this API (the UI team's browser/app)."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response


# ─────────────────────────────────────────────
# Root Redirect
# ─────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    """Root endpoint - serves the Gateway Hub for browsers or JSON for API clients."""
    accept = request.headers.get('Accept', '')
    wants_json = ('application/json' in accept) or (request.args.get('format') == 'json')
    if not wants_json:
        hub_path = os.path.join(_web_visualizer_dir(), 'hub.html')
        if os.path.isfile(hub_path):
            return send_from_directory(_web_visualizer_dir(), 'hub.html')

    health_info = _compute_health()
    return jsonify({
        'message': 'Welcome to the OpenArm Robot API',
        'version': OPENARM_API_VERSION,
        'status': health_info.get('status', 'ok'),
        'active_build': health_info.get('active_build', 'qvic_2026'),
        'documentation': '/api/docs',
        'dashboard': '/dashboard/',
        'fsm_viewer': '/dashboard/fsm.html',
        'health': '/health',
    })


# ─────────────────────────────────────────────
# 3D Web Dashboard (web_visualizer/) + URDF/mesh serving
#
# Lets a teammate open http://<robot-ip>:5050/dashboard/ from a phone or PC
# browser and see a live 3D model of the robot, driven by the same
# joint_states WebSocket stream used by the app - no monitor/RViz needed on
# the robot's own machine (see web_visualizer/README.md).
# ─────────────────────────────────────────────

def _web_visualizer_dir():
    """Resolves the web_visualizer/ directory to serve: the installed
    package's share dir if available, else the source-tree copy (for
    running straight out of the workspace, uninstalled)."""
    try:
        share_dir = get_package_share_directory('moveit_api')
        candidate = os.path.join(share_dir, 'web_visualizer')
        if os.path.isdir(candidate):
            return candidate
    except PackageNotFoundError:
        pass
    # Fallback for running straight out of the source tree (not installed).
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web_visualizer')


@app.route('/dashboard/', methods=['GET'])
@app.route('/dashboard/<path:filename>', methods=['GET'])
def dashboard(filename='index.html'):
    """GET /dashboard/[path]: serves the 3D web dashboard's static files."""
    return send_from_directory(_web_visualizer_dir(), filename)


@app.route('/api/urdf', methods=['GET'])
def get_urdf():
    """
    Current robot_description (URDF XML), as published by
    robot_state_publisher. Used by the 3D web dashboard to build the model;
    mesh files it references are served from /packages/<pkg>/<path>.
    """
    if not controller:
        return jsonify({'success': False,
                         'message': 'URDF not received yet - controller is still starting.'}), 503
    urdf = controller.get_urdf()
    if not urdf:
        return jsonify({'success': False,
                         'message': 'URDF not received yet - robot_state_publisher may still be starting.'}), 503
    return jsonify({'success': True, 'urdf': urdf})


@app.route('/packages/<pkg_name>/<path:filepath>', methods=['GET'])
def serve_package_file(pkg_name, filepath):
    """
    Serves files (meshes, etc) out of an installed ROS package's share
    directory - resolves the URDF's package://<pkg>/<path> mesh URIs for
    the browser-side URDF loader. Read-only, and only for packages actually
    installed in this ROS environment (no arbitrary filesystem access).
    """
    try:
        share_dir = get_package_share_directory(pkg_name)
    except PackageNotFoundError:
        return jsonify({'success': False, 'message': f'Unknown package: {pkg_name}'}), 404
    return send_from_directory(share_dir, filepath)


# ─────────────────────────────────────────────
# Health Check & Telemetry
# ─────────────────────────────────────────────

# camera_bridge_node runs as its own process, outside this launch - all this
# can check is whether it has advertised its image topics on the ROS graph,
# not whether frames are actually flowing. Throttled to
# _OPTIONAL_HW_PROBE_INTERVAL_S so the dashboard's 3s /health poll doesn't
# turn into a ROS graph query on every single hit - the same "check once,
# then re-check on an interval, never spam" shape as HeadHW's own connection
# retry.
_OPTIONAL_HW_PROBE_INTERVAL_S = 5.0
_camera_bridge_probe = {'ready': False, 'checked_at': 0.0}


def _camera_bridge_ready():
    """Cached, rate-limited check for camera_bridge_node's image topics."""
    now = time.time()
    if now - _camera_bridge_probe['checked_at'] < _OPTIONAL_HW_PROBE_INTERVAL_S:
        return _camera_bridge_probe['ready']
    ready = False
    if controller:
        try:
            names = {name for name, _ in controller.get_topic_names_and_types()}
            ready = '/camera_front/image_raw' in names or '/camera_left/image_raw' in names
        except Exception:
            ready = False
    _camera_bridge_probe['ready'] = ready
    _camera_bridge_probe['checked_at'] = now
    return ready


def _compute_health():
    """Computes comprehensive health status dictionary matching dsr-gateway pattern."""
    ros_ok = False
    try:
        ros_ok = bool(rclpy.ok())
    except Exception:
        pass

    joint_states_available = False
    urdf_available = False
    moveit_ready = False
    if controller:
        try:
            joint_states_available = (controller.get_current_joint_state() is not None)
            urdf_available = (controller.get_urdf() is not None)
            moveit_ready = controller.is_movegroup_ready()
        except Exception:
            pass

    fsm_connected = False
    run_action_ready = False
    if fsm:
        try:
            fsm_connected = fsm.is_connected()
            run_action_ready = bool(fsm._run_client and fsm._run_client.server_is_ready())
        except Exception:
            pass

    store_ready = None
    try:
        from qvic_2026 import store
        store_ready = True
    except ImportError:
        store_ready = None
    except Exception:
        store_ready = False

    active_build = os.environ.get('OPENARM_BUILD', 'qvic_2026')
    fake_hw = os.environ.get('OPENARM_USE_FAKE_HARDWARE', 'false').lower() in ('true', '1')
    hardware_mode = 'fake' if fake_hw else 'real'

    degraded_reasons = []
    if not ros_ok:
        degraded_reasons.append("ROS 2 context not running")
    if not joint_states_available:
        degraded_reasons.append("No joint_states received from broadcaster")
    if not moveit_ready:
        degraded_reasons.append("MoveGroup action server not connected")
    if not fsm_connected:
        degraded_reasons.append("Sequence executor FSM not connected")

    if not ros_ok:
        status = "unhealthy"
    elif degraded_reasons:
        status = "degraded"
    else:
        status = "ok"

    return {
        'status': status,
        'version': OPENARM_API_VERSION,
        'active_build': active_build,
        'hardware_mode': hardware_mode,
        'ros_ok': ros_ok,
        'joint_states_available': joint_states_available,
        'urdf_available': urdf_available,
        'moveit_ready': moveit_ready,
        'fsm_ready': fsm_connected,
        'run_action_ready': run_action_ready,
        'store_ready': store_ready,
        'camera_bridge_ready': _camera_bridge_ready(),
        'degraded_reasons': degraded_reasons,
        'timestamp': time.time(),
    }


@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint: reports ok or degraded status."""
    data = _compute_health()
    status_code = 503 if data['status'] == 'unhealthy' else 200
    return jsonify(data), status_code


# ─────────────────────────────────────────────
# Robot Status
# ─────────────────────────────────────────────

@app.route('/api/status', methods=['GET'])
def get_status():
    """
    Get robot status including joint positions and motion state.
    
    Response:
    {
        "success": true,
        "is_moving": {"left_arm": false, "right_arm": false},
        "joint_states_available": true,
        "joints": {"openarm_left_joint1": 0.0, ...}
    }
    """
    if not controller:
        return jsonify({
            'success': False,
            'message': 'Controller is still starting up or offline',
            'is_moving': {'left_arm': False, 'right_arm': False},
            'joint_states_available': False,
            'joints': {},
        }), 503
    try:
        result = controller.get_status()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ─────────────────────────────────────────────
# Get Current End-Effector Pose
# ─────────────────────────────────────────────

@app.route('/api/pose/<group_name>', methods=['GET'])
def get_pose(group_name):
    """
    Get current end-effector pose for a planning group.
    
    URL params:
        group_name: 'left_arm' or 'right_arm'
    
    Response:
    {
        "success": true,
        "group": "left_arm",
        "position": {"x": 0.1, "y": 0.2, "z": 0.5},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    }
    """
    if not controller:
        return jsonify({'success': False, 'message': 'Controller is still starting up or offline'}), 503

    if group_name not in ('left_arm', 'right_arm'):
        return jsonify({
            'success': False,
            'message': 'group_name must be "left_arm" or "right_arm"'
        }), 400
    
    try:
        result = controller.get_ee_pose(group_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ─────────────────────────────────────────────
# Move to End-Effector Pose
# ─────────────────────────────────────────────

@app.route('/api/move/pose', methods=['POST'])
def move_to_pose():
    """
    Move end-effector to a target pose using MoveIt2 planning.
    
    Request body (JSON):
    {
        "group": "left_arm",           // or "right_arm"
        "position": {
            "x": -0.3,
            "y": 0.15,
            "z": 0.8
        },
        "orientation": {               // quaternion (optional, uses current if omitted)
            "x": 0.0,
            "y": 0.707,
            "z": 0.0,
            "w": 0.707
        },
        "position_only": false,        // optional, if true ignore orientation
        "pipeline_id": "ompl",         // optional: "ompl" or "pilz_industrial_motion_planner"
        "planner_id": "RRTConnect",    // optional: "RRTConnect", "RRTstar", "LIN", "PTP", etc.
        "velocity_scaling": 0.3,       // optional, 0.0-1.0
        "acceleration_scaling": 0.3,   // optional, 0.0-1.0
        "planning_time": 5.0,         // optional, seconds
        "num_attempts": 10,            // optional
        "position_tolerance": 0.01,    // optional, meters (default 1cm)
        "orientation_tolerance": 0.5   // optional, radians (default ~29 deg)
    }
    
    Response:
    {
        "success": true,
        "message": "Motion executed successfully",
        "planning_time": 0.45
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'JSON body required'}), 400
    
    # Validate required fields
    group = data.get('group')
    if group not in ('left_arm', 'right_arm'):
        return jsonify({
            'success': False,
            'message': 'group must be "left_arm" or "right_arm"'
        }), 400
    
    position = data.get('position')
    if not position or not all(k in position for k in ('x', 'y', 'z')):
        return jsonify({
            'success': False,
            'message': 'position with x, y, z is required'
        }), 400
    
    position_only = bool(data.get('position_only', False))
    
    # Orientation: use provided, or fetch current, or skip if position_only
    orientation = data.get('orientation')
    if orientation is None and not position_only:
        # Default to current EE orientation so the arm just moves position
        current = controller.get_ee_pose(group)
        if current.get('success') and 'orientation' in current:
            orientation = current['orientation']
        else:
            orientation = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
    elif orientation is None:
        orientation = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
    
    if not all(k in orientation for k in ('x', 'y', 'z', 'w')):
        return jsonify({
            'success': False,
            'message': 'orientation must have x, y, z, w (quaternion)'
        }), 400
    
    target_pose = {
        'position': position,
        'orientation': orientation,
    }
    
    velocity = float(data.get('velocity_scaling', 0.3))
    acceleration = float(data.get('acceleration_scaling', 0.3))
    planning_time = float(data.get('planning_time', 5.0))
    num_attempts = int(data.get('num_attempts', 10))
    pos_tol = float(data.get('position_tolerance', 0.01))
    orient_tol = float(data.get('orientation_tolerance', 0.5))
    pipeline = data.get('pipeline_id', 'ompl')
    planner = data.get('planner_id')
    
    # Auto-map friendly names to MoveIt config names
    if pipeline == 'ompl' and planner:
        mapping = {
            'RRTConnect': 'RRTConnectkConfigDefault',
            'RRTstar': 'RRTstarkConfigDefault',
            'TRRT': 'TRRTkConfigDefault',
            'LBKPIECE': 'LBKPIECEkConfigDefault'
        }
        planner = mapping.get(planner, planner)
    
    try:
        result = controller.move_to_pose(
            group_name=group,
            target_pose=target_pose,
            velocity_scaling=velocity,
            acceleration_scaling=acceleration,
            planning_time=planning_time,
            num_planning_attempts=num_attempts,
            position_tolerance=pos_tol,
            orientation_tolerance=orient_tol,
            position_only=position_only,
            pipeline_id=pipeline,
            planner_id=planner,
        )
        # Include current pose in response for reference
        if not result['success']:
            current = controller.get_ee_pose(group)
            if current.get('success'):
                result['current_position'] = current.get('position')
                result['current_orientation'] = current.get('orientation')
        status_code = 200 if result['success'] else 422
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ─────────────────────────────────────────────
# Move to Joint Positions
# ─────────────────────────────────────────────

@app.route('/api/move/joints', methods=['POST'])
def move_to_joints():
    """
    Move every joint of a planning group directly to target positions.

    Request body (JSON):
    {
        "group": "left_arm",          // any of: left_arm, right_arm, both_arms,
                                       // left_hand_fingers, right_hand_fingers, head
        "positions": [0.0, -0.5, 0.0, 0.0, 0.0, 1.0, 0.0],  // one value per joint, in order
        "unit": "rad",                 // optional, "rad" (default) or "deg"
        "duration": 3.0,               // optional, seconds (unused, kept for compat)
        "velocity_scaling": 0.3        // optional
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'JSON body required'}), 400

    group = data.get('group')
    if group not in controller.GROUP_JOINTS:
        return jsonify({
            'success': False,
            'message': f'Unknown group: {group}. Valid groups: {list(controller.GROUP_JOINTS)}'
        }), 400

    positions = data.get('positions')
    expected_joints = len(controller.GROUP_JOINTS[group])
    if not positions or len(positions) != expected_joints:
        return jsonify({
            'success': False,
            'message': f'positions must be a list of {expected_joints} joint values '
                       f'for group "{group}" (order: {controller.GROUP_JOINTS[group]})'
        }), 400

    unit = data.get('unit', 'rad')
    duration = float(data.get('duration', 3.0))
    velocity = float(data.get('velocity_scaling', 0.3))

    try:
        result = controller.move_to_joint_positions(
            group_name=group,
            positions=positions,
            duration=duration,
            velocity_scaling=velocity,
            unit=unit,
        )
        status_code = 200 if result['success'] else 422
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ─────────────────────────────────────────────
# Plan-only Joint Check (RViz2 MotionPlanning "Plan" button equivalent)
# ─────────────────────────────────────────────

@app.route('/api/plan/joints', methods=['POST'])
def plan_to_joints():
    """
    Ask MoveIt whether a joint-space target is reachable/collision-free,
    WITHOUT moving the robot. Same request/response shape as
    POST /api/move/joints - `success` tells you whether /api/move/joints
    with the same body would be expected to succeed.

    Request body (JSON): identical to POST /api/move/joints
    {
        "group": "left_arm",
        "positions": [0.0, -0.5, 0.0, 0.0, 0.0, 1.0, 0.0],
        "unit": "rad"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'JSON body required'}), 400

    group = data.get('group')
    if group not in controller.GROUP_JOINTS:
        return jsonify({
            'success': False,
            'message': f'Unknown group: {group}. Valid groups: {list(controller.GROUP_JOINTS)}'
        }), 400

    positions = data.get('positions')
    expected_joints = len(controller.GROUP_JOINTS[group])
    if not positions or len(positions) != expected_joints:
        return jsonify({
            'success': False,
            'message': f'positions must be a list of {expected_joints} joint values '
                       f'for group "{group}" (order: {controller.GROUP_JOINTS[group]})'
        }), 400

    unit = data.get('unit', 'rad')
    velocity = float(data.get('velocity_scaling', 0.3))

    try:
        result = controller.plan_to_joint_positions(
            group_name=group,
            positions=positions,
            velocity_scaling=velocity,
            unit=unit,
        )
        status_code = 200 if result['success'] else 422
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ─────────────────────────────────────────────
# Move a Single Joint
# ─────────────────────────────────────────────

@app.route('/api/move/joint', methods=['POST'])
def move_single_joint():
    """
    Move exactly one joint of a group, holding every other joint in that
    group at its current measured position. Useful for a per-joint slider UI.

    Request body (JSON):
    {
        "group": "left_arm",          // any of: left_arm, right_arm, both_arms,
                                       // left_hand_fingers, right_hand_fingers, head
        "joint": "openarm_left_joint4",  // joint name, OR 0-based index in the group
        "value": 45,                  // target value
        "unit": "deg",                // optional, "rad" or "deg" (default "rad")
        "velocity_scaling": 0.3       // optional
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'JSON body required'}), 400

    group = data.get('group')
    if group not in controller.GROUP_JOINTS:
        return jsonify({
            'success': False,
            'message': f'Unknown group: {group}. Valid groups: {list(controller.GROUP_JOINTS)}'
        }), 400

    joint = data.get('joint')
    if joint is None:
        return jsonify({'success': False, 'message': 'joint (name or index) is required'}), 400

    value = data.get('value')
    if value is None:
        return jsonify({'success': False, 'message': 'value is required'}), 400

    unit = data.get('unit', 'rad')
    velocity = float(data.get('velocity_scaling', 0.3))

    try:
        result = controller.move_single_joint(
            group_name=group,
            joint=joint,
            value=float(value),
            unit=unit,
            velocity_scaling=velocity,
        )
        status_code = 200 if result['success'] else 422
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ─────────────────────────────────────────────
# Move to Named Pose
# ─────────────────────────────────────────────

@app.route('/api/move/named', methods=['POST'])
def move_to_named():
    """
    Move to a pre-defined named pose from SRDF.
    
    Request body (JSON):
    {
        "group": "left_arm",           // left_arm, right_arm, both_arms,
                                        // left_hand_fingers, or right_hand_fingers
        "pose": "home",                // arms: "home"/"ready". hands: "open"/"close"/"home"
        "velocity_scaling": 0.3        // optional
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'JSON body required'}), 400

    group = data.get('group')
    pose_name = data.get('pose')
    velocity = float(data.get('velocity_scaling', 0.3))

    if group not in ('left_arm', 'right_arm', 'both_arms',
                      'left_hand_fingers', 'right_hand_fingers'):
        return jsonify({
            'success': False,
            'message': 'group must be one of: left_arm, right_arm, both_arms, '
                       'left_hand_fingers, right_hand_fingers'
        }), 400

    try:
        result = controller.move_to_named_pose(
            group_name=group,
            pose_name=pose_name,
            velocity_scaling=velocity,
        )
        status_code = 200 if result['success'] else 422
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ─────────────────────────────────────────────
# Gripper Control
# ─────────────────────────────────────────────

@app.route('/api/gripper', methods=['POST'])
def control_gripper():
    """
    Open or close a gripper.
    
    Request body (JSON):
    {
        "side": "left",                // or "right"
        "position": 0.044,             // 0.0 (closed) to 0.044 (fully open)
        "duration": 1.0                // optional
    }
    
    Shortcuts:
    {
        "side": "left",
        "action": "open"               // or "close"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'JSON body required'}), 400
    
    side = data.get('side')
    if side not in ('left', 'right'):
        return jsonify({
            'success': False,
            'message': 'side must be "left" or "right"'
        }), 400
    
    # Support action shortcuts
    action = data.get('action')
    if action == 'open':
        position = 0.044
    elif action == 'close':
        position = 0.0
    else:
        position = data.get('position')
        if position is None:
            return jsonify({
                'success': False,
                'message': 'position (0.0-0.044) or action ("open"/"close") required'
            }), 400
    
    duration = float(data.get('duration', 1.0))
    
    try:
        result = controller.move_gripper(side, float(position), duration)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ─────────────────────────────────────────────
# Head Control
# ─────────────────────────────────────────────

_HEAD_ROTATE_RAD = 0.1745  # ~10 deg, matches builtin:action_03's current angle

@app.route('/api/head', methods=['POST'])
def control_head():
    """
    Rotate the head. Not MoveIt-planned (head is not a planning group) -
    goes straight to head_controller, so it's fast and always available
    regardless of ee_type.

    Request body (JSON), shortcut:
    {
        "action": "left"                // or "right" / "home"
    }

    Request body (JSON), raw:
    {
        "pan": 0.0,                     // openarm_body_neck_joint, radians
        "tilt": 0.1745,                 // openarm_body_head_joint, radians -
                                          // this is the left/right axis on
                                          // this rig, despite the name
        "duration": 0.4                 // optional
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'JSON body required'}), 400

    duration = float(data.get('duration', 0.4))
    action = data.get('action')

    if action is not None:
        if action == 'left':
            pan, tilt = 0.0, _HEAD_ROTATE_RAD
        elif action == 'right':
            pan, tilt = 0.0, -_HEAD_ROTATE_RAD
        elif action == 'home':
            pan, tilt = 0.0, 0.0
        else:
            return jsonify({
                'success': False,
                'message': 'action must be "left", "right", or "home"'
            }), 400
    else:
        pan = data.get('pan')
        tilt = data.get('tilt')
        if pan is None or tilt is None:
            return jsonify({
                'success': False,
                'message': 'action ("left"/"right"/"home") or pan+tilt (radians) required'
            }), 400

    try:
        result = controller.move_head(float(pan), float(tilt), duration)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ─────────────────────────────────────────────
# amazing_hand Control (5-finger hand)
# ─────────────────────────────────────────────

@app.route('/api/hand', methods=['POST'])
def control_hand():
    """
    Control the amazing_hand (5-finger, ee_type:=amazing_hand) via named
    action or raw finger joint values. Collision-checked via MoveGroup -
    a failed/blocked move returns success:false with a MoveIt error reason.

    Request body (JSON), named shortcut:
    {
        "side": "left",                // or "right"
        "action": "open"               // or "close" / "home"
    }

    Request body (JSON), raw joints:
    {
        "side": "left",
        "positions": [0,0,0,0, 0,0,0,1.308997],  // 8 values (radians):
                                                   // j11,j12,j13,j14 (yaw),
                                                   // j21,j22,j23,j24 (flex)
                                                   // thumb = finger 4 = j14/j24
        "velocity_scaling": 0.3        // optional
    }

    Response:
    {
        "success": true,
        "message": "left_hand_fingers reached target",
        "planning_time": 0.12
    }
    or, on collision/failure:
    {
        "success": false,
        "message": "START_STATE_IN_COLLISION: Planning failed for left_hand_fingers.",
        "error_code": -10
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'JSON body required'}), 400

    side = data.get('side')
    if side not in ('left', 'right'):
        return jsonify({'success': False, 'message': 'side must be "left" or "right"'}), 400

    action = data.get('action')
    positions = data.get('positions')
    velocity = float(data.get('velocity_scaling', 0.3))

    if action is None and positions is None:
        return jsonify({
            'success': False,
            'message': 'action ("open"/"close"/"home") or positions (8 values) required'
        }), 400

    try:
        result = controller.move_hand(side, positions=positions, action=action,
                                       velocity_scaling=velocity)
        status_code = 200 if result['success'] else 422
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ─────────────────────────────────────────────
# Stop / Emergency Cancel Motion
# ─────────────────────────────────────────────

@app.route('/api/stop', methods=['POST'])
def stop_motion():
    """Stop everything now.

    Goes to the FSM's estop command, which cancels the in-flight motion goal
    and parks the robot in ESTOP until clear_fault. The old implementation set
    a module-level flag that only the (now removed) in-request sequence loop
    ever read, so it could not stop a sequence run by anything else.
    """
    ok, message = _require_fsm()
    if not ok:
        return jsonify({'success': False, 'message': message}), 503
    ok, message = fsm.send_command('estop')
    app.logger.info(f'Emergency stop: {message}')
    return jsonify({'success': ok, 'message': message}), (200 if ok else 409)


# ─────────────────────────────────────────────
# Move Both Arms Simultaneously
# ─────────────────────────────────────────────

@app.route('/api/move/both', methods=['POST'])
def move_both_arms():
    """
    Move both arms to target joint positions simultaneously.
    
    Request body (JSON):
    {
        "left_positions": [0.0, -0.5, 0.0, 0.0, 0.0, 1.0, 0.0],
        "right_positions": [0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0],
        "duration": 3.0,
        "velocity_scaling": 0.3
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'JSON body required'}), 400
    
    left_pos = data.get('left_positions')
    right_pos = data.get('right_positions')
    duration = float(data.get('duration', 3.0))
    velocity = float(data.get('velocity_scaling', 0.3))
    
    results = {}
    threads = []
    
    def move_arm(group, positions, key):
        results[key] = controller.move_to_joint_positions(
            group_name=group,
            positions=positions,
            duration=duration,
            velocity_scaling=velocity,
        )
    
    if left_pos and len(left_pos) == 7:
        t = threading.Thread(target=move_arm, args=('left_arm', left_pos, 'left'))
        threads.append(t)
    
    if right_pos and len(right_pos) == 7:
        t = threading.Thread(target=move_arm, args=('right_arm', right_pos, 'right'))
        threads.append(t)
    
    if not threads:
        return jsonify({
            'success': False,
            'message': 'Provide left_positions and/or right_positions (7 values each)'
        }), 400
    
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
    
    overall_success = all(r.get('success', False) for r in results.values())
    return jsonify({
        'success': overall_success,
        'results': results,
    })


# ─────────────────────────────────────────────
# Finite state machine: status, control, and running sequences
# ─────────────────────────────────────────────
#
# Everything sequence-related now goes through sequence_executor's two-layer
# FSM over ROS. What used to live here was a second, parallel sequence engine:
# a while-loop inside the Flask request thread driving MoveGroup directly,
# guarded by a module-level stop flag. Two engines meant two different code
# paths to the same hardware, and `loop_count: -1` pinned a worker until the
# process was killed. There is one engine now, and this is a client of it.
#
# Progress does not come back in the HTTP response - a run can last minutes.
# It arrives as `fsm_state` WebSocket events, one per transition.

FSM_UNAVAILABLE = (
    'the robot state machine is not reachable - is sequence_executor_node running?'
)


def _require_fsm():
    """Common guard for every FSM-backed route: fails if the bridge never
    started, or if sequence_executor_node has never published a state."""
    if fsm is None:
        return False, 'the ROS bridge did not start'
    if not fsm.is_connected():
        return False, FSM_UNAVAILABLE
    return True, ''


def _fsm_graph():
    """The FSM's shape, read from the executor's own config file so the diagram
    can never drift from the states the C++ actually emits."""
    global _FSM_GRAPH_CACHE
    if _FSM_GRAPH_CACHE is not None:
        return _FSM_GRAPH_CACHE
    try:
        path = os.path.join(
            get_package_share_directory('sequence_executor'), 'config', 'fsm_graph.json'
        )
        with open(path, 'r') as handle:
            _FSM_GRAPH_CACHE = json.load(handle)
    except (PackageNotFoundError, OSError, ValueError) as exc:
        app.logger.warning(f'could not read fsm_graph.json: {exc}')
        _FSM_GRAPH_CACHE = {'layers': []}
    return _FSM_GRAPH_CACHE


_FSM_GRAPH_CACHE = None


@app.route('/api/fsm/graph', methods=['GET'])
def fsm_graph():
    """Static: nodes and edges of both layers. The web page and the Android app
    lay out their diagram from this instead of hardcoding a state list."""
    return jsonify({'success': True, 'graph': _fsm_graph()})


@app.route('/api/fsm/state', methods=['GET'])
def fsm_state():
    """The latest snapshot. Prefer the `fsm_state` socket event for live use -
    this is for a cold page load."""
    if fsm is None:
        return jsonify({'success': False, 'message': 'the ROS bridge did not start'}), 503
    state = fsm.latest_state()
    if state is None:
        return jsonify({'success': False, 'message': FSM_UNAVAILABLE}), 503
    return jsonify({'success': True, 'state': state})


@app.route('/api/fsm/command', methods=['POST'])
def fsm_command():
    """pause | resume | step | cancel | estop | clear_fault | enter_teach |
    exit_teach."""
    ok, message = _require_fsm()
    if not ok:
        return jsonify({'success': False, 'message': message}), 503

    data = request.get_json(silent=True) or {}
    command = data.get('command')
    if not command:
        return jsonify({'success': False, 'message': "'command' is required"}), 400

    ok, message = fsm.send_command(command)
    # A refused command is a legitimate answer ("not paused", "no fault to
    # clear"), not a server error - 409 so a client can show the message.
    return jsonify({'success': ok, 'message': message}), (200 if ok else 409)


@app.route('/api/sequence/run', methods=['POST'])
def run_sequence():
    """Start a sequence and return immediately.

    Body: {name, repeat, velocity, dry_run}
      repeat   0 uses the sequence's own setting, -1 loops forever
      velocity 0 uses the sequence's own setting
      dry_run  walk and validate every step without sending a motion goal
    """
    ok, message = _require_fsm()
    if not ok:
        return jsonify({'success': False, 'message': message}), 503

    data = request.get_json(silent=True) or {}
    name = data.get('name')
    if not name:
        return jsonify({'success': False, 'message': "'name' is required"}), 400

    ok, message = fsm.run_sequence(
        name,
        repeat=data.get('repeat', 0),
        velocity=data.get('velocity', 0.0),
        dry_run=bool(data.get('dry_run', False)),
    )
    return jsonify({'success': ok, 'message': message}), (200 if ok else 409)


@app.route('/api/actions', methods=['GET'])
def list_actions():
    """The project's hardcoded actions. They run through the same endpoint as
    stored sequences - the name is just prefixed with `builtin:`."""
    ok, message = _require_fsm()
    if not ok:
        return jsonify({'success': False, 'message': message}), 503
    # The registry lives in the executor; there is no service to enumerate it,
    # so the names are what the client needs and the executor validates.
    return jsonify({
        'success': True,
        'hint': 'run one with POST /api/sequence/run {"name": "builtin:<id>"}',
        'actions': [
            {'id': f'action_{i:02d}', 'name': f'builtin:action_{i:02d}'}
            for i in range(1, 13)
        ],
    })


@app.route('/api/control-mode', methods=['GET'])
def control_mode():
    """Which hardware control mode the arm came up in.

    Fixed at startup - the damiao register is written once during init - so a
    UI should use this to grey out steps the robot cannot run rather than
    offering a switch that does not exist.
    """
    if fsm is None:
        return jsonify({'success': False, 'message': 'the ROS bridge did not start'}), 503
    state = fsm.latest_state()
    if state is None:
        return jsonify({'success': False, 'message': FSM_UNAVAILABLE}), 503
    mode = state['control_mode_active']
    return jsonify({
        'success': True,
        'control_mode': mode,
        'runnable_step_modes': ['any'] + (['torque'] if mode == 'torque' else ['position|mit']),
        'switchable': False,
        'note': 'Set control_mode in robot_hardware_interface/config/hardware_config.yaml '
                'and restart the hardware to change this.',
    })


# ─────────────────────────────────────────────
# Planning scene: obstacles and grasping
# ─────────────────────────────────────────────

@app.route('/api/scene/<action>', methods=['POST'])
def scene_command(action):
    """add | remove | attach | detach | allow | disallow | clear.

    `attach` is the grasp: it makes MoveIt carry the object with the arm and
    stop flagging it against the hand's links, which is what "allow collision
    while grasping" means in practice.
    """
    if fsm is None:
        return jsonify({'success': False, 'message': 'the ROS bridge did not start'}), 503

    data = request.get_json(silent=True) or {}
    ok, message = fsm.scene_command(
        action,
        object_id=data.get('object_id', ''),
        link=data.get('link', ''),
        touch_links=data.get('touch_links'),
        primitive=data.get('primitive', ''),
        dimensions=data.get('dimensions'),
        position=data.get('position'),
        orientation=data.get('orientation'),
        frame_id=data.get('frame_id', 'openarm_body_link0'),
    )
    return jsonify({'success': ok, 'message': message}), (200 if ok else 400)



@app.route('/api/move/workspace', methods=['POST'])
def move_to_workspace_point():
    """
    Move an arm to a point in the robot workspace (simplified).
    Uses position-only mode so orientation is unconstrained.

    Request body:
    {
        "group": "left_arm",
        "x": -0.3, "y": 0.15, "z": 0.8,
        "velocity_scaling": 0.3
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'JSON body required'}), 400

    group = data.get('group')
    if group not in ('left_arm', 'right_arm'):
        return jsonify({'success': False, 'message': 'group must be left_arm or right_arm'}), 400

    for k in ('x', 'y', 'z'):
        if k not in data:
            return jsonify({'success': False, 'message': f'Missing coordinate: {k}'}), 400

    vel = float(data.get('velocity_scaling', 0.3))
    target = {
        'position': {'x': float(data['x']), 'y': float(data['y']), 'z': float(data['z'])},
        'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
    }

    try:
        result = controller.move_to_pose(
            group_name=group, target_pose=target,
            velocity_scaling=vel, position_only=True,
        )
        return jsonify(result), 200 if result['success'] else 422
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ─────────────────────────────────────────────
# WebSocket: Real-time Joint State Streaming
# ─────────────────────────────────────────────

_streaming = False

@socketio.on('connect')
def ws_connect():
    """Client connected via WebSocket."""
    app.logger.info('WebSocket client connected')


@socketio.on('subscribe_joint_states')
def ws_subscribe(data=None):
    """
    Start streaming joint states to the client at ~10Hz.
    
    UI team can connect via:
        const socket = io('http://<robot-ip>:5050');
        socket.emit('subscribe_joint_states');
        socket.on('joint_states', (data) => { console.log(data); });
    """
    global _streaming
    _streaming = True
    
    def stream():
        while _streaming:
            try:
                status = controller.get_status()
                socketio.emit('joint_states', status)
            except Exception:
                pass
            time.sleep(0.1)  # 10 Hz
    
    threading.Thread(target=stream, daemon=True).start()


@socketio.on('unsubscribe_joint_states')
def ws_unsubscribe(data=None):
    """Stops the ~10Hz joint_states stream started by ws_subscribe()."""
    global _streaming
    _streaming = False


@socketio.on('disconnect')
def ws_disconnect():
    """Stops streaming when the WebSocket client disconnects."""
    global _streaming
    _streaming = False
    app.logger.info('WebSocket client disconnected')


# ─────────────────────────────────────────────
# API Documentation Endpoint
# ─────────────────────────────────────────────

@app.route('/api/docs', methods=['GET'])
def api_docs():
    """Return API documentation as JSON."""
    return jsonify({
        'name': 'OpenArm Bimanual Robot API',
        'version': '1.0.0',
        'description': 'REST API for controlling the OpenArm bimanual robot via MoveIt2',
        'base_url': 'http://<robot-ip>:5050',
        'endpoints': {
            'GET /api/health': 'Health check',
            'GET /api/status': 'Robot status & joint positions',
            'GET /api/pose/<group>': 'Current EE pose (left_arm / right_arm)',
            'POST /api/move/pose': 'Move EE to target pose {group, position:{x,y,z}, orientation:{x,y,z,w}}',
            'POST /api/move/joints': 'Move every joint of a group {group, positions:[N values], unit:"rad"/"deg"}',
            'POST /api/plan/joints': 'Dry-run of /api/move/joints - checks reachability/collision without moving '
                                      '{group, positions:[N values], unit:"rad"/"deg"}',
            'POST /api/move/joint': 'Move exactly one joint, others stay put {group, joint:name-or-index, value, unit:"rad"/"deg"}',
            'POST /api/move/named': 'Named pose {group, pose:"home"/"ready"/"open"/"close"}',
            'POST /api/move/both': 'Move both arms {left_positions, right_positions}',
            'POST /api/move/workspace': 'Move to workspace point {group, x, y, z} (position-only)',
            'POST /api/gripper': 'Gripper control {side, action:"open"/"close"} or {side, position:0.0-0.044}',
            'POST /api/hand': 'amazing_hand control {side, action:"open"/"close"/"home"} or {side, positions:[8 values]}',
            'GET /api/sequences': 'List available pose sequences',
            'POST /api/sequence': 'Run a named pose sequence {name:"wave"/"greet"/...}',
            'GET /api/docs': 'This documentation',
        },
        'websocket': {
            'url': 'ws://<robot-ip>:5050',
            'events': {
                'subscribe_joint_states': 'Start streaming joint states at 10Hz',
                'unsubscribe_joint_states': 'Stop streaming',
                'joint_states': 'Received event with current joint data',
            },
        },
        'planning_groups': {
            'left_arm': {'joints': 7, 'order': MoveItEEController.LEFT_ARM_JOINTS},
            'right_arm': {'joints': 7, 'order': MoveItEEController.RIGHT_ARM_JOINTS},
            'both_arms': {'joints': 14, 'order': MoveItEEController.BOTH_ARM_JOINTS},
            'left_hand_fingers': {'joints': 8, 'order': MoveItEEController.LEFT_HAND_JOINTS},
            'right_hand_fingers': {'joints': 8, 'order': MoveItEEController.RIGHT_HAND_JOINTS},
            'head': {'joints': 2, 'order': MoveItEEController.HEAD_JOINTS,
                     'note': 'neck (pan), then head (tilt). No named poses yet - use '
                             '/api/move/joints or /api/move/joint.'},
        },
        'units': 'All joint values default to radians. Pass "unit": "deg" in '
                 '/api/move/joints or /api/move/joint to use degrees instead.',
        'pipelines': {
            'ompl': {
                'description': 'Sampling-based planners (default). Best for complex obstacle avoidance.',
                'planners': ['RRTConnect', 'RRTstar', 'TRRT', 'LBKPIECE'],
                'smoothing': 'Ruckig (jerk-limited)'
            },
            'pilz_industrial_motion_planner': {
                'description': 'Deterministic industrial planners. Best for straight lines and simple PTP.',
                'planners': ['PTP', 'LIN', 'CIRC'],
                'restrictions': 'Single arm only (left_arm or right_arm)'
            }
        },
        'gripper_range': {'min': 0.0, 'max': 0.044, 'unit': 'meters'},
        'hand_joint_order': ['j11', 'j12', 'j13', 'j14', 'j21', 'j22', 'j23', 'j24'],
        'named_poses': {
            'left_arm': ['home', 'ready'],
            'right_arm': ['home', 'ready'],
            'both_arms': ['home', 'ready'],
            'left_hand_fingers': ['open', 'close', 'home'],
            'right_hand_fingers': ['open', 'close', 'home'],
        },
    })


# ─────────────────────────────────────────────
# Project blueprint (optional)
# ─────────────────────────────────────────────

def _register_project_blueprint():
    """Attach the sequence-store CRUD, if this workspace has a project that
    provides it.

    The store belongs to builds/qvic_2026, not here: it holds that project's
    waypoints and its idea of what a step is. This server stays generic and
    talks to the robot only over ROS, so a workspace without qvic_2026 still
    gets everything except the CRUD routes.
    """
    try:
        from qvic_2026 import sequence_api
    except ImportError:
        app.logger.info(
            'qvic_2026 not importable - sequence CRUD endpoints are not available. '
            'Everything else (FSM control, scene, direct joint moves) still works.'
        )
        return

    def read_live_joints(arm):
        """Current joint positions for one arm, so a waypoint can be recorded
        from the browser the way `record_waypoint` does from a terminal."""
        state = controller.get_current_joint_state() if controller else None
        if state is None:
            return None
        # A sensor_msgs/JointState carries parallel name/position arrays in
        # whatever order the broadcaster publishes; a waypoint needs the
        # group's own joint order.
        by_name = dict(zip(state.name, state.position))
        names = controller.GROUP_JOINTS.get(arm, [])
        values = [by_name[name] for name in names if name in by_name]
        return values if len(values) == len(names) else None

    sequence_api.register(app, joint_state_reader=read_live_joints)
    app.logger.info('Registered qvic_2026 sequence store endpoints')


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

def main():
    """Entry point: brings up the ROS 2 node/executor in a background thread,
    verifies communication, then runs the Flask+SocketIO server in
    the foreground until interrupted."""
    # Initialize ROS 2 without signal handlers to avoid conflict with Flask
    ros_initialized = False
    try:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        ros_initialized = True
    except Exception as e:
        app.logger.warning(f'[OpenArm API] rclpy.init failed: {e}')

    global controller, fsm
    executor = None
    ros_thread = None

    if ros_initialized:
        try:
            controller = MoveItEEController()
        except Exception as e:
            app.logger.warning(f'[OpenArm API] Failed to initialize MoveItEEController: {e}')
            controller = None

        try:
            fsm = FsmBridge(on_state=lambda state: socketio.emit('fsm_state', state))
        except Exception as e:
            app.logger.warning(f'[OpenArm API] Failed to initialize FsmBridge: {e}')
            fsm = None

        try:
            executor = MultiThreadedExecutor(num_threads=4)
            if controller:
                executor.add_node(controller)
            if fsm:
                executor.add_node(fsm)

            ros_thread = threading.Thread(target=executor.spin, daemon=True)
            ros_thread.start()
        except Exception as e:
            app.logger.warning(f'[OpenArm API] Failed to start ROS 2 executor: {e}')

    try:
        _register_project_blueprint()
    except Exception as e:
        app.logger.warning(f'[OpenArm API] Failed to register project blueprint: {e}')

    # Check for joint states (fast 3s timeout so the HTTP server is ready for CI)
    if controller:
        controller.get_logger().info('Checking joint states (up to 3s)...')
        for _ in range(30):
            if controller.get_current_joint_state() is not None:
                break
            time.sleep(0.1)

        if controller.get_current_joint_state() is None:
            controller.get_logger().warn('No joint states received yet - API server starting in degraded mode')
        else:
            controller.get_logger().info('Joint states verified!')

    # Get port from environment or default to 5050
    port = int(os.environ.get('ROBOT_API_PORT', 5050))
    host = os.environ.get('ROBOT_API_HOST', '0.0.0.0')

    print(f'[OpenArm API] Starting Gateway REST API server on {host}:{port}')
    print(f'[OpenArm API] Hub UI:   http://{host}:{port}/')
    print(f'[OpenArm API] Health:   http://{host}:{port}/health')
    print(f'[OpenArm API] API docs: http://{host}:{port}/api/docs')

    try:
        # Run Flask with SocketIO (for WebSocket support)
        socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        print('[OpenArm API] Shutting down...')
        if executor:
            try:
                executor.shutdown()
            except Exception:
                pass
        if controller:
            try:
                controller.destroy_node()
            except Exception:
                pass
        if fsm:
            try:
                fsm.destroy_node()
            except Exception:
                pass

        # Shutdown ROS 2 context
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

        if ros_thread and ros_thread.is_alive():
            ros_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
