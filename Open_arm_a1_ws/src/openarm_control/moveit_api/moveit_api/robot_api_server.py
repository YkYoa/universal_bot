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
from ament_index_python.packages import get_package_share_directory

def euler_to_quaternion(roll, pitch, yaw):
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

from flask import Flask, request, jsonify
from flask_socketio import SocketIO

# Import our ROS 2 node
from moveit_api.moveit_ee_controller import MoveItEEController


# ─────────────────────────────────────────────
# Flask App Setup
# ─────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY'] = 'openarm-robot-api-2026'

# Enable CORS for UI team access from any origin
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global reference to the ROS 2 controller node
controller: MoveItEEController = None


# ─────────────────────────────────────────────
# Middleware: CORS headers for REST API
# ─────────────────────────────────────────────

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response


# ─────────────────────────────────────────────
# Root Redirect
# ─────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    """Root endpoint - redirect to documentation."""
    return jsonify({
        'message': 'Welcome to the OpenArm Robot API',
        'documentation': '/api/docs'
    })


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'robot': 'openarm_bimanual',
        'timestamp': time.time(),
    })


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
    Move arm joints directly to target positions.
    
    Request body (JSON):
    {
        "group": "left_arm",
        "positions": [0.0, -0.5, 0.0, 0.0, 0.0, 1.0, 0.0],  // 7 joint values (radians)
        "duration": 3.0,              // optional, seconds
        "velocity_scaling": 0.3       // optional
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'JSON body required'}), 400
    
    group = data.get('group')
    if group not in ('left_arm', 'right_arm', 'both_arms'):
        return jsonify({
            'success': False,
            'message': 'group must be "left_arm", "right_arm", or "both_arms"'
        }), 400
    
    positions = data.get('positions')
    expected_joints = 14 if group == 'both_arms' else 7
    if not positions or len(positions) != expected_joints:
        return jsonify({
            'success': False,
            'message': f'positions must be a list of {expected_joints} joint values (radians)'
        }), 400
    
    duration = float(data.get('duration', 3.0))
    velocity = float(data.get('velocity_scaling', 0.3))
    
    try:
        result = controller.move_to_joint_positions(
            group_name=group,
            positions=positions,
            duration=duration,
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
        "group": "left_arm",           // or "right_arm"
        "pose": "home",                // or "ready"
        "velocity_scaling": 0.3        // optional
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'JSON body required'}), 400
    
    group = data.get('group')
    pose_name = data.get('pose')
    velocity = float(data.get('velocity_scaling', 0.3))
    
    if group not in ('left_arm', 'right_arm', 'both_arms'):
        return jsonify({
            'success': False,
            'message': 'group must be "left_arm", "right_arm", or "both_arms"'
        }), 400
    
    if pose_name not in ('home', 'ready'):
        return jsonify({
            'success': False,
            'message': 'pose must be "home" or "ready"'
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
# Stop / Emergency Cancel Motion
# ─────────────────────────────────────────────

_stop_requested = False

@app.route('/api/stop', methods=['POST'])
def stop_motion():
    """
    Emergency stop. Cancels active sequence loop execution.
    """
    global _stop_requested
    _stop_requested = True
    app.logger.info("Emergency Stop triggered by user!")
    return jsonify({
        'success': True,
        'message': 'Stop requested. Current sequences will abort immediately.'
    })


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
# Hardcoded Pose Sequences (Android team)
# ─────────────────────────────────────────────

SEQUENCES = {}

def load_sequences():
    global SEQUENCES
    try:
        share_dir = get_package_share_directory('moveit_api')
        yaml_path = os.path.join(share_dir, 'config', 'sequences.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if data and 'speed' in data:
                    SEQUENCES.clear()
                    # Parse the new compact format
                    for seq_name, speeds in data['speed'].items():
                        pose_key = None
                        if f"{seq_name}Poses" in data:
                            pose_key = f"{seq_name}Poses"
                        elif f"{seq_name}Joints" in data:
                            pose_key = f"{seq_name}Joints"
                            
                        if pose_key is not None:
                            steps = []
                            poses_dict = data[pose_key]
                            # Maintain order by iterating dict (Python 3.7+ preserves order)
                            for step_idx, (step_name, value) in enumerate(poses_dict.items()):
                                # Determine group from prefix
                                group = 'both_arms'
                                side = 'left'
                                if step_name.startswith('la'):
                                    group = 'left_arm'
                                    side = 'left'
                                elif step_name.startswith('ra'):
                                    group = 'right_arm'
                                    side = 'right'
                                elif step_name.startswith('ba'):
                                    group = 'both_arms'
                                elif step_name.startswith('grip'):
                                    group = 'grippers'
                                    side = 'left' if 'Left' in step_name else 'right'
                                    
                                vel = speeds[step_idx] if step_idx < len(speeds) else 0.3
                                
                                step_dict = {'group': group, 'velocity': vel, 'name': step_name}
                                
                                if group == 'grippers':
                                    step_dict['type'] = 'gripper'
                                    step_dict['side'] = side
                                    if isinstance(value, str):
                                        step_dict['action'] = value
                                    else:
                                        step_dict['position'] = value[0] if isinstance(value, list) else value
                                elif isinstance(value, str):
                                    step_dict['type'] = 'named'
                                    step_dict['pose'] = value
                                elif isinstance(value, list):
                                    if len(value) == 7:
                                        # Auto-detect pose vs joints
                                        # If the last 4 elements form a valid quaternion, treat as pose
                                        qx, qy, qz, qw = value[3:]
                                        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
                                        if 0.95 < norm < 1.05 and any(q != 0 for q in [qx, qy, qz]):
                                            step_dict['type'] = 'pose'
                                            step_dict['position'] = {'x': value[0], 'y': value[1], 'z': value[2]}
                                            step_dict['orientation'] = {'x': qx, 'y': qy, 'z': qz, 'w': qw}
                                        else:
                                            step_dict['type'] = 'joints'
                                            # AUTO-CONVERT degrees to radians
                                            # We assume if the absolute value of any angle is > 2*PI, it's degrees.
                                            # But safely, let's just always assume degrees based on user request "auto convert degree to radian"
                                            step_dict['positions'] = [math.radians(p) for p in value]
                                    elif len(value) == 14:
                                        step_dict['type'] = 'joints'
                                        step_dict['positions'] = [math.radians(p) for p in value]
                                    else:
                                        app.logger.warning(f"Invalid array length for {step_name}: {len(value)}")
                                        continue
                                
                                steps.append(step_dict)
                                
                            SEQUENCES[seq_name] = {
                                'description': f"Sequence {seq_name}",
                                'group': "mixed",
                                'steps': steps
                            }
                    app.logger.info(f"Loaded {len(SEQUENCES)} sequences from {yaml_path}")
        else:
            app.logger.warning(f"Sequences file not found: {yaml_path}")
    except Exception as e:
        app.logger.error(f"Failed to load sequences: {e}")

# Call immediately
load_sequences()

@app.route('/api/sequences', methods=['GET'])
def list_sequences():
    """List all available named sequences."""
    load_sequences()
    result = {}
    for name, seq in SEQUENCES.items():
        result[name] = {
            'description': seq.get('description', ''),
            'group': seq.get('group', ''),
            'num_steps': len(seq.get('steps', [])),
        }
    return jsonify({'success': True, 'sequences': result})


@app.route('/api/sequence', methods=['POST'])
def run_sequence():
    """
    Run a named pose sequence.

    Request body:
    {
        "name": "wave",              // sequence name
        "velocity_scaling": 0.5,     // optional override
        "loop_count": 1              // optional, number of loops (use -1 or 0 for infinite)
    }
    """
    global _stop_requested
    _stop_requested = False

    load_sequences()
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'JSON body required'}), 400

    seq_name = data.get('name')
    if seq_name not in SEQUENCES:
        return jsonify({
            'success': False,
            'message': f'Unknown sequence: {seq_name}',
            'available': list(SEQUENCES.keys()),
        }), 400

    vel_override = data.get('velocity_scaling')
    seq = SEQUENCES[seq_name]
    
    # Extract loop configurations
    loop_val = data.get('loop_count', 1)
    infinite = False
    if isinstance(loop_val, str):
        if loop_val.lower() == 'infinite':
            infinite = True
            loop_count = 1
        else:
            try:
                loop_count = int(loop_val)
            except ValueError:
                loop_count = 1
    else:
        try:
            loop_count = int(loop_val)
        except (ValueError, TypeError):
            loop_count = 1

    if loop_count <= 0:
        infinite = True

    results = []
    iteration = 0

    while infinite or iteration < loop_count:
        if _stop_requested:
            app.logger.info("Sequence aborted by user request before starting iteration.")
            break
            
        iter_suffix = f" (Iteration {iteration + 1})" if not infinite else f" (Infinite Iteration {iteration + 1})"
        app.logger.info(f"Executing sequence: {seq_name}{iter_suffix}")

        for i, step in enumerate(seq.get('steps', [])):
            if _stop_requested:
                break
                
            try:
                vel = float(vel_override) if vel_override else float(step.get('velocity', 0.3))
                
                if step['type'] == 'named':
                    r = controller.move_to_named_pose(
                        group_name=step['group'],
                        pose_name=step['pose'],
                        velocity_scaling=vel,
                    )
                elif step['type'] == 'joints':
                    r = controller.move_to_joint_positions(
                        group_name=step['group'],
                        positions=step['positions'],
                        duration=float(step.get('duration', 3.0)),
                        velocity_scaling=vel,
                    )
                elif step['type'] == 'pose':
                    target = {'position': step['position']}
                    if 'orientation' in step:
                        target['orientation'] = step['orientation']
                    else:
                        target['orientation'] = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
                        
                    position_only = step.get('position_only', False)
                    if 'orientation' not in step:
                        position_only = True
                        
                    r = controller.move_to_pose(
                        group_name=step['group'], 
                        target_pose=target,
                        velocity_scaling=vel, 
                        position_only=position_only,
                    )
                elif step['type'] == 'gripper':
                    if 'action' in step:
                        pos = 0.044 if step['action'] == 'open' else 0.0
                    else:
                        pos = step['position']
                    r = controller.move_gripper(step['side'], pos)
                else:
                    r = {'success': False, 'message': f'Unknown step type: {step["type"]}'}

                results.append({
                    'iteration': iteration + 1,
                    'step': i,
                    'name': step.get('name'),
                    'success': r.get('success', False)
                })

                if not r.get('success', False):
                    return jsonify({
                        'success': False,
                        'message': f'Sequence failed at iteration {iteration + 1}, step {i}',
                        'step_results': results,
                    }), 422
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'Iteration {iteration + 1}, Step {i} error: {str(e)}',
                    'step_results': results,
                }), 500
        
        iteration += 1

    if _stop_requested:
        _stop_requested = False # Reset flag for future runs
        return jsonify({
            'success': True,
            'message': f'Sequence "{seq_name}" aborted by user stop request.',
            'step_results': results
        })

    return jsonify({
        'success': True,
        'message': f'Sequence "{seq_name}" completed successfully for {iteration} loop(s).',
        'step_results': results
    })


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
    global _streaming
    _streaming = False


@socketio.on('disconnect')
def ws_disconnect():
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
            'POST /api/move/joints': 'Move joints {group, positions:[7 values]}',
            'POST /api/move/named': 'Named pose {group, pose:"home"/"ready"}',
            'POST /api/move/both': 'Move both arms {left_positions, right_positions}',
            'POST /api/move/workspace': 'Move to workspace point {group, x, y, z} (position-only)',
            'POST /api/gripper': 'Gripper control {side, action:"open"/"close"} or {side, position:0.0-0.044}',
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
        'planning_groups': ['left_arm', 'right_arm', 'both_arms'],
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
        'named_poses': {
            'left_arm': ['home', 'ready'],
            'right_arm': ['home', 'ready'],
            'both_arms': ['home', 'ready'],
        },
    })


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

def main():
    # Initialize ROS 2
    # Initialize ROS 2 without signal handlers to avoid conflict with Flask
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    
    global controller
    controller = MoveItEEController()
    
    # Run ROS 2 executor in a background thread
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(controller)
    
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()
    
    # Wait for joint states
    controller.get_logger().info('Waiting for joint states...')
    for _ in range(100):  # Wait up to 10 seconds
        if controller.get_current_joint_state() is not None:
            break
        time.sleep(0.1)
    
    if controller.get_current_joint_state() is None:
        controller.get_logger().warn('No joint states received yet - API will start anyway')
    else:
        controller.get_logger().info('Joint states received!')
    
    # Get port from environment or default to 5050
    port = int(os.environ.get('ROBOT_API_PORT', 5050))
    host = os.environ.get('ROBOT_API_HOST', '0.0.0.0')
    
    controller.get_logger().info(f'Starting REST API server on {host}:{port}')
    controller.get_logger().info(f'API docs: http://{host}:{port}/api/docs')
    
    try:
        # Run Flask with SocketIO (for WebSocket support)
        socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        if controller:
            controller.get_logger().info('Shutting down...')
            # Stop the executor and join the ROS thread
            executor.shutdown()
            controller.destroy_node()
        
        # Shutdown ROS 2 context
        if rclpy.ok():
            rclpy.shutdown()
        
        if ros_thread.is_alive():
            ros_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
