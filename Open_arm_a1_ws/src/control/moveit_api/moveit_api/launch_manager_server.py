#!/usr/bin/env python3
"""
OpenArm Launch Manager REST API

A small, ROS-independent Flask server that lets a teammate remotely
start/stop/monitor the actual `ros2 launch ...` process that brings up
the robot stack (MoveIt2, controllers, the robot_api_server, etc).

This is deliberately NOT a ROS 2 node. robot_api_server.py (port 5050)
runs *inside* the stack this manager launches, so it can't be the thing
that starts that stack in the first place (chicken-and-egg). This
process just supervises `ros2 launch` as a plain subprocess and exposes
that lifecycle over HTTP.

Endpoints:
  GET  /api/health                - health check
  GET  /api/launch/presets        - list allowed launch presets + default args
  GET  /api/launch/status         - current process state (running/stopped/crashed, pid, uptime)
  POST /api/launch/start          - start a preset ({"preset": "bimanual", "args": {...}})
  POST /api/launch/stop           - stop the running launch (graceful SIGINT, escalates)
  GET  /api/launch/logs           - tail of captured stdout/stderr (?lines=200)

Only whitelisted presets/arg names can be launched - this deliberately
does not accept an arbitrary launch command from the network.
"""

import os
import re
import signal
import subprocess
import threading
import time

from flask import Flask, request, jsonify

app = Flask(__name__)

# ─────────────────────────────────────────────
# Whitelisted launch presets
#
# Add a new entry here to expose another top-level launch file - do NOT
# accept a package/launch_file pair directly from the request body, that
# would let any caller run arbitrary `ros2 launch` targets.
# ─────────────────────────────────────────────

PRESETS = {
    'bimanual': {
        'package': 'openarm_moveit_config',
        'launch_file': 'moveit_bimanual.launch.py',
        'description': 'Full bimanual stack: robot_state_publisher, ros2_control, '
                        'MoveGroup, robot_skills, RViz.',
        'default_args': {
            'use_fake_hardware': 'true',
            'use_rviz': 'true',
            'use_sim_time': 'false',
            'ee_type': 'amazing_hand',
            'body_type': 'v2',
            'use_robot_skills': 'true',
        },
    },
    'robot_api': {
        'package': 'moveit_api',
        'launch_file': 'robot_api.launch.py',
        'description': 'Full stack plus the REST/WebSocket robot_api_server on port 5050 '
                        '(what the app/UI team actually talks to for moves and sequences).',
        'default_args': {
            'use_rviz': 'false',
            'use_moveit': 'true',
            'use_api': 'true',
            'use_controllers': 'true',
            'use_rsp': 'true',
            'ee_type': 'amazing_hand',
            'body_type': 'v2',
        },
    },
}

# Only these string values are legal for a launch arg - "true"/"false" or a
# small identifier (ee_type/body_type values etc). Blocks anything that
# isn't a plain token, out of caution, even though argv (no shell=True)
# already rules out shell injection.
_ARG_VALUE_RE = re.compile(r'^[A-Za-z0-9_./:-]{1,64}$')

STOP_SIGINT_TIMEOUT = 15.0
STOP_SIGTERM_TIMEOUT = 5.0


class LaunchSupervisor:
    """Tracks at most one running `ros2 launch` subprocess at a time."""

    def __init__(self, log_dir=None):
        self._lock = threading.Lock()
        self._proc: subprocess.Popen = None
        self._preset_name = None
        self._args = {}
        self._started_at = None
        self._returncode = None
        self._log_path = None
        self._log_dir = log_dir or os.environ.get('LAUNCH_MANAGER_LOG_DIR', '/tmp')
        self._log_file = None

    def status(self) -> dict:
        with self._lock:
            if self._proc is None:
                return {'state': 'stopped'}

            poll = self._proc.poll()
            if poll is None:
                return {
                    'state': 'running',
                    'preset': self._preset_name,
                    'args': self._args,
                    'pid': self._proc.pid,
                    'uptime_sec': round(time.time() - self._started_at, 1),
                    'log_path': self._log_path,
                }

            # process exited since we last checked
            self._returncode = poll
            state = 'stopped' if poll == 0 else 'crashed'
            result = {
                'state': state,
                'preset': self._preset_name,
                'args': self._args,
                'returncode': poll,
                'log_path': self._log_path,
            }
            self._proc = None
            return result

    def start(self, preset_name: str, args: dict, force: bool = False) -> dict:
        preset = PRESETS.get(preset_name)
        if preset is None:
            return {'success': False, 'message': f'Unknown preset: {preset_name}. '
                    f'Valid presets: {list(PRESETS)}'}, 400

        allowed_keys = set(preset['default_args'])
        merged_args = dict(preset['default_args'])
        for key, value in (args or {}).items():
            if key not in allowed_keys:
                return {'success': False, 'message': f'Unknown arg "{key}" for preset '
                        f'"{preset_name}". Valid args: {sorted(allowed_keys)}'}, 400
            value_str = str(value).lower() if isinstance(value, bool) else str(value)
            if not _ARG_VALUE_RE.match(value_str):
                return {'success': False, 'message': f'Invalid value for "{key}": {value_str!r}'}, 400
            merged_args[key] = value_str

        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                if not force:
                    return {'success': False, 'message': 'A launch is already running. '
                            'Stop it first, or pass "force": true to replace it.',
                            'status': self.status()}, 409
                self._stop_locked()

            cmd = ['ros2', 'launch', preset['package'], preset['launch_file']]
            cmd += [f'{k}:={v}' for k, v in merged_args.items()]

            os.makedirs(self._log_dir, exist_ok=True)
            log_path = os.path.join(self._log_dir, f'launch_{preset_name}_{int(time.time())}.log')
            log_file = open(log_path, 'w')

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,  # own process group, so we can signal the whole tree
                )
            except FileNotFoundError:
                log_file.close()
                return {'success': False, 'message': '"ros2" not found on PATH - is the '
                        'ROS 2 environment sourced in this process?'}, 500

            self._proc = proc
            self._preset_name = preset_name
            self._args = merged_args
            self._started_at = time.time()
            self._log_path = log_path
            self._log_file = log_file

            return {
                'success': True,
                'message': f'Started "{preset_name}"',
                'pid': proc.pid,
                'command': cmd,
                'log_path': log_path,
            }, 200

    def stop(self) -> dict:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return {'success': True, 'message': 'Nothing running.'}, 200
            result = self._stop_locked()
            return result, 200

    def _stop_locked(self) -> dict:
        """Caller must hold self._lock."""
        proc = self._proc
        pid = proc.pid
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            self._proc = None
            return {'success': True, 'message': 'Process already gone.'}

        # SIGINT first - ros2 launch treats this as a clean shutdown request
        # and tears down every node it started (see README: don't kill -9).
        os.killpg(pgid, signal.SIGINT)
        try:
            proc.wait(timeout=STOP_SIGINT_TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=STOP_SIGTERM_TIMEOUT)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait(timeout=5.0)

        returncode = proc.returncode
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        self._proc = None
        return {'success': True, 'message': f'Stopped (exit code {returncode}).',
                'returncode': returncode}

    def tail_logs(self, lines: int) -> str:
        with self._lock:
            log_path = self._log_path
        if not log_path or not os.path.exists(log_path):
            return ''
        with open(log_path, 'r', errors='replace') as f:
            return ''.join(f.readlines()[-lines:])


supervisor = LaunchSupervisor()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'launch_manager'})


@app.route('/api/launch/presets', methods=['GET'])
def list_presets():
    return jsonify({'success': True, 'presets': {
        name: {'package': p['package'], 'launch_file': p['launch_file'],
               'description': p['description'], 'default_args': p['default_args']}
        for name, p in PRESETS.items()
    }})


@app.route('/api/launch/status', methods=['GET'])
def get_status():
    return jsonify({'success': True, **supervisor.status()})


@app.route('/api/launch/start', methods=['POST'])
def start_launch():
    data = request.get_json(silent=True) or {}
    preset = data.get('preset')
    if not preset:
        return jsonify({'success': False, 'message': 'preset is required',
                         'available': list(PRESETS)}), 400
    args = data.get('args', {})
    force = bool(data.get('force', False))
    result, code = supervisor.start(preset, args, force=force)
    return jsonify(result), code


@app.route('/api/launch/stop', methods=['POST'])
def stop_launch():
    result, code = supervisor.stop()
    return jsonify(result), code


@app.route('/api/launch/logs', methods=['GET'])
def get_logs():
    lines = int(request.args.get('lines', 200))
    return jsonify({'success': True, 'logs': supervisor.tail_logs(lines)})


def main():
    port = int(os.environ.get('LAUNCH_MANAGER_PORT', 5060))
    host = os.environ.get('LAUNCH_MANAGER_HOST', '0.0.0.0')
    print(f'Starting Launch Manager API on {host}:{port}')
    print(f'Presets: {list(PRESETS)}')
    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    main()
