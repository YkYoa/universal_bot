#!/usr/bin/env python3
"""ROS side of the FSM API: one rclpy node living inside the Flask process.

It exists so the HTTP layer never touches rclpy directly, and - more
importantly - so no HTTP request ever blocks on robot motion. The old
/api/sequence ran its whole loop inside the request thread with a module-level
stop flag; an infinite sequence pinned a Flask worker until the process was
killed. Here a run request sends a RunSequence goal and returns immediately,
and progress reaches the client as `fsm_state` WebSocket events instead.

Every transition is pushed, not polled: the executor publishes FsmState only
when something actually changes, and the socket forwards each one. That is a
different pattern from the 10 Hz joint_states stream in robot_api_server, and
deliberately so - an idle robot should produce no traffic at all.
"""

import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from openarm_messages.action import RunSequence
from openarm_messages.msg import FsmState
from openarm_messages.srv import FsmCommand, SceneCommand

# The executor node's name is fixed in the launch file precisely so these
# resolve, whichever executable is actually running (sequence_executor_node or
# qvic_2026's qvic_fsm_node).
FSM_NODE = 'sequence_executor_node'
STATE_TOPIC = f'/{FSM_NODE}/state'
RUN_ACTION = f'/{FSM_NODE}/run_sequence'
COMMAND_SERVICE = f'/{FSM_NODE}/fsm_command'
SCENE_SERVICE = '/robot_skills_server/scene_command'

SERVICE_TIMEOUT_S = 5.0


def state_to_dict(msg):
    """FsmState -> the JSON shape both the web page and the Android app read."""
    if msg is None:
        return None
    return {
        'stamp': msg.stamp.sec + msg.stamp.nanosec * 1e-9,
        'robot_state': msg.robot_state,
        'sequence_name': msg.sequence_name,
        'sequence_state': msg.sequence_state,
        'step_index': msg.step_index,
        'step_total': msg.step_total,
        'step_name': msg.step_name,
        'step_type': msg.step_type,
        'loop_index': msg.loop_index,
        'loop_total': msg.loop_total,
        'control_mode_active': msg.control_mode_active,
        'progress': round(float(msg.progress), 4),
        'fault_reason': msg.fault_reason,
        'motors_enabled': msg.motors_enabled,
    }


class FsmBridge(Node):
    """Subscribes to the FSM, and calls into it on behalf of HTTP clients."""

    def __init__(self, on_state=None):
        """Subscribes to FSM_NODE's state topic and opens action/service clients
        to it; `on_state(payload)` is called on every published transition."""
        super().__init__('fsm_bridge')

        # The publisher is transient_local so a late subscriber gets the
        # current state immediately; the reader has to match or it sees nothing
        # until the next transition.
        qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._state_lock = threading.Lock()
        self._latest = None
        self._on_state = on_state

        self.create_subscription(FsmState, STATE_TOPIC, self._on_state_msg, qos)
        self._run_client = ActionClient(self, RunSequence, RUN_ACTION)
        self._command_client = self.create_client(FsmCommand, COMMAND_SERVICE)
        self._scene_client = self.create_client(SceneCommand, SCENE_SERVICE)

        self.get_logger().info(f'FSM bridge watching {STATE_TOPIC}')

    # ── state ────────────────────────────────────────────────────────────────

    def _on_state_msg(self, msg):
        """Subscription callback: caches the latest state and forwards it
        (as a dict) to the on_state listener, if any."""
        payload = state_to_dict(msg)
        with self._state_lock:
            self._latest = payload
        if self._on_state:
            # Straight through to the socket - the executor only publishes on
            # real transitions, so there is nothing to throttle here.
            try:
                self._on_state(payload)
            except Exception as exc:                      # noqa: BLE001
                self.get_logger().warning(f'state listener raised: {exc}')

    def latest_state(self):
        """Returns the last received state dict, or None before the first message."""
        with self._state_lock:
            return self._latest

    def is_connected(self):
        """Has the executor ever published? False means it is not running."""
        return self.latest_state() is not None

    # ── commands ─────────────────────────────────────────────────────────────

    def ensure_ready_for_motion(self, timeout_sec=3.0):
        """Ensure the robot is cleared of faults and motors are enabled.

        Auto-clears FAULT if present, and auto-enables motors if disabled.
        Waits up to timeout_sec for confirmation.
        Returns (ok, message).
        """
        deadline = time.time() + timeout_sec
        latest = self.latest_state()

        # Step 1: If in FAULT, auto-clear fault
        if latest and latest.get('robot_state') == 'FAULT':
            ok, msg = self.send_command('clear_fault')
            if not ok:
                return False, f"Failed to clear fault before motion: {msg}"
            while time.time() < deadline:
                latest = self.latest_state()
                if latest and latest.get('robot_state') != 'FAULT':
                    break
                time.sleep(0.05)

        # Step 2: If motors not enabled, auto-enable
        latest = self.latest_state()
        if latest is None or not latest.get('motors_enabled', False):
            ok, msg = self.send_command('enable')
            if not ok:
                return False, f"Failed to enable motors: {msg}"

            while time.time() < deadline:
                latest = self.latest_state()
                if latest and latest.get('motors_enabled', False) and latest.get('robot_state') != 'FAULT':
                    return True, "Motors enabled and ready"
                time.sleep(0.05)

            return False, "Motors failed to enable within 3.0s: check physical E-stop or 48V power"

        return True, "Motors enabled and ready"

    def run_sequence(self, name, repeat=0, velocity=0.0, dry_run=False):
        """Fire and forget. Returns as soon as the goal is accepted or refused;
        watch `fsm_state` for what happens next."""
        if not dry_run:
            ok, msg = self.ensure_ready_for_motion(timeout_sec=3.0)
            if not ok:
                return False, msg

        if not self._run_client.wait_for_server(timeout_sec=SERVICE_TIMEOUT_S):
            return False, f'{RUN_ACTION} is not available - is the executor running?'

        goal = RunSequence.Goal()
        goal.sequence_name = name
        goal.repeat_override = int(repeat)
        goal.velocity_override = float(velocity)
        goal.dry_run = bool(dry_run)

        future = self._run_client.send_goal_async(goal)
        if not _wait(future, SERVICE_TIMEOUT_S):
            return False, 'the executor did not answer the goal request in time'

        handle = future.result()
        if handle is None or not handle.accepted:
            # The executor rejects rather than queues: already running, or in
            # FAULT/TEACHING. Its own log line says which.
            return False, (
                f"'{name}' was rejected - the robot is busy, faulted, or in teach mode. "
                'Check /api/fsm/state.'
            )
        return True, f"'{name}' started"

    def send_command(self, command):
        """pause / resume / step / stop / cancel / clear_fault / enter_teach /
        exit_teach / abort_to_home / enable."""
        if not self._command_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
            return False, f'{COMMAND_SERVICE} is not available - is the executor running?'

        request = FsmCommand.Request()
        request.command = command
        future = self._command_client.call_async(request)
        if not _wait(future, SERVICE_TIMEOUT_S):
            return False, f"'{command}' timed out"
        response = future.result()
        return response.success, response.message

    def scene_command(self, action, object_id='', link='', touch_links=None,
                      primitive='', dimensions=None, position=None, orientation=None,
                      frame_id='openarm_body_link0'):
        """Planning-scene edits: add/remove/attach/detach/allow/disallow/clear."""
        if not self._scene_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
            return False, f'{SCENE_SERVICE} is not available - is robot_skills_node running?'

        request = SceneCommand.Request()
        request.action = action
        request.object_id = object_id
        request.link = link
        request.touch_links = list(touch_links or [])
        request.primitive = primitive
        request.dimensions = [float(d) for d in (dimensions or [])]
        request.pose.header.frame_id = frame_id
        if position:
            request.pose.pose.position.x = float(position[0])
            request.pose.pose.position.y = float(position[1])
            request.pose.pose.position.z = float(position[2])
        quat = orientation or [0.0, 0.0, 0.0, 1.0]
        request.pose.pose.orientation.x = float(quat[0])
        request.pose.pose.orientation.y = float(quat[1])
        request.pose.pose.orientation.z = float(quat[2])
        request.pose.pose.orientation.w = float(quat[3])

        future = self._scene_client.call_async(request)
        if not _wait(future, SERVICE_TIMEOUT_S):
            return False, f"scene '{action}' timed out"
        response = future.result()
        return response.success, response.message


def _wait(future, timeout_s):
    """Wait on a future without spinning here.

    The node is already being spun by robot_api_server's executor thread, so
    spinning it again from a Flask request thread would re-enter the executor
    and deadlock. Waiting on the future's own event is the safe half.
    """
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    return done.wait(timeout_s) and future.done()
