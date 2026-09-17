#!/usr/bin/env python3
"""The real-arm side of gesture teleop: UDP landmarks in, rate-limited
joint commands out, gated by the dead-man switch, the FSM, and the
controller-exclusivity switch (plan interlocks 1-4, 11, 13-14).

Runs under the SYSTEM ROS install (never the mediapipe venv - see the
plan's dependency-isolation section): this node imports rclpy,
controller_manager_msgs, std_srvs and sensor_msgs, but never mediapipe or
cv2, so it can be smoke-tested and demoed even with the robot's own camera
dead (it never touches a camera at all - see udp_link.py's module
docstring on why the landmark source is a UDP socket, not an image topic).

Engage is refused unless:
  - the FSM (sequence_executor_node) reports robot_state == "IDLE" (plan
    interlock 2 - refuses RUNNING/PAUSED/FAULT/ESTOP/TEACHING/BOOTING);
  - the corresponding GestureSession is currently TRACKING (a good-quality
    frame must already be flowing - see pipeline.py);
  - the controller switch itself succeeds (ros2_control's own guarantee
    that MoveIt/the FSM/the Android app cannot also claim the interface).

The 100Hz control loop (`_control_tick`) is independent of the landmark
UDP arrival rate (camera framerate, much lower): every tick calls each
engaged/holding session's own `tick()` (interlock 4's independent
watchdog) and steps that side's RateLimiter toward the session's latest
target - or toward the CURRENT held position during HOLD, via
`RateLimiter.brake_to_stop()` (interlock 13: never an instant stop).
"""
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool

from gesture_teleop import command_sink as cs
from gesture_teleop import udp_link
from gesture_teleop.controller_switch import ControllerSwitch
from gesture_teleop.core import frames as fr
from gesture_teleop.core.filters import RateLimiter
from gesture_teleop.core.pipeline import GestureSession

FSM_NODE = "sequence_executor_node"
STATE_TOPIC = f"/{FSM_NODE}/state"
FSM_IDLE_STATE = "IDLE"

CONTROL_RATE_HZ = 100.0
UDP_RECV_TIMEOUT_S = 0.05  # short poll, not a blocking wait - the 100Hz timer is the real clock

# Conservative teleop-specific rate-limiter caps (plan Stage 6 starting
# point). NOT robot_hardware_interface/config/joint_limits.yaml's
# max_jerk (documented there as a placeholder, 1000.0, not a real limit -
# see plan Finding/interlock 11).
DEFAULT_MAX_VELOCITY = 0.2  # rad/s
DEFAULT_MAX_ACCELERATION = 0.5  # rad/s^2
DEFAULT_MAX_JERK = 10.0  # rad/s^3


@dataclass
class _ArmRuntime:
    session: GestureSession
    limiter: RateLimiter = field(init=False)
    engaged_by_switch: bool = False

    def __post_init__(self):
        self.limiter = RateLimiter(n=7, max_velocity=DEFAULT_MAX_VELOCITY,
                                    max_acceleration=DEFAULT_MAX_ACCELERATION, max_jerk=DEFAULT_MAX_JERK)


class GestureTeleopNode(Node):
    def __init__(self, sink: Optional[object] = None, controller_switch: Optional[object] = None,
                 mirror_mode: str = "mirror"):
        super().__init__("gesture_teleop_node")

        self._fsm_state = "BOOTING"
        self._fsm_lock = threading.Lock()
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                          reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(_FsmStateMsg(), STATE_TOPIC, self._on_fsm_state, qos)

        self._sink = sink or self._make_default_sink()
        self._switch = controller_switch or ControllerSwitch(self)

        self._arms = {
            "left": _ArmRuntime(session=GestureSession(side="left", mirror_mode=mirror_mode)),
            "right": _ArmRuntime(session=GestureSession(side="right", mirror_mode=mirror_mode)),
        }

        self._udp_socket = udp_link.open_receiver("127.0.0.1", 5056, timeout_s=UDP_RECV_TIMEOUT_S)
        self._latest_torso_frame = None
        self._latest_skeleton = None

        self.create_service(SetBool, "~/engage_left", lambda req, resp: self._handle_engage("left", req, resp))
        self.create_service(SetBool, "~/engage_right", lambda req, resp: self._handle_engage("right", req, resp))

        self.create_timer(1.0 / CONTROL_RATE_HZ, self._control_tick)
        self._udp_thread = threading.Thread(target=self._udp_recv_loop, daemon=True)
        self._udp_thread.start()

        self.get_logger().info("gesture_teleop_node ready (engage refused until FSM IDLE and TRACKING)")

    # -- FSM gate ---------------------------------------------------------

    def _on_fsm_state(self, msg) -> None:
        with self._fsm_lock:
            self._fsm_state = msg.robot_state

    def _fsm_is_idle(self) -> bool:
        with self._fsm_lock:
            return self._fsm_state == FSM_IDLE_STATE

    # -- landmark ingestion (independent thread, UDP is the only I/O here) --

    def _udp_recv_loop(self) -> None:
        while rclpy.ok():
            skeleton = udp_link.recv_skeleton(self._udp_socket)
            if skeleton is None:
                continue
            torso_frame, _reason = fr.build_torso_frame(skeleton)
            self._latest_skeleton = skeleton
            self._latest_torso_frame = torso_frame

    # -- engage / disengage service handlers --------------------------------

    def _handle_engage(self, side: str, request, response):
        if request.data:
            ok, reason = self.engage(side)
        else:
            ok, reason = self.disengage(side)
        response.success = ok
        response.message = reason
        return response

    def engage(self, side: str):
        """Returns (success, reason). See module docstring for the full
        gate: FSM IDLE + TRACKING + controller switch success, in that
        order (cheapest/most-informative check first)."""
        if not self._fsm_is_idle():
            return False, f"refused: FSM is not IDLE (currently {self._fsm_state})"
        arm = self._arms[side]
        if arm.session.state.value != "TRACKING":
            return False, f"refused: {side} arm is not currently TRACKING a person"
        ok, reason = self._switch.engage(side)
        if not ok:
            return False, f"controller switch failed: {reason}"
        arm.engaged_by_switch = True
        arm.limiter.reset(arm.session.q_prev if arm.session.q_prev is not None else [0.0] * 7)
        ok2, _ = arm.session.engage(time.monotonic())
        return ok2, "" if ok2 else "session refused engage after switch succeeded (race - retry)"

    def disengage(self, side: str):
        arm = self._arms[side]
        arm.session.disengage(time.monotonic())
        if arm.engaged_by_switch:
            self._switch.disengage(side)
            arm.engaged_by_switch = False
        return True, ""

    # -- 100Hz control loop --------------------------------------------------

    def _control_tick(self) -> None:
        now = time.monotonic()
        skeleton = self._latest_skeleton
        torso_frame = self._latest_torso_frame

        for side, arm in self._arms.items():
            if skeleton is not None:
                target = arm.session.process_frame(skeleton, torso_frame, now)
            else:
                target = arm.session.tick(now)

            if arm.session.state.value == "ENGAGED" and target.valid:
                command = arm.limiter.step(target.q, 1.0 / CONTROL_RATE_HZ)
                self._sink.publish(side, command)
            elif arm.session.state.value == "HOLD":
                command = arm.limiter.brake_to_stop(1.0 / CONTROL_RATE_HZ)
                self._sink.publish(side, command)
            elif arm.session.state.value == "DISENGAGED" and arm.engaged_by_switch:
                # tick()/process_frame() already drove the session to
                # DISENGAGED (hold timeout elapsed) - release the
                # controller switch here too, matching explicit
                # disengage()'s own cleanup.
                self._switch.disengage(side)
                arm.engaged_by_switch = False

    def _make_default_sink(self):
        pub_left = self.create_publisher(Float64MultiArray, "/left_forward_position_controller/commands", 10)
        pub_right = self.create_publisher(Float64MultiArray, "/right_forward_position_controller/commands", 10)

        def make_msg(data):
            msg = Float64MultiArray()
            msg.data = data
            return msg

        return cs.ForwardPositionSink(publish_left=pub_left.publish, publish_right=pub_right.publish,
                                       make_msg=make_msg)


def _FsmStateMsg():
    from openarm_messages.msg import FsmState

    return FsmState


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror-mode", choices=["mirror", "same_side"], default="mirror")
    args, ros_args = parser.parse_known_args(argv)

    rclpy.init(args=ros_args)
    node = GestureTeleopNode(mirror_mode=args.mirror_mode)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
