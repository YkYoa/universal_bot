"""Pluggable output targets for a retargeted, rate-limited joint command.

Every sink has the same tiny interface: `publish(side, q)`. Swapping sinks
is how the staged rollout (plan) moves from zero-risk to real actuation
without touching teleop_node's control loop:

  Stage 5 (preview, zero actuation): PreviewSink
  Stage 6-7 (real arm):              ForwardPositionSink
  tests:                             NullSink / RecordingSink
  fallback recipe (JTC, see plan):   JtcSink

None of these import rclpy at import time except where genuinely
necessary for the ROS message types - teleop_node.py owns node lifecycle
and passes an already-constructed publisher in, so sinks stay trivially
unit-testable (see test_command_sink.py) with a plain injected callable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

# Joint name order used by every sink - must match bimanual_controllers.yaml
# (left_arm_controller / right_arm_controller / left_forward_position_controller
# / right_forward_position_controller's own `joints:` list).
JOINT_NAMES = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}


class CommandSink(Protocol):
    def publish(self, side: str, q: np.ndarray) -> None: ...


@dataclass
class NullSink:
    """Publishes nothing - used wherever a sink is required but the
    session isn't ENGAGED (e.g. TRACKING/HOLD/DISENGAGED), or in tests that
    only care about upstream behavior."""

    def publish(self, side: str, q: np.ndarray) -> None:
        pass


@dataclass
class RecordingSink:
    """Records every publish() call for test assertions - no ROS. Used by
    test_command_sink.py and by teleop_node's own integration tests
    (test_teleop_node.py) via dependency injection instead of a real
    publisher."""

    calls: list = None

    def __post_init__(self):
        if self.calls is None:
            self.calls = []

    def publish(self, side: str, q: np.ndarray) -> None:
        self.calls.append((side, np.asarray(q, dtype=np.float64).copy()))


@dataclass
class ForwardPositionSink:
    """Streams to `/<side>_forward_position_controller/commands`
    (std_msgs/Float64MultiArray) at the caller's own rate (100Hz to match
    `update_rate` in bimanual_controllers.yaml - see plan Finding 3). This
    is the PRIMARY real-arm sink: the ForwardCommandController takes a raw
    position target with no trajectory/interpolation semantics, unlike
    left_arm_controller/right_arm_controller's JointTrajectoryController,
    which re-anchors interpolation to the *measured* state on every
    message (open_loop_control defaults false) and produces a stutter at
    exactly the publish rate under gravity droop - see
    controller_switch.py's docstring for why this requires deactivating
    the trajectory controller first.

    `publish_fn` is injected (a bound `Publisher.publish` from
    teleop_node), not constructed here, so this class needs no rclpy
    import and stays unit-testable (test_command_sink.py) with a plain
    Python callable.
    """

    publish_left: Callable[[object], None]
    publish_right: Callable[[object], None]
    make_msg: Callable[[list], object]  # constructs Float64MultiArray(data=...) - injected to avoid an
                                         # unconditional rclpy import here; teleop_node supplies the real one.

    def publish(self, side: str, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (7,):
            raise ValueError(f"ForwardPositionSink expects 7 values, got {q.shape}")
        msg = self.make_msg(q.tolist())
        (self.publish_left if side == "left" else self.publish_right)(msg)


@dataclass
class JtcSink:
    """Fallback recipe (plan Finding 3's alternative, if the controller
    switch is judged too risky for a given demo): publishes a short
    JointTrajectory with 2-3 points spaced 40-60ms ahead, non-zero
    velocities, `header.stamp = 0` (start immediately - avoids racing the
    controller clock), at 20-25Hz (NOT 50Hz+ - the update_rate comment in
    bimanual_controllers.yaml notes low-velocity JTC jerk at finer
    periods). Requires `open_loop_control: true` on the target controller,
    which is NOT the deployed default for left_arm_controller/
    right_arm_controller - do not use this sink against them unless that
    parameter has been changed and redeployed.
    """

    publish_left: Callable[[object], None]
    publish_right: Callable[[object], None]
    make_msg: Callable[[list, list], object]  # (positions, velocities) -> JointTrajectory, injected
    step_s: float = 0.05

    _prev_q: dict = None

    def __post_init__(self):
        if self._prev_q is None:
            self._prev_q = {}

    def publish(self, side: str, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64)
        prev = self._prev_q.get(side, q)
        velocities = ((q - prev) / self.step_s).tolist()
        self._prev_q[side] = q.copy()
        msg = self.make_msg(q.tolist(), velocities)
        (self.publish_left if side == "left" else self.publish_right)(msg)


@dataclass
class PreviewSink:
    """Stage 5: zero actuation. Publishes to a plain topic
    (`/gesture_teleop/preview_joint_states`, sensor_msgs/JointState) that
    the EXISTING Three.js dashboard (web_visualizer/index.html) already
    knows how to render via its `joint_states` WebSocket stream pattern -
    reusing the current dashboard rather than building a second one. Real
    `*_arm_controller`s are never touched, so this validates retargeting
    against the real robot's URDF at zero risk (plan Stage 5's whole
    point)."""

    publish_fn: Callable[[object], None]
    make_msg: Callable[[list, list], object]  # (name, position) -> JointState, injected

    def publish(self, side: str, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64)
        self.publish_fn(self.make_msg(JOINT_NAMES[side], q.tolist()))
