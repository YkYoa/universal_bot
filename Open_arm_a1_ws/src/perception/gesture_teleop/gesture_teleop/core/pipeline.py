"""GestureSession: the per-arm state machine tying frames.py, retarget.py,
validate.py and filters.py together.

State machine (plan): IDLE -> TRACKING -> ENGAGED -> HOLD -> DISENGAGED.

  IDLE        No valid skeleton seen (yet, or recently). No output.
  TRACKING    A good-quality frame is being seen, but engage() has not been
              called - this is the "ready, waiting for the operator" state.
              No command is ever emitted here (explicit-arm requirement).
  ENGAGED     engage() succeeded and frames keep validating: q is retargeted,
              One-Euro smoothed, and returned each frame.
  HOLD        Something just went wrong while ENGAGED (bad frame, stale
              frame, dead-man timeout) - the joint filter's last value is
              held (not ramped by this class; the 100Hz command loop in
              teleop_node/command_sink is what actually ramps velocity to
              zero via RateLimiter.brake_to_stop - see plan interlock 13).
              A good frame within hold_timeout_s returns to ENGAGED; letting
              the timeout lapse - or an explicit disengage() - moves to
              DISENGAGED.
  DISENGAGED  Not commanding. A fresh good-quality frame moves back to
              TRACKING, ready for another engage().

No command is EVER emitted outside ENGAGED (see test_pipeline.py's
"no command in IDLE or DISENGAGED" invariant test) - this is what makes
"explicit arm" real rather than a naming convention.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gesture_teleop.core import frames as fr
from gesture_teleop.core import retarget as rt
from gesture_teleop.core import validate as val
from gesture_teleop.core.filters import VectorOneEuro
from gesture_teleop.core.types import ArmTarget, GestureFrameResult, RejectReason, Skeleton

# One Euro tuning for the retargeted joint-angle stream (radians). Modest
# min_cutoff keeps slow motion jitter-free; beta lets fast motion track with
# low lag - the same trade-off the plan's filtering section describes.
JOINT_FILTER_MIN_CUTOFF = 1.0
JOINT_FILTER_BETA = 0.3
JOINT_FILTER_D_CUTOFF = 1.0

DEAD_MAN_TIMEOUT_S = 0.15
HOLD_TIMEOUT_S = 0.5


class SessionState(str, enum.Enum):
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    ENGAGED = "ENGAGED"
    HOLD = "HOLD"
    DISENGAGED = "DISENGAGED"


@dataclass
class GestureSession:
    """One instance per ROBOT arm (side="left"|"right"). Bimanual operation
    runs two independent instances sharing one TorsoFrame per video frame -
    NOT one instance per human arm, since mirror_mode can route the SAME
    human arm's landmarks to a robot side that changes independently."""

    side: str
    mirror_mode: str = "mirror"
    dead_man_timeout_s: float = DEAD_MAN_TIMEOUT_S
    hold_timeout_s: float = HOLD_TIMEOUT_S

    state: SessionState = field(default=SessionState.IDLE, init=False)
    q_prev: Optional[np.ndarray] = field(default=None, init=False)
    _filter: VectorOneEuro = field(init=False)
    _depth_flip: val.DepthFlipDetector = field(init=False)
    _last_good_time: Optional[float] = field(default=None, init=False)
    _last_heartbeat_time: Optional[float] = field(default=None, init=False)
    _hold_since: Optional[float] = field(default=None, init=False)

    def __post_init__(self):
        self._filter = VectorOneEuro(7, JOINT_FILTER_MIN_CUTOFF, JOINT_FILTER_BETA, JOINT_FILTER_D_CUTOFF)
        self._depth_flip = val.DepthFlipDetector()

    # -- explicit control surface -------------------------------------

    def engage(self, now_s: float):
        """Explicit enable (plan: OFF by default, explicit enable). Only
        succeeds from TRACKING (a good-quality frame must already be
        flowing) - never from IDLE/DISENGAGED, which is what makes this a
        real gate rather than a formality. Returns (success, reason)."""
        if self.state != SessionState.TRACKING:
            return False, f"cannot engage from {self.state.value}"
        self.state = SessionState.ENGAGED
        self._hold_since = None
        if self.q_prev is not None:
            self._filter.snap(self.q_prev, now_s)
        self._last_heartbeat_time = now_s
        return True, ""

    def disengage(self, now_s: float):
        """Explicit disable, from any state - always succeeds."""
        self.state = SessionState.DISENGAGED
        self._hold_since = None

    def heartbeat(self, now_s: float) -> None:
        """Dead-man heartbeat from the client (plan interlock 3) - call
        this on every "key held" message. Does NOT by itself keep the
        session engaged; tick()/process_frame() still check the elapsed
        time against dead_man_timeout_s."""
        self._last_heartbeat_time = now_s

    # -- per-video-frame processing -------------------------------------

    def process_frame(self, skeleton: Optional[Skeleton], torso_frame: Optional[fr.TorsoFrame],
                       now_s: float) -> ArmTarget:
        """Advance the state machine with one frame's data (or None/None
        for a frame where the whole-body skeleton or torso basis itself
        was rejected upstream) and return this side's ArmTarget for this
        tick. Never raises - every failure mode is a RejectReason."""
        reason = self._quality_gate(skeleton, torso_frame, now_s)

        if reason is not RejectReason.NONE:
            return self._on_bad_frame(now_s, reason)

        human_side = fr.arm_source_landmarks(skeleton, self.side, self.mirror_mode)
        shoulder_local, elbow_local, wrist_local, lm_reason = fr.arm_vectors_in_torso_frame(
            skeleton, human_side, torso_frame
        )
        if lm_reason is not RejectReason.NONE:
            return self._on_bad_frame(now_s, lm_reason)

        flip_reason = self._depth_flip.check(elbow_local, wrist_local)
        if flip_reason is not RejectReason.NONE:
            return self._on_bad_frame(now_s, flip_reason)

        target = rt.solve_arm(self.side, shoulder_local, elbow_local, wrist_local, q_prev=self.q_prev)
        if not target.valid:
            return self._on_bad_frame(now_s, target.reason)

        collision_reason = val.check_self_collision(self.side, target.q)
        if collision_reason is not RejectReason.NONE:
            return self._on_bad_frame(now_s, collision_reason)

        # Good frame.
        self.q_prev = target.q
        self._last_good_time = now_s
        if self.state in (SessionState.IDLE, SessionState.DISENGAGED):
            self.state = SessionState.TRACKING
        elif self.state == SessionState.HOLD:
            self.state = SessionState.ENGAGED
            self._hold_since = None
            self._filter.snap(target.q, now_s)  # resume without a smoothing ramp from the held value
        # ENGAGED and TRACKING: state unchanged, just refresh q_prev above.

        if self.state == SessionState.ENGAGED:
            filtered_q = self._filter.filter(target.q, now_s)
            return ArmTarget(side=self.side, q=filtered_q, valid=True, reason=RejectReason.NONE,
                              elbow_theta=target.elbow_theta, gimbal_band=target.gimbal_band)
        # TRACKING: never emit a command (explicit-arm requirement).
        return ArmTarget(side=self.side, q=self._held_q(), valid=False, reason=RejectReason.NOT_TRACKING)

    def tick(self, now_s: float) -> ArmTarget:
        """Advance the state machine with NO new frame this tick - the
        independent watchdog path (plan interlock 4): call this from a
        timer even when frames stop arriving entirely, so a hung upstream
        still triggers HOLD -> DISENGAGED rather than commanding the last
        good value forever."""
        if self.state not in (SessionState.ENGAGED, SessionState.HOLD):
            return ArmTarget(side=self.side, q=self._held_q(), valid=False, reason=RejectReason.NOT_TRACKING)
        if self._last_good_time is not None and now_s - self._last_good_time > self.dead_man_timeout_s:
            return self._on_bad_frame(now_s, RejectReason.STALE_FRAME)
        if self.state == SessionState.HOLD and self._hold_since is not None:
            if now_s - self._hold_since > self.hold_timeout_s:
                self.disengage(now_s)
        return ArmTarget(side=self.side, q=self._held_q(), valid=self.state == SessionState.ENGAGED,
                          reason=RejectReason.NONE if self.state == SessionState.ENGAGED else RejectReason.NOT_TRACKING)

    # -- internals --------------------------------------------------------

    def _quality_gate(self, skeleton, torso_frame, now_s) -> RejectReason:
        if skeleton is None or torso_frame is None:
            return RejectReason.DEGENERATE_TORSO_BASIS
        stale = val.check_staleness(now_s, skeleton.timestamp_s)
        if stale is not RejectReason.NONE:
            return stale
        if not fr.facing_camera(skeleton):
            return RejectReason.AMBIGUOUS_FACING
        human_side = fr.arm_source_landmarks(skeleton, self.side, self.mirror_mode)
        return val.check_landmark_quality(skeleton, human_side)

    def _on_bad_frame(self, now_s: float, reason: RejectReason) -> ArmTarget:
        if self.state == SessionState.ENGAGED:
            self.state = SessionState.HOLD
            self._hold_since = now_s
        elif self.state == SessionState.HOLD:
            if self._hold_since is not None and now_s - self._hold_since > self.hold_timeout_s:
                self.disengage(now_s)
        elif self.state == SessionState.TRACKING:
            self.state = SessionState.IDLE
        # IDLE/DISENGAGED: no state change on a bad frame.
        return ArmTarget(side=self.side, q=self._held_q(), valid=False, reason=reason)

    def _held_q(self) -> np.ndarray:
        return np.zeros(7) if self.q_prev is None else self.q_prev.copy()
