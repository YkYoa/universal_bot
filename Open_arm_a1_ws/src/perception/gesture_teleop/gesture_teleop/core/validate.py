"""Frame-level and command-level validation gates.

Each function here returns a RejectReason (NONE on success) rather than
raising or silently clamping - every rejection must be traceable to a
named cause (plan interlocks 5-9, 12).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gesture_teleop.core import landmarks as lm
from gesture_teleop.core import robot_model as rm
from gesture_teleop.core.types import RejectReason, Skeleton

VISIBILITY_THRESHOLD = 0.5
PRESENCE_THRESHOLD = 0.5
STALE_FRAME_SECONDS = 0.4  # matches teleop_node's independent landmark watchdog

# Depth-flip detector: a large out-of-plane (z) sign change together with a
# small in-plane (x,y) change indicates a monocular depth-lift flip, not
# real motion (see plan: "reaching toward vs away from the camera projects
# almost identically"). Weight x,y far above z accordingly.
DEPTH_FLIP_XY_MAX = 0.05  # meters
DEPTH_FLIP_Z_MIN = 0.10  # meters


def check_landmark_quality(skeleton: Skeleton, human_side: str) -> RejectReason:
    """Visibility/presence gate on shoulder/elbow/wrist for the given human
    arm (plan interlock 5). Missing landmarks are treated the same as
    low-confidence ones."""
    idx = lm.ARM_LANDMARKS[human_side]
    for key in ("shoulder", "elbow", "wrist"):
        point = skeleton.get_world(idx[key])
        if point is None:
            return RejectReason.LOW_VISIBILITY
        if point.visibility < VISIBILITY_THRESHOLD or point.presence < PRESENCE_THRESHOLD:
            return RejectReason.LOW_VISIBILITY
    return RejectReason.NONE


def check_staleness(now_s: float, skeleton_timestamp_s: float,
                     max_age_s: float = STALE_FRAME_SECONDS) -> RejectReason:
    if now_s - skeleton_timestamp_s > max_age_s:
        return RejectReason.STALE_FRAME
    return RejectReason.NONE


@dataclass
class DepthFlipDetector:
    """Per-arm-per-side stateful detector: tracks the previous elbow/wrist
    LOCAL (torso-frame) positions and flags a frame whose z (forward/out-
    of-plane) component flips sign while x,y barely move - the monocular
    depth-lift ambiguity, not real motion. Reject (do not smooth) on
    detection - One Euro would otherwise interpolate straight through the
    flip (plan interlock 8)."""

    _prev_elbow: np.ndarray = None
    _prev_wrist: np.ndarray = None

    def check(self, elbow_local: np.ndarray, wrist_local: np.ndarray) -> RejectReason:
        reason = RejectReason.NONE
        if self._prev_elbow is not None:
            if _is_depth_flip(self._prev_elbow, elbow_local) or _is_depth_flip(self._prev_wrist, wrist_local):
                reason = RejectReason.DEPTH_FLIP
        if reason is RejectReason.NONE:
            # Only advance state on a non-flip frame - a flip's own
            # (probably wrong) position must not become the new baseline.
            self._prev_elbow = elbow_local.copy()
            self._prev_wrist = wrist_local.copy()
        return reason

    def reset(self) -> None:
        self._prev_elbow = None
        self._prev_wrist = None


def _is_depth_flip(prev: np.ndarray, curr: np.ndarray) -> bool:
    dxy = np.hypot(curr[1] - prev[1], curr[2] - prev[2])  # torso frame: index 0 = x/forward
    dz = curr[0] - prev[0]
    sign_flip = (prev[0] > 0) != (curr[0] > 0)
    return sign_flip and dxy < DEPTH_FLIP_XY_MAX and abs(dz) > DEPTH_FLIP_Z_MIN


def check_joint_limits(side: str, q: np.ndarray, margin: float = 0.0) -> RejectReason:
    lo, hi = rm.LIMITS[side][:, 0], rm.LIMITS[side][:, 1]
    if np.any(q < lo + margin - 1e-9) or np.any(q > hi - margin + 1e-9):
        return RejectReason.JOINT_LIMIT
    return RejectReason.NONE


# Both arms' link0 (base) frames coincide at the same physical mounting
# point (openarm_robot.xacro connects both to openarm_body_link0 at
# xyz="0 0 0" - verified directly against that file), which is also
# exactly on TORSO_CAPSULE's own central axis. That makes each upper-arm
# capsule's proximal end permanently "colliding" with the torso capsule
# AND with the other arm's upper-arm capsule by construction, at ANY q
# (caught by test_check_self_collision_no_collision_at_home and
# test_check_self_collision_bimanual_arms_apart both reporting a collision
# at the home pose, where nothing is remotely close) - trim that segment's
# start away from the shared mount point before checking it against
# anything. Forearm capsules start at each arm's own elbow (genuinely
# different points) and are never trimmed.
_SHARED_MOUNT_TRIM_FRACTION = 0.65


def _trim_shared_mount(capsule: rm.Capsule) -> rm.Capsule:
    if not capsule.name.endswith("_upper_arm"):
        return capsule
    p0 = capsule.p0 + _SHARED_MOUNT_TRIM_FRACTION * (capsule.p1 - capsule.p0)
    return rm.Capsule(capsule.name, p0, capsule.p1, capsule.radius)


def check_self_collision(side: str, q: np.ndarray, other_side_q=None) -> RejectReason:
    """Capsule check against this side's own torso proximity and,
    optionally, the OTHER arm's current capsules (bimanual, plan interlock
    12). Always uses per-side FK, never mirrored (see robot_model's
    MIRROR_SIGN docstring on the elbow asymmetry)."""
    capsules = [_trim_shared_mount(c) for c in rm.arm_capsules(side, q)]
    torso = rm.torso_capsule()
    for cap in capsules:
        if rm.capsules_collide(cap, torso):
            return RejectReason.SELF_COLLISION
    if other_side_q is not None:
        other_side = "right" if side == "left" else "left"
        other_capsules = [_trim_shared_mount(c) for c in rm.arm_capsules(other_side, other_side_q)]
        for a in capsules:
            for b in other_capsules:
                if rm.capsules_collide(a, b):
                    return RejectReason.SELF_COLLISION
    return RejectReason.NONE
