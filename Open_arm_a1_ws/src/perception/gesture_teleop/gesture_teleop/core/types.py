"""Frozen dataclasses shared across the gesture_teleop core.

Pure Python + numpy only. No rclpy, no flask, no mediapipe, no cv2 - see
test/test_no_ros_in_core.py, which enforces this with an AST walk.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SIDES = ("left", "right")


class RejectReason(enum.Enum):
    """Why a frame or a solve did not produce a command. Every rejection
    surfaces one of these - never a silent clamp or a swallowed exception
    (see interlocks 5-9 in the plan)."""

    NONE = "none"
    LOW_VISIBILITY = "low_visibility"
    STALE_FRAME = "stale_frame"
    DEGENERATE_TORSO_BASIS = "degenerate_torso_basis"
    AMBIGUOUS_FACING = "ambiguous_facing"
    DEPTH_FLIP = "depth_flip"
    UNREACHABLE = "unreachable"
    JOINT_LIMIT = "joint_limit"
    SELF_COLLISION = "self_collision"
    NOT_TRACKING = "not_tracking"


@dataclass(frozen=True)
class LandmarkPoint:
    """One MediaPipe landmark. `xyz` is in the source's native units - either
    normalized image coordinates (pose_landmarks, z is a rough relative
    depth) or metric world coordinates in meters (pose_world_landmarks,
    hip-midpoint origin). `visibility` and `presence` are MediaPipe's own
    per-landmark confidence signals, always in [0, 1]."""

    xyz: np.ndarray  # shape (3,)
    visibility: float
    presence: float

    def __post_init__(self):
        arr = np.asarray(self.xyz, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"LandmarkPoint.xyz must be shape (3,), got {arr.shape}")
        object.__setattr__(self, "xyz", arr)


@dataclass(frozen=True)
class Skeleton:
    """One MediaPipe Pose detection, image-space and world-space landmarks
    kept side by side because they serve different purposes: image-space for
    the overlay and for visibility gating, world-space (metric, hip origin)
    for all angle math. `timestamp_s` is the server's own monotonic clock,
    never the browser's."""

    timestamp_s: float
    image_landmarks: dict[int, LandmarkPoint]
    world_landmarks: dict[int, LandmarkPoint]

    def get_world(self, idx: int) -> Optional[LandmarkPoint]:
        return self.world_landmarks.get(idx)

    def get_image(self, idx: int) -> Optional[LandmarkPoint]:
        return self.image_landmarks.get(idx)


@dataclass(frozen=True)
class ArmTarget:
    """Result of retargeting one arm for one frame."""

    side: str  # "left" | "right"
    q: np.ndarray  # shape (7,), radians, may be stale/held if not valid
    valid: bool
    reason: RejectReason = RejectReason.NONE
    swivel: float = 0.0  # solver debug: chosen swivel angle psi (radians)
    elbow_theta: float = 0.0  # angle(upper_arm, forearm) achieved, radians
    gimbal_band: bool = False  # True while |sin(beta)| < the branch-guard threshold

    def __post_init__(self):
        arr = np.asarray(self.q, dtype=np.float64)
        if arr.shape != (7,):
            raise ValueError(f"ArmTarget.q must be shape (7,), got {arr.shape}")
        object.__setattr__(self, "q", arr)


@dataclass(frozen=True)
class GestureFrameResult:
    """The complete output of running one frame through the pipeline: one
    ArmTarget per side plus the raw skeleton, so mock_server can render the
    web overlay and the robot mockup from the SAME numbers the retargeter
    used, rather than re-deriving anything client-side."""

    timestamp_s: float
    skeleton: Optional[Skeleton]
    arms: dict[str, ArmTarget]
    mirror_mode: str = "mirror"  # "mirror" | "same_side"
    state: str = "IDLE"  # GestureSession FSM state name, see pipeline.py


@dataclass(frozen=True)
class CapsuleCollision:
    """One self-collision check result."""

    colliding: bool
    capsule_a: str = ""
    capsule_b: str = ""
    distance: float = float("inf")
