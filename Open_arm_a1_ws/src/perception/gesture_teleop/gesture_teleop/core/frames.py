"""Torso reference frame construction, degeneracy/facing gates, and the
mirror-mode landmark selection.

Body-frame convention used throughout this package: x = forward (out of the
person's chest), y = left (person's own left), z = up. This matches the
robot's own base-frame handedness closely enough for direct use as
retarget.py's target directions (both are right-handed with z up).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gesture_teleop.core import landmarks as lm
from gesture_teleop.core.types import RejectReason, Skeleton

DEGENERATE_BASIS_SIN_THRESHOLD = 0.15  # ~8.6 degrees between shoulder and hip lines
FACING_MARGIN = 0.01  # normalized image-coordinate margin for the nose/ear test


@dataclass(frozen=True)
class TorsoFrame:
    """Orthonormal basis (x=forward, y=left, z=up) and origin (shoulder
    midpoint), built fresh each frame from the current world landmarks -
    deliberately not smoothed, since validate.py rejects degenerate frames
    outright rather than trying to filter through a bad basis."""

    origin: np.ndarray
    x_forward: np.ndarray
    y_left: np.ndarray
    z_up: np.ndarray

    def to_local(self, world_point: np.ndarray) -> np.ndarray:
        rel = world_point - self.origin
        return np.array([rel @ self.x_forward, rel @ self.y_left, rel @ self.z_up])


def build_torso_frame(skeleton: Skeleton):
    """Returns (TorsoFrame | None, RejectReason). None with
    DEGENERATE_TORSO_BASIS if the shoulder and hip lines are too close to
    parallel (person side-on to the camera) to define a reliable forward
    direction via Gram-Schmidt."""
    sl = skeleton.get_world(lm.LEFT_SHOULDER)
    sr = skeleton.get_world(lm.RIGHT_SHOULDER)
    hl = skeleton.get_world(lm.LEFT_HIP)
    hr = skeleton.get_world(lm.RIGHT_HIP)
    if None in (sl, sr, hl, hr):
        return None, RejectReason.LOW_VISIBILITY

    shoulder_mid = (sl.xyz + sr.xyz) / 2.0
    hip_mid = (hl.xyz + hr.xyz) / 2.0

    # y_left: shoulder_right -> shoulder_left (person's own left).
    shoulder_line = sl.xyz - sr.xyz
    shoulder_norm = np.linalg.norm(shoulder_line)
    # up_ish: hip_mid -> shoulder_mid (not yet orthogonal to y_left).
    up_ish = shoulder_mid - hip_mid
    up_norm = np.linalg.norm(up_ish)
    if shoulder_norm < 1e-6 or up_norm < 1e-6:
        return None, RejectReason.DEGENERATE_TORSO_BASIS

    y_left = shoulder_line / shoulder_norm
    up_ish_hat = up_ish / up_norm

    # Degeneracy check BEFORE Gram-Schmidt removes the shared component -
    # if the two source vectors are nearly parallel, the orthogonalized
    # result is numerically unstable even though it's normalizable.
    sin_angle = np.linalg.norm(np.cross(y_left, up_ish_hat))
    if sin_angle < DEGENERATE_BASIS_SIN_THRESHOLD:
        return None, RejectReason.DEGENERATE_TORSO_BASIS

    # Gram-Schmidt: remove the y_left component from up_ish to get z_up,
    # then x_forward completes a right-handed frame.
    z_up = up_ish_hat - (up_ish_hat @ y_left) * y_left
    z_up = z_up / np.linalg.norm(z_up)
    x_forward = np.cross(y_left, z_up)
    x_forward = x_forward / np.linalg.norm(x_forward)

    return TorsoFrame(origin=shoulder_mid, x_forward=x_forward, y_left=y_left, z_up=z_up), RejectReason.NONE


def facing_camera(skeleton: Skeleton) -> bool:
    """Cheap facing test using IMAGE-space (not world) landmarks: for a
    person facing the camera, the nose sits between the two ears in image
    x; MediaPipe does not flag this itself and happily reports mirrored
    landmarks for a person facing away, so this is a real gate, not a nicety
    (see plan interlock 7)."""
    nose = skeleton.get_image(lm.NOSE)
    left_ear = skeleton.get_image(lm.LEFT_EAR)
    right_ear = skeleton.get_image(lm.RIGHT_EAR)
    if None in (nose, left_ear, right_ear):
        return False
    lo = min(left_ear.xyz[0], right_ear.xyz[0]) - FACING_MARGIN
    hi = max(left_ear.xyz[0], right_ear.xyz[0]) + FACING_MARGIN
    return lo <= nose.xyz[0] <= hi


def arm_source_landmarks(skeleton: Skeleton, robot_side: str, mirror_mode: str):
    """Which HUMAN arm's landmarks drive the ROBOT arm `robot_side`, given
    `mirror_mode` ("mirror" | "same_side"). This is the entire
    implementation of "mirror mode" - deliberately NOT a q-space transform
    (see robot_model.MIRROR_SIGN's docstring for why that does not work past
    the elbow): each robot side already retargets independently from
    body-frame direction vectors via its own closed-form solver, so mirror
    mode is just a choice of which human keypoints feed which solver.

        mirror mode:    person's right arm -> robot's left arm (face-to-face)
        same_side mode: person's right arm -> robot's right arm

    Returns the human landmark key ("left" | "right") to read from
    core.landmarks.ARM_LANDMARKS.
    """
    if mirror_mode not in ("mirror", "same_side"):
        raise ValueError(f"unknown mirror_mode: {mirror_mode!r}")
    if mirror_mode == "same_side":
        return robot_side
    return "right" if robot_side == "left" else "left"


def arm_vectors_in_torso_frame(skeleton: Skeleton, human_side: str, frame: TorsoFrame):
    """Returns (shoulder_local, elbow_local, wrist_local, reason). Any
    missing/low-visibility landmark yields (None, None, None,
    LOW_VISIBILITY)."""
    idx = lm.ARM_LANDMARKS[human_side]
    shoulder = skeleton.get_world(idx["shoulder"])
    elbow = skeleton.get_world(idx["elbow"])
    wrist = skeleton.get_world(idx["wrist"])
    if None in (shoulder, elbow, wrist):
        return None, None, None, RejectReason.LOW_VISIBILITY
    return (
        frame.to_local(shoulder.xyz),
        frame.to_local(elbow.xyz),
        frame.to_local(wrist.xyz),
        RejectReason.NONE,
    )
