"""Unit tests for core/frames.py."""
import numpy as np
import pytest

from gesture_teleop.core import frames as fr
from gesture_teleop.core.types import LandmarkPoint, RejectReason, Skeleton
from gesture_teleop.core import landmarks as lm


def _pt(x, y, z, vis=1.0, pres=1.0):
    return LandmarkPoint(xyz=np.array([x, y, z]), visibility=vis, presence=pres)


def _skeleton_upright(facing=True, side_on=False):
    """A synthetic person standing upright, roughly facing +x (world),
    shoulders along world y, torso along world z. World landmarks are in
    meters, hip-midpoint-ish origin; image landmarks are normalized [0,1]."""
    world = {
        lm.LEFT_SHOULDER: _pt(0.0, 0.2, 1.4),
        lm.RIGHT_SHOULDER: _pt(0.0, -0.2, 1.4),
        lm.LEFT_HIP: _pt(0.0, 0.1, 1.0),
        lm.RIGHT_HIP: _pt(0.0, -0.1, 1.0),
        lm.LEFT_ELBOW: _pt(0.1, 0.35, 1.2),
        lm.RIGHT_ELBOW: _pt(0.1, -0.35, 1.2),
        lm.LEFT_WRIST: _pt(0.2, 0.45, 1.05),
        lm.RIGHT_WRIST: _pt(0.2, -0.45, 1.05),
    }
    if side_on:
        # Shoulder line nearly vertical, matching the (vertical)
        # hip-to-shoulder direction - degenerate for Gram-Schmidt.
        world[lm.LEFT_SHOULDER] = _pt(0.0, 0.0, 1.4)
        world[lm.RIGHT_SHOULDER] = _pt(0.0, 0.001, 1.0)
        world[lm.LEFT_HIP] = _pt(0.0, 0.0, 1.0)
        world[lm.RIGHT_HIP] = _pt(0.0, 0.001, 0.6)

    image = {
        lm.NOSE: _pt(0.5, 0.3, 0.0),
        lm.LEFT_EAR: _pt(0.55 if facing else 0.3, 0.3, 0.0),
        lm.RIGHT_EAR: _pt(0.45 if facing else 0.2, 0.3, 0.0),
    }
    return Skeleton(timestamp_s=0.0, image_landmarks=image, world_landmarks=world)


def test_build_torso_frame_orthonormal():
    skel = _skeleton_upright()
    frame, reason = fr.build_torso_frame(skel)
    assert reason is RejectReason.NONE
    assert frame is not None
    basis = np.array([frame.x_forward, frame.y_left, frame.z_up])
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-9)
    # Right-handed: x cross y == z
    np.testing.assert_allclose(np.cross(frame.x_forward, frame.y_left), frame.z_up, atol=1e-9)


def test_build_torso_frame_y_left_points_toward_left_shoulder():
    skel = _skeleton_upright()
    frame, _ = fr.build_torso_frame(skel)
    # left shoulder should have a positive y_left-frame component relative
    # to the shoulder midpoint origin
    local = frame.to_local(np.array([0.0, 0.2, 1.4]))
    assert local[1] > 0


def test_build_torso_frame_degenerate_when_side_on():
    skel = _skeleton_upright(side_on=True)
    frame, reason = fr.build_torso_frame(skel)
    assert frame is None
    assert reason is RejectReason.DEGENERATE_TORSO_BASIS


def test_build_torso_frame_missing_landmark():
    skel = _skeleton_upright()
    del skel.world_landmarks[lm.LEFT_HIP]
    frame, reason = fr.build_torso_frame(skel)
    assert frame is None
    assert reason is RejectReason.LOW_VISIBILITY


def test_facing_camera_true_and_false():
    assert fr.facing_camera(_skeleton_upright(facing=True))
    assert not fr.facing_camera(_skeleton_upright(facing=False))


def test_facing_camera_missing_landmark_is_false():
    skel = _skeleton_upright()
    del skel.image_landmarks[lm.NOSE]
    assert not fr.facing_camera(skel)


@pytest.mark.parametrize(
    "robot_side,mode,expected_human_side",
    [
        ("left", "mirror", "right"),
        ("right", "mirror", "left"),
        ("left", "same_side", "left"),
        ("right", "same_side", "right"),
    ],
)
def test_arm_source_landmarks_mapping(robot_side, mode, expected_human_side):
    assert fr.arm_source_landmarks(None, robot_side, mode) == expected_human_side


def test_arm_source_landmarks_invalid_mode():
    with pytest.raises(ValueError):
        fr.arm_source_landmarks(None, "left", "bogus")


def test_arm_vectors_in_torso_frame_returns_local_points():
    skel = _skeleton_upright()
    frame, _ = fr.build_torso_frame(skel)
    shoulder, elbow, wrist, reason = fr.arm_vectors_in_torso_frame(skel, "left", frame)
    assert reason is RejectReason.NONE
    assert shoulder.shape == (3,)
    assert elbow.shape == (3,)
    assert wrist.shape == (3,)


def test_arm_vectors_in_torso_frame_missing_landmark():
    skel = _skeleton_upright()
    frame, _ = fr.build_torso_frame(skel)
    del skel.world_landmarks[lm.LEFT_WRIST]
    shoulder, elbow, wrist, reason = fr.arm_vectors_in_torso_frame(skel, "left", frame)
    assert shoulder is None and elbow is None and wrist is None
    assert reason is RejectReason.LOW_VISIBILITY
