"""Tests for the sagittal mirror relationship between the two arms.

Scoped deliberately: MIRROR_SIGN is exact for the shoulder+elbow (q1-q4)
only. It does NOT extend to the wrist - see robot_model.MIRROR_SIGN's
docstring for why, and why pipeline.py/retarget.py do not use a q-space
mirror trick at all (each side retargets independently from body-frame
direction vectors). These tests exist to pin down exactly what does and
does not hold, so nobody "fixes" MIRROR_SIGN to be length-7 again without
re-deriving it.
"""
import numpy as np
import pytest

from gesture_teleop.core import robot_model as rm


def test_mirror_sign_exact_for_shoulder_and_elbow():
    rng = np.random.default_rng(42)
    for _ in range(1000):
        q_left = np.zeros(7)
        q_left[:4] = rng.uniform(rm.LIMITS["left"][:4, 0], rm.LIMITS["left"][:4, 1])
        q_right = np.zeros(7)
        q_right[:4] = rm.MIRROR_SIGN * q_left[:4]

        left = rm.forward_kinematics("left", q_left)
        right = rm.forward_kinematics("right", q_right)
        mirrored = left.ee_position.copy()
        mirrored[1] *= -1
        np.testing.assert_allclose(mirrored, right.ee_position, atol=1e-9)


def test_mirror_maps_shoulder_elbow_limits_onto_each_other():
    """The per-joint limit derivation (xacro's reflect/offset rule) already
    produces this - MIRROR_SIGN is consistent with it, not a separate fact."""
    left = rm.LIMITS["left"][:4]
    right = rm.LIMITS["right"][:4]
    for i in range(4):
        lo, hi = left[i]
        mapped = sorted((rm.MIRROR_SIGN[i] * lo, rm.MIRROR_SIGN[i] * hi))
        np.testing.assert_allclose(mapped, right[i], atol=1e-5)


def test_mirror_does_not_extend_to_wrist():
    """Documents the negative result: no single sign choice for q5 makes
    the mirror exact once the wrist is involved. If this test starts
    failing (error goes to ~0), the wrist geometry changed and MIRROR_SIGN's
    docstring + this file's module docstring should be revisited."""
    rng = np.random.default_rng(3)
    worst_either_sign = 0.0
    for _ in range(50):
        q_left = np.zeros(7)
        q_left[4] = rng.uniform(*rm.LIMITS["left"][4])
        left = rm.forward_kinematics("left", q_left)
        mirrored = left.ee_position.copy()
        mirrored[1] *= -1
        best_this_sample = min(
            np.abs(mirrored - rm.forward_kinematics("right", _q5(sign * q_left[4])).ee_position).max()
            for sign in (1.0, -1.0)
        )
        worst_either_sign = max(worst_either_sign, best_this_sample)
    assert worst_either_sign > 1e-3  # genuinely not zero for either sign


def _q5(value):
    q = np.zeros(7)
    q[4] = value
    return q


def test_elbow_asymmetry_is_real_not_a_bug():
    """The +0.0315m y-offset at joint4 has the SAME sign on both sides in
    the URDF (verified directly against the generated file in
    test_robot_model.py), so it is not mirrored like the rest of the
    geometry. Capsules and any position-based reasoning must use each
    side's own FK - never mirror one side's capsules onto the other."""
    left = rm.forward_kinematics("left", np.zeros(7))
    right = rm.forward_kinematics("right", np.zeros(7))
    # Elbow z differs between sides at q=0 - this is the asymmetry.
    assert left.origins[3][2] != pytest.approx(right.origins[3][2], abs=1e-6)
