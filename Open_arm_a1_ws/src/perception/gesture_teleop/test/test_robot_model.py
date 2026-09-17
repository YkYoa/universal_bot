"""Unit tests for core/robot_model.py. Pure numpy - no ROS, no MediaPipe, no
hardware. Run with: python -m pytest test/test_robot_model.py -q
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from gesture_teleop.core import robot_model as rm

FIXTURES = Path(__file__).parent / "fixtures"
# Repo-relative paths to the authoritative yaml and the checked-in
# single-arm demo URDF, resolved from this test file's location so they
# work regardless of cwd. Layout as of commit bc37f8db5 ("restructure into
# assets/robot + assets/end_effector layout") - openarm_description moved
# its flat config/urdf/meshes dirs under
# assets/robot/openarm_v1.0/... mid-development-session; update these two
# paths together if it moves again (content, not just location, mattered
# enough to verify by diff when this happened - see git log on this file).
_OPENARM_DESCRIPTION = Path(__file__).resolve().parents[3] / "openarm_description"
JOINT_LIMITS_YAML = (
    _OPENARM_DESCRIPTION / "assets" / "robot" / "openarm_v1.0" / "config"
    / "arm" / "v10" / "joint_limits.yaml"
)
# NOTE: this checked-in file is a SINGLE-ARM demo URDF (joint names like
# "openarm_joint4", no left_/right_ prefix) generated with arm_prefix=''
# (see openarm_robot.xacro), not the bimanual robot description actually
# deployed - core/robot_model.py's per-side constants were independently
# verified against the real bimanual URDF generated on the robot itself
# (tools/gen_golden_fk.py -> test/fixtures/fk_golden.json), which this file
# has no bearing on. It is used here only as a second, local source to
# cross-check RAW_LIMITS's j4 value against.
GENERATED_URDF = _OPENARM_DESCRIPTION / "assets" / "robot" / "openarm_v1.0" / "urdf" / "v10.urdf"


def _parse_raw_limits_from_yaml(text: str):
    """Minimal, dependency-free parser for this specific yaml's shape (avoids
    adding a PyYAML dependency to the test suite for one file)."""
    limits = []
    for i in range(1, 8):
        block = re.search(rf"joint{i}:.*?(?=\njoint{i + 1}:|\Z)", text, re.S)
        body = block.group(0) if block else re.search(rf"joint{i}:.*", text, re.S).group(0)
        lower = float(re.search(r"lower:\s*(-?[\d.]+)", body).group(1))
        upper = float(re.search(r"upper:\s*(-?[\d.]+)", body).group(1))
        limits.append((lower, upper))
    return tuple(limits)


@pytest.mark.skipif(not JOINT_LIMITS_YAML.is_file(), reason="repo layout not available")
def test_raw_limits_match_yaml():
    """RAW_LIMITS must track
    assets/robot/openarm_v1.0/config/arm/v10/joint_limits.yaml exactly,
    including joint4's lower bound (the yaml's authoritative -0.02 safety
    margin - see the module docstring, and
    test_generated_urdf_j4_lower_matches_authoritative_margin for why a
    generated URDF is not always trustworthy for this specific value)."""
    text = JOINT_LIMITS_YAML.read_text(encoding="utf-8")
    parsed = _parse_raw_limits_from_yaml(text)
    assert parsed == rm.RAW_LIMITS


@pytest.mark.skipif(not GENERATED_URDF.is_file(), reason="repo layout not available")
def test_generated_urdf_j4_lower_matches_authoritative_margin():
    """Cross-check RAW_LIMITS's j4 lower bound against this checked-in
    single-arm demo URDF (see GENERATED_URDF's comment above - it has no
    left_/right_ prefix). This copy happens to already carry the
    authoritative -0.02 safety margin (regenerated fresh as of commit
    bc37f8db5), unlike the bimanual URDF actually deployed to the robot at
    the time robot_model.py's module docstring was written, which still
    had the stale 0.0 - that staleness was a fact about the ROBOT's build
    artifact, not something re-provable from whatever happens to be
    checked in here at any given moment. This test only pins down that
    RAW_LIMITS uses the authoritative value, not a claim about which
    generated file is or isn't stale right now."""
    text = GENERATED_URDF.read_text(encoding="utf-8", errors="replace")
    body = re.search(
        r'<joint name="openarm_joint4" type="revolute">(.*?)</joint>', text, re.S
    ).group(1)
    lower = float(re.search(r'lower="([-\d.eE]+)"', body).group(1))
    assert lower == -0.02
    assert rm.RAW_LIMITS[3][0] == -0.02


def test_derive_limits_left_right_asymmetry():
    """Per-side limits are NOT symmetric and NOT the raw yaml values for
    j1/j2 - this is the central, easy-to-get-wrong fact from Finding 2."""
    left = rm.LIMITS["left"]
    right = rm.LIMITS["right"]
    np.testing.assert_allclose(left[0], [-3.490659, 1.396263], atol=1e-6)
    np.testing.assert_allclose(right[0], [-1.396263, 3.490659], atol=1e-6)
    np.testing.assert_allclose(left[1], [-3.316125, 0.174533], atol=1e-5)
    np.testing.assert_allclose(right[1], [-0.174533, 3.316125], atol=1e-5)
    # j3, j5, j6, j7 pass through unchanged and identical both sides.
    for i in (2, 4, 5, 6):
        np.testing.assert_allclose(left[i], right[i], atol=1e-12)
    np.testing.assert_allclose(left[3], [-0.02, 2.443461], atol=1e-9)
    np.testing.assert_allclose(right[3], [-0.02, 2.443461], atol=1e-9)


def test_derive_limits_lower_always_below_upper():
    for side in rm.SIDES:
        bounds = rm.LIMITS[side]
        assert np.all(bounds[:, 0] < bounds[:, 1])


def test_fk_against_golden():
    """The anchor test: this package's hand-rolled FK must match PyKDL's
    ChainFkSolverPos_recursive (an independent implementation, run on the
    robot against the real URDF) to double precision. See
    tools/gen_golden_fk.py for how the fixture was produced."""
    golden = json.loads((FIXTURES / "fk_golden.json").read_text())
    for side in rm.SIDES:
        for case in golden[side]:
            q = np.array(case["q"])
            frames = rm.forward_kinematics(side, q)
            np.testing.assert_allclose(
                frames.ee_position, case["ee_position"], atol=1e-9,
                err_msg=f"{side} q={q}",
            )
            quat = rm.ee_quaternion_xyzw(frames.ee_rotation)
            golden_quat = np.array(case["ee_quaternion_xyzw"])
            # Quaternions double-cover SO(3): q and -q represent the same
            # rotation, so compare both signs.
            err = min(np.abs(quat - golden_quat).max(), np.abs(quat + golden_quat).max())
            assert err < 1e-8, f"{side} q={q} quat mismatch: {quat} vs {golden_quat}"


def test_zxz_shoulder_identity():
    """R_shoulder(q1,q2,q3) == Rz(q1) @ Rx(reflect*pi/2 - q2) @ Rz(q3) exactly
    - the algebraic fact retarget.py's closed-form seed depends on. This
    holds to double precision even though the shoulder is NOT a zero-offset
    spherical joint positionally (see test_shoulder_is_not_exactly_spherical)."""
    def rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    rng = np.random.default_rng(7)
    for side in rm.SIDES:
        refl = rm.SHOULDER_REFLECT[side]
        for _ in range(300):
            q = np.zeros(7)
            q[:3] = rng.uniform(-2.0, 2.0, 3)
            frames = rm.forward_kinematics(side, q)
            expected = rz(q[0]) @ rx(refl * np.pi / 2 - q[1]) @ rz(q[2])
            np.testing.assert_allclose(frames.ee_rotation, expected, atol=1e-9)


def test_shoulder_is_not_exactly_spherical():
    """Documents the corrected Finding 2: j1/j2/j3's axes do NOT intersect
    at one point in general (residual on the order of centimeters). This
    test exists so nobody re-introduces a "spherical shoulder" assumption
    into retarget.py without noticing this is only approximately true - see
    the module docstring for the full story and why the rotation identity
    above is still exact and still the basis for the closed form."""
    rng = np.random.default_rng(11)
    max_resid = 0.0
    for side in rm.SIDES:
        for _ in range(100):
            q = np.zeros(7)
            q[:3] = rng.uniform(-2.0, 2.0, 3)
            frames = rm.forward_kinematics(side, q)
            resid = _concurrency_residual(frames.origins[:3], frames.axes_world[:3])
            max_resid = max(max_resid, resid)
    # It is NOT near zero (that would mean it IS spherical - surprising, and
    # worth re-reading the docstring if this ever fires) ...
    assert max_resid > 0.01
    # ... but it is bounded and small relative to the arm's own segment
    # lengths (~0.22m), consistent with a small mechanical offset rather
    # than a modeling error.
    assert max_resid < 0.10


def _concurrency_residual(points, axes):
    """Least-squares best-fit common point across N (point, axis) lines;
    returns the worst perpendicular distance from that point to any line."""
    m, b = [], []
    for p, a in zip(points, axes):
        proj = np.eye(3) - np.outer(a, a)
        m.append(proj)
        b.append(proj @ p)
    x, *_ = np.linalg.lstsq(np.vstack(m), np.concatenate(b), rcond=None)
    return max(
        np.linalg.norm((np.eye(3) - np.outer(a, a)) @ (x - p))
        for p, a in zip(points, axes)
    )


def test_segment_lengths_at_zero():
    """The shoulder-to-elbow length is NOT the same on both sides: joint4's
    y-offset (+0.0315m) is identical in sign on both sides of the URDF
    (confirmed directly against the generated URDF, not mirrored like most
    of the geometry - this is the "elbow asymmetry" the module docstring
    and MIRROR_SIGN both warn about), so it adds to the reach on one side
    and subtracts on the other. The forearm length is unaffected (elbow to
    wrist doesn't involve joint4's offset)."""
    frames_left = rm.forward_kinematics("left", np.zeros(7))
    frames_right = rm.forward_kinematics("right", np.zeros(7))
    upper_left = np.linalg.norm(frames_left.origins[3] - rm.shoulder_reference("left"))
    upper_right = np.linalg.norm(frames_right.origins[3] - rm.shoulder_reference("right"))
    assert upper_left == pytest.approx(0.2218, abs=1e-3)
    assert upper_right == pytest.approx(0.2383, abs=1e-3)
    for side, frames in (("left", frames_left), ("right", frames_right)):
        forearm = np.linalg.norm(frames.origins[4] - frames.origins[3])
        assert forearm == pytest.approx(0.1006, abs=1e-3), side


def test_elbow_angle_never_reaches_zero_or_pi():
    """The robot arm can never be perfectly straight or perfectly folded -
    theta(q4) stays in an interior range. See canonical_forearm_vector."""
    upper = rm.canonical_upper_arm_vector("left")
    upper_hat = upper / np.linalg.norm(upper)
    lo, hi = rm.LIMITS["left"][3]
    thetas = []
    for q4 in np.linspace(lo, hi, 25):
        forearm = rm.canonical_forearm_vector("left", q4)
        forearm_hat = forearm / np.linalg.norm(forearm)
        thetas.append(np.arccos(np.clip(upper_hat @ forearm_hat, -1, 1)))
    thetas = np.array(thetas)
    assert thetas.min() > 0.05
    assert thetas.max() < np.pi - 0.05
    # Monotone in q4 over the REAL mechanical range (q4 >= 0): there is a
    # genuine, tiny (<0.02 rad) non-monotonic dip confined to joint4's
    # software safety margin below its mechanical stop (q4 in [-0.02, ~0],
    # see joint_limits.yaml's comment on that margin and
    # retarget.py::_ElbowMap, which enforces monotonicity there for
    # interpolation purposes) - a coarse 25-point sweep like this one is
    # too sparse to see that narrow dip, which is why this passed even
    # before the dip was discovered with a finer grid; do not read that as
    # this test having missed a real bug.
    real_range = np.linspace(lo, hi, 25) >= 0
    assert np.all(np.diff(thetas[real_range]) > 0)


def test_capsule_distance_symmetric_and_zero_at_overlap():
    a = rm.Capsule("a", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.1]), 0.02)
    b = rm.Capsule("b", np.array([0.0, 0.0, 0.05]), np.array([0.1, 0.0, 0.05]), 0.02)
    d_ab = rm.capsule_distance(a, b)
    d_ba = rm.capsule_distance(b, a)
    assert d_ab == pytest.approx(d_ba, abs=1e-9)
    assert d_ab < 0.0  # these two overlap by construction
    assert rm.capsules_collide(a, b)

    far = rm.Capsule("far", np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.1]), 0.02)
    assert not rm.capsules_collide(a, far)


def test_arm_capsules_use_own_side_fk():
    """Capsules must come from each side's own FK, never mirrored FK - the
    elbow offset is not symmetric between sides (see MIRROR_SIGN docstring)."""
    q = np.array([0.1, -0.3, 0.2, 1.0, 0.0, 0.0, 0.0])
    left_caps = rm.arm_capsules("left", q)
    right_caps = rm.arm_capsules("right", q)
    # Same q, opposite sides: elbow positions must differ (not mirror-equal)
    # because of the known 31.5mm asymmetry - this would only be near-zero
    # by coincidence, so a loose bound is enough to catch someone
    # accidentally mirroring instead of recomputing per-side FK.
    left_elbow = left_caps[0].p1
    right_elbow = right_caps[0].p1
    mirrored = left_elbow.copy()
    mirrored[1] *= -1
    assert np.abs(mirrored - right_elbow).max() > 1e-6
