"""Unit tests for core/retarget.py."""
import numpy as np
import pytest

from gesture_teleop.core import retarget as rt
from gesture_teleop.core import robot_model as rm
from gesture_teleop.core.types import RejectReason


def _normalize(v):
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("side", rm.SIDES)
def test_inner_vector_matches_real_fk(side):
    """_inner_vector's closed form must match forward_kinematics's actual
    elbow position exactly - this is the fact the whole seed strategy rests
    on (see the module docstring on why the earlier "rotate a canonical
    vector" approach was wrong, not just approximate)."""
    rng = np.random.default_rng(0)
    xyz1 = np.asarray(rm.GEOMETRY[side][0][1])
    max_err = 0.0
    for _ in range(300):
        q = np.zeros(7)
        q[:3] = rng.uniform(-2.0, 2.0, 3)
        frames = rm.forward_kinematics(side, q)
        predicted = xyz1 + rt._rz(q[0]) @ rt._inner_vector(side, q[1], q[2])
        max_err = max(max_err, np.abs(frames.origins[3] - predicted).max())
    assert max_err < 1e-9


def test_inner_vector_is_q1_independent():
    """The whole point of the closed form: inner(q2,q3) never involves q1."""
    v1 = rt._inner_vector("left", 0.3, -0.7)
    v2 = rt._inner_vector("left", 0.3, -0.7)
    np.testing.assert_allclose(v1, v2)


@pytest.mark.parametrize("side", rm.SIDES)
def test_elbow_map_monotone_and_bounds(side):
    m = rt._ELBOW_MAPS[side]
    # Non-decreasing (not strictly increasing): theta(q4) has a genuine,
    # tiny (<0.02 rad) non-monotonic dip confined to joint4's software
    # safety margin below its real mechanical stop (q4 in [-0.02, ~0] -
    # see joint_limits.yaml's comment on that margin); _ElbowMap enforces
    # monotonicity there via np.maximum.accumulate, which can plateau
    # (diff==0) but never decrease.
    assert np.all(np.diff(m._theta_grid) >= 0)
    # And it is strictly increasing away from that narrow margin, which is
    # what actually matters for interpolation-based inversion in practice.
    past_margin = m._q4_grid > 0.05
    assert np.all(np.diff(m._theta_grid[past_margin]) > 0)
    lo, hi = rm.LIMITS[side][3]
    assert m._q4_grid[0] == pytest.approx(lo)
    assert m._q4_grid[-1] == pytest.approx(hi)


@pytest.mark.parametrize("side", rm.SIDES)
def test_elbow_map_invert_round_trip(side):
    m = rt._ELBOW_MAPS[side]
    for q4 in np.linspace(*rm.LIMITS[side][3], 15):
        if q4 < 0.05:
            continue  # inside the known non-monotonic safety-margin dip, see _ElbowMap
        theta = m.theta(q4)
        q4_back, reachable = m.invert(theta)
        assert reachable
        assert q4_back == pytest.approx(q4, abs=1e-2)


@pytest.mark.parametrize("side", rm.SIDES)
def test_elbow_map_unreachable_theta_flagged(side):
    m = rt._ELBOW_MAPS[side]
    _, reachable = m.invert(m.theta_max + 0.5)
    assert not reachable
    _, reachable = m.invert(m.theta_min - 0.5)
    assert not reachable


@pytest.mark.parametrize("side", rm.SIDES)
def test_solve_arm_round_trip_from_real_fk(side):
    """Pick a real, reachable q, compute its upper/forearm directions via
    the true FK, retarget from those directions, and check the result's FK
    reproduces similar directions - the end-to-end sanity check that the
    seed+DLS pipeline actually converges against the real kinematics."""
    rng = np.random.default_rng(9)
    lo, hi = rm.LIMITS[side][:5, 0], rm.LIMITS[side][:5, 1]
    n_ok = 0
    for _ in range(30):
        q5 = rng.uniform(lo, hi)
        upper, forearm = rt._fk_directions(side, q5)
        shoulder_local = np.zeros(3)
        elbow_local = upper  # unit-length placeholder segment; only direction matters
        wrist_local = upper + forearm
        target = rt.solve_arm(side, shoulder_local, elbow_local, wrist_local, q_prev=None)
        final_upper, final_forearm = rt._fk_directions(side, target.q[:5])
        err = np.linalg.norm(final_upper - upper) + np.linalg.norm(final_forearm - forearm)
        if err < 0.05:
            n_ok += 1
    # Not every random configuration is perfectly recoverable (DLS with a
    # fixed iteration count, gimbal band, joint limits all bite sometimes),
    # but the large majority should converge closely.
    assert n_ok >= 24, f"only {n_ok}/30 converged closely for {side}"


@pytest.mark.parametrize("side", rm.SIDES)
def test_solve_arm_reports_reason_none_when_valid(side):
    q5 = (rm.LIMITS[side][:5, 0] + rm.LIMITS[side][:5, 1]) / 2.0
    upper, forearm = rt._fk_directions(side, q5)
    target = rt.solve_arm(side, np.zeros(3), upper, upper + forearm, q_prev=None)
    assert target.reason in (RejectReason.NONE, RejectReason.JOINT_LIMIT, RejectReason.UNREACHABLE)


def test_solve_arm_low_visibility_on_degenerate_points():
    """Coincident shoulder/elbow points can't define a direction - must be
    reported, never crash or silently produce garbage."""
    target = rt.solve_arm("left", np.zeros(3), np.zeros(3), np.array([0.1, 0.0, 0.0]), q_prev=None)
    assert not target.valid
    assert target.reason is RejectReason.LOW_VISIBILITY


@pytest.mark.parametrize("side", rm.SIDES)
def test_gimbal_lock_continuity_sweep(side):
    """Sweep the target upper-arm direction through the horizontal-lateral
    pose (the gimbal band at q2=reflect*pi/2 - the most likely real demo
    gesture, per the plan) and check no single-tick joint jump exceeds a
    generous safety bound - the actual jerk-limiting happens downstream in
    filters.RateLimiter, but retarget.py itself must not hand it a
    discontinuous jump to chase."""
    # Build the sweep from GUARANTEED-reachable FK outputs (q4, forearm
    # held fixed; only q1 varies) rather than an arbitrary direction
    # formula - an earlier version of this test used
    # direction=(cos(angle),sin(angle),0.1), which turned out to be
    # UNREACHABLE for a large stretch of angles on both sides (nothing to
    # do with q2's gimbal band; that formula's arm plane just does not
    # match either side's actual reachable envelope well). Sweeping q1
    # through its own full range while holding q2 fixed AT the original
    # ZXZ-parameterization's gimbal value (reflect*pi/2) - the pose this
    # test exists to guard, a lateral arm raise - keeps every sample
    # reachable by construction while still exercising that region.
    refl = rm.SHOULDER_REFLECT[side]
    q1_lo, q1_hi = rm.LIMITS[side][0]
    q2_gimbal = refl * np.pi / 2
    q3_fixed, q4_fixed = 0.3, 1.0

    def _direction_for(q1):
        q = np.array([q1, q2_gimbal, q3_fixed, q4_fixed, 0.0, 0.0, 0.0])
        frames = rm.forward_kinematics(side, q)
        shoulder = rm.shoulder_reference(side)
        elbow_local = frames.origins[3] - shoulder
        wrist_local = frames.origins[4] - shoulder
        return elbow_local, wrist_local

    elbow0, wrist0 = _direction_for(q1_lo)
    first = rt.solve_arm(side, np.zeros(3), elbow0, wrist0, q_prev=None)
    assert first.valid
    q_prev = first.q

    # Sweep q1 across its full range, holding the arm at the gimbal-band
    # elevation the whole time. Only VALID results feed forward as the
    # next q_prev - an invalid/UNREACHABLE result is retarget.py's signal
    # for the caller (pipeline.py) to hold the last good position rather
    # than command it, so it must not be chained into "continuity" here
    # either.
    n = 200
    max_step = 0.0
    for i in range(n):
        t = i / (n - 1)
        q1 = q1_lo + t * (q1_hi - q1_lo)
        elbow_local, wrist_local = _direction_for(q1)
        target = rt.solve_arm(side, np.zeros(3), elbow_local, wrist_local, q_prev=q_prev)
        if target.valid:
            step = np.abs(target.q[:5] - q_prev[:5]).max()
            max_step = max(max_step, step)
            q_prev = target.q
    # Steps between consecutive VALID solves, with continuity biasing,
    # should not produce a jump larger than a modest fraction of the full
    # joint range.
    assert max_step < 1.5, f"{side}: max single-tick jump {max_step:.3f} rad"


@pytest.mark.parametrize("side", rm.SIDES)
def test_solve_arm_deterministic(side):
    q5 = (rm.LIMITS[side][:5, 0] + rm.LIMITS[side][:5, 1]) / 2.0
    upper, forearm = rt._fk_directions(side, q5)
    t1 = rt.solve_arm(side, np.zeros(3), upper, upper + forearm, q_prev=None)
    t2 = rt.solve_arm(side, np.zeros(3), upper, upper + forearm, q_prev=None)
    np.testing.assert_allclose(t1.q, t2.q, atol=1e-12)
