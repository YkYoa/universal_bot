"""Human-arm-direction to OpenArm-joint-angle retargeting.

Pipeline per side, per frame:
  1. Closed-form seed for q1,q2,q3 matching the target upper-arm direction
     EXACTLY (to numerical precision) against the real forward_kinematics -
     see _inner_vector's docstring for the derivation. This replaced an
     earlier version of this module that rotated a single "canonical
     upper-arm vector" by the shoulder's ZXZ rotation matrix; that looked
     plausible (the ZXZ rotation identity itself is exact, see
     robot_model's module docstring) but was structurally wrong for
     POSITION, off by 60-90 degrees on typical poses, not a small
     approximation error - because each of joint1/2/3's own translations
     rotate with THAT joint's own angle under this robot's kinematics
     convention (robot_model.forward_kinematics's docstring), not with the
     angles of joints downstream of it. _inner_vector accounts for this
     exactly; only q1's SIGN convention (a global rotation about the
     shared vertical axis) is trivial, so the seed reduces to a small 2D
     solve for (q2,q3) plus a closed-form q1.
  2. q4 from a per-side theta(q4) elbow-angle lookup table, inverted by
     interpolation.
  3. A fixed damped Gauss-Newton refinement of (q1..q5) against the REAL
     forward_kinematics, correcting the position seed's remaining
     approximation (representative arm length only, not per-pose exact -
     see _inner_vector) and choosing q5 (which, under this robot's
     kinematics convention, also affects wrist position - see
     robot_model.forward_kinematics's docstring - not a pure roll).

q6=q7=0 always (Stage 1 scope, see plan: the wrist is over-constrained -
j6's +-45 deg limit means most hand orientations are unreachable anyway,
and MediaPipe's wrist/hand landmarks are its noisiest).
"""
from __future__ import annotations

import numpy as np

from gesture_teleop.core import robot_model as rm
from gesture_teleop.core.types import ArmTarget, RejectReason

BASE_DAMPING = 1e-3
GIMBAL_DAMPING_GAIN = 0.05
DLS_ITERATIONS = 8
FD_EPS = 1e-5
_ELBOW_TABLE_POINTS = 200
_SEED_2D_GRID_POINTS = 4  # per axis -> 16 starting points per side
# |horizontal component| below this: the target points nearly straight up
# or down, where q1's atan2 is ill-conditioned (this parameterization's own
# gimbal case - see module docstring on why it differs from the ZXZ one an
# earlier version of this module guarded against).
STRAIGHT_UP_DOWN_THRESHOLD = 0.02


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("cannot normalize a near-zero vector")
    return v / n


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _inner_vector(side: str, q2: float, q3: float) -> np.ndarray:
    """Exact closed form for elbow(q1,q2,q3) = xyz1 + Rz(q1) @ inner(q2,q3),
    derived directly from robot_model.forward_kinematics's per-joint
    relative transform (R_rel = Rot(axis,q) @ Rrpy, p_rel = Rot(axis,q) @
    xyz - see its docstring) by expanding the chain through joints 1-3 and
    collecting every term that carries a factor of Rz(q1) outside it (every
    term does, since q1's own axis is the shared z-axis and nothing
    upstream of it exists to entangle it with q2/q3):

        inner(q2,q3) = Rx(-q2) @ xyz2 + Rx(reflect*pi/2 - q2) @ Rz(q3) @ (xyz3 + xyz4)

    Verified against forward_kinematics to 1e-16 over 300 random
    (q1,q2,q3) - see test_retarget.py::test_inner_vector_matches_fk. This
    is q1-INDEPENDENT by construction, which is exactly what makes the
    seed solvable as "2D position match for (q2,q3), then 1D angle match
    for q1" instead of a joint 3D search.
    """
    refl = rm.SHOULDER_REFLECT[side]
    xyz2 = np.asarray(rm.GEOMETRY[side][1][1], dtype=np.float64)
    xyz34 = np.asarray(rm.GEOMETRY[side][2][1], dtype=np.float64) + np.asarray(
        rm.GEOMETRY[side][3][1], dtype=np.float64
    )
    return _rx(-q2) @ xyz2 + _rx(refl * np.pi / 2 - q2) @ _rz(q3) @ xyz34


_NEWTON_STEP_CAP = 0.5  # radians per iteration - prevents the unbounded
                         # Gauss-Newton step from running away to a
                         # numerically-huge (though periodically-equivalent
                         # in direction) angle when the local Jacobian is
                         # near-singular; caught by test_solve_arm_round_trip
                         # producing seeds like q2=-42 rad before this cap.


def _direction_z_hmag(v: np.ndarray):
    """z-component and horizontal magnitude of v's UNIT direction (not v
    itself) - see _solve_q2_q3's docstring on why matching the direction,
    not a length-scaled position, is the correct target."""
    v_hat = v / np.linalg.norm(v)
    return float(v_hat[2]), float(np.hypot(v_hat[0], v_hat[1]))


def _solve_q2_q3(side: str, target_z: float, target_hmag: float, q2_0: float, q3_0: float,
                  side_limits, iters: int = 20):
    """2D damped Newton solve for (q2,q3) matching normalize(inner(q2,q3))'s
    z-component and horizontal magnitude to the given targets - see
    _inner_vector and _direction_z_hmag.

    Matching the DIRECTION of inner(q2,q3), not a length-scaled absolute
    position, matters: xyz1 (the shoulder offset) cancels out of the
    upper-arm-direction equation entirely (upper_dir = normalize(elbow -
    shoulder_reference) = normalize(Rz(q1) @ inner(q2,q3)) =
    Rz(q1) @ normalize(inner(q2,q3)), since Rz preserves norm and
    shoulder_reference IS xyz1) - so there is no length to assume in the
    first place. An earlier version of this function scaled the target by
    a fixed representative length (this side's canonical upper-arm length
    at q=0) and subtracted xyz1, which is only valid at exactly that one
    length; |inner(q2,q3)| genuinely varies with q2,q3 (the shoulder is not
    exactly spherical, see robot_model's module docstring), so that fixed
    length was infeasible for a lot of real targets and the Newton solve
    correctly reported it could not be reached, converging to a joint-limit
    boundary instead (caught by test_solve_arm_round_trip_from_real_fk
    failing at scale on targets the FK-generated ground truth proves ARE
    reachable). Matching the direction only removes the length assumption.

    Step-capped and clipped to `side_limits` (this side's actual q2,q3
    bounds) every iteration, so a near-singular Jacobian cannot walk the
    solution out to a numerically-huge angle.
    """
    (q2_lo, q2_hi), (q3_lo, q3_hi) = side_limits
    q2 = float(np.clip(q2_0, q2_lo, q2_hi))
    q3 = float(np.clip(q3_0, q3_lo, q3_hi))
    for _ in range(iters):
        z, hmag = _direction_z_hmag(_inner_vector(side, q2, q3))
        f = np.array([z - target_z, hmag - target_hmag])
        eps = 1e-6
        z_dq2, hmag_dq2 = _direction_z_hmag(_inner_vector(side, q2 + eps, q3))
        z_dq3, hmag_dq3 = _direction_z_hmag(_inner_vector(side, q2, q3 + eps))
        j = np.array([
            [(z_dq2 - z) / eps, (z_dq3 - z) / eps],
            [(hmag_dq2 - hmag) / eps, (hmag_dq3 - hmag) / eps],
        ])
        jtj = j.T @ j + 1e-6 * np.eye(2)
        try:
            d = np.linalg.solve(jtj, -j.T @ f)
        except np.linalg.LinAlgError:
            break
        d = np.clip(d, -_NEWTON_STEP_CAP, _NEWTON_STEP_CAP)
        q2 = float(np.clip(q2 + d[0], q2_lo, q2_hi))
        q3 = float(np.clip(q3 + d[1], q3_lo, q3_hi))
    z, hmag = _direction_z_hmag(_inner_vector(side, q2, q3))
    resid = abs(z - target_z) + abs(hmag - target_hmag)
    return q2, q3, resid


def _seed_2d_starts(side: str):
    """Starting points for the 2D (q2,q3) Newton solve, spanning each
    side's ACTUAL (asymmetric - see robot_model's per-side LIMITS) joint
    ranges. A fixed small set of points near the origin (e.g. (+-1,+-1))
    once missed every solution branch entirely for the left arm, whose q2
    range is [-3.32, 0.17] - almost none of it near zero - so the Newton
    solve only ever found branches reachable from near q2=0 and never
    discovered the ones the round-trip test's randomly sampled ground
    truth actually used (see git history of this file). A grid over each
    side's own limits fixes that structurally instead of guessing more
    fixed points that happen to work for the samples seen so far."""
    q2_lo, q2_hi = rm.LIMITS[side][1]
    q3_lo, q3_hi = rm.LIMITS[side][2]
    q2s = np.linspace(q2_lo, q2_hi, _SEED_2D_GRID_POINTS)
    q3s = np.linspace(q3_lo, q3_hi, _SEED_2D_GRID_POINTS)
    return [(float(q2), float(q3)) for q2 in q2s for q3 in q3s]


def _shoulder_seed_candidates(side: str, upper_dir_target: np.ndarray, q_prev123):
    """Closed-form (q1,q2,q3) seed CANDIDATES placing the elbow along
    upper_dir_target - NOT reduced to a single best guess here. Multiple
    (q2,q3) can satisfy the same upper-arm endpoint (an elbow-swing
    redundancy, analogous to elbow-up/elbow-down in a classic
    spherical-shoulder IK), and only evaluating the FULL solve (seed ->
    elbow q4 -> DLS -> final residual against BOTH upper and forearm
    direction) per candidate reveals which branch is actually useful -
    picking by the 2D upper-arm-only residual alone let a wrong-plane
    branch through often enough to fail the round-trip test at scale (see
    git history of this file for the numbers).

    Performance-critical distinction: with q_prev123 available (every
    frame after the first, in steady-state tracking), this returns a
    SINGLE candidate seeded at q_prev's own (q2,q3) - continuity means the
    true solution is expected to be near there, and the full grid search
    below costs ~1.3s per call (16 grid points x a full DLS refine each),
    fine for a one-time engage but not for a per-frame retargeting loop.
    Only when q_prev123 is None (engage / re-engage after a track loss,
    i.e. a genuine cold start with no continuity to rely on) does this
    fall back to the full per-side grid search over _seed_2d_starts."""
    target = _normalize(upper_dir_target)
    target_z = float(target[2])
    target_hmag = float(np.hypot(target[0], target[1]))

    side_limits = (tuple(rm.LIMITS[side][1]), tuple(rm.LIMITS[side][2]))
    if q_prev123 is not None:
        starts = [(float(q_prev123[1]), float(q_prev123[2]))]
    else:
        starts = _seed_2d_starts(side)

    seeds = []
    seen = set()
    for q2_0, q3_0 in starts:
        q2, q3, _ = _solve_q2_q3(side, target_z, target_hmag, q2_0, q3_0, side_limits)
        key = (round(q2, 3), round(q3, 3))
        if key in seen:
            continue
        seen.add(key)

        v = _inner_vector(side, q2, q3)
        _, hmag_v_unit = _direction_z_hmag(v)  # unit-direction hmag - comparable to target_hmag's [0,1] scale
        gimbal_band = target_hmag < STRAIGHT_UP_DOWN_THRESHOLD or hmag_v_unit < STRAIGHT_UP_DOWN_THRESHOLD
        if gimbal_band:
            # Target (or the solved inner vector) points nearly straight up
            # or down: q1's atan2 is ill-conditioned (a tiny horizontal
            # error swings q1 by up to pi). Freeze q1 at its previous value
            # rather than let it jump - the residual is picked up by the
            # DLS refinement's continuity pull instead (plan interlock 9's
            # equivalent for this parameterization).
            q1 = float(q_prev123[0]) if q_prev123 is not None else 0.0
        else:
            q1 = float(np.arctan2(target[1], target[0]) - np.arctan2(v[1], v[0]))
        seeds.append((q1, q2, q3, gimbal_band))

    return seeds


class _ElbowMap:
    """Per-side theta(q4) lookup, built once from robot_model's canonical
    vectors, inverted by linear interpolation over a fine grid."""

    def __init__(self, side: str):
        lo, hi = rm.LIMITS[side][3]
        self._q4_grid = np.linspace(lo, hi, _ELBOW_TABLE_POINTS)
        upper = _normalize(rm.canonical_upper_arm_vector(side))
        thetas = []
        for q4 in self._q4_grid:
            forearm = _normalize(rm.canonical_forearm_vector(side, q4))
            thetas.append(np.arccos(np.clip(upper @ forearm, -1.0, 1.0)))
        thetas = np.array(thetas)
        # theta(q4) has a tiny (<0.02 rad) non-monotonic dip confined
        # entirely to q4 in [-0.02, ~0] - the software safety margin below
        # joint4's real 0.0 mechanical stop (see joint_limits.yaml's own
        # comment on that margin), not the real mechanical range. Enforce
        # monotonicity so interpolation-based inversion is well-defined;
        # this only ever nudges q4 within that same sub-degree sliver.
        self._theta_grid = np.maximum.accumulate(thetas)
        self.theta_min = float(self._theta_grid.min())
        self.theta_max = float(self._theta_grid.max())

    def theta(self, q4: float) -> float:
        return float(np.interp(q4, self._q4_grid, self._theta_grid))

    def invert(self, theta: float):
        """Returns (q4, reachable). Clamps and reports unreachable if theta
        is outside [theta_min, theta_max] - never silently clamps without
        telling the caller (plan interlock 10)."""
        reachable = self.theta_min <= theta <= self.theta_max
        clamped = float(np.clip(theta, self.theta_min, self.theta_max))
        q4 = float(np.interp(clamped, self._theta_grid, self._q4_grid))
        return q4, reachable


_ELBOW_MAPS = {side: _ElbowMap(side) for side in rm.SIDES}


def _fk_directions(side: str, q5vec: np.ndarray):
    q = np.array([q5vec[0], q5vec[1], q5vec[2], q5vec[3], q5vec[4], 0.0, 0.0])
    frames = rm.forward_kinematics(side, q)
    shoulder = rm.shoulder_reference(side)
    upper = frames.origins[3] - shoulder
    forearm = frames.origins[4] - frames.origins[3]
    return _normalize(upper), _normalize(forearm)


def _residual(side: str, q5vec: np.ndarray, upper_target: np.ndarray, forearm_target: np.ndarray) -> np.ndarray:
    upper_actual, forearm_actual = _fk_directions(side, q5vec)
    return np.concatenate([upper_actual - upper_target, forearm_actual - forearm_target])


def _jacobian(side: str, q5vec: np.ndarray, upper_target: np.ndarray, forearm_target: np.ndarray) -> np.ndarray:
    r0 = _residual(side, q5vec, upper_target, forearm_target)
    j = np.empty((6, 5))
    for i in range(5):
        dq = q5vec.copy()
        dq[i] += FD_EPS
        j[:, i] = (_residual(side, dq, upper_target, forearm_target) - r0) / FD_EPS
    return j


def _refine(side: str, q5vec: np.ndarray, upper_target: np.ndarray, forearm_target: np.ndarray,
            gimbal_band: bool, q_prev5vec) -> np.ndarray:
    lo = rm.LIMITS[side][:5, 0]
    hi = rm.LIMITS[side][:5, 1]
    damping = BASE_DAMPING + (GIMBAL_DAMPING_GAIN if gimbal_band else 0.0)
    q = q5vec.copy()
    for _ in range(DLS_ITERATIONS):
        r = _residual(side, q, upper_target, forearm_target)
        j = _jacobian(side, q, upper_target, forearm_target)
        jtj = j.T @ j + damping * np.eye(5)
        rhs = -j.T @ r
        if q_prev5vec is not None:
            # Small pull toward the previous solution, stronger in the
            # gimbal band, so a near-singular Jacobian doesn't send the
            # branch off in an arbitrary direction (plan interlock 9).
            pull = GIMBAL_DAMPING_GAIN if gimbal_band else BASE_DAMPING
            rhs = rhs + pull * (np.asarray(q_prev5vec) - q)
        try:
            dq = np.linalg.solve(jtj, rhs)
        except np.linalg.LinAlgError:
            break
        q = np.clip(q + dq, lo, hi)
    return q


def solve_arm(side: str, shoulder_local: np.ndarray, elbow_local: np.ndarray,
              wrist_local: np.ndarray, q_prev=None) -> ArmTarget:
    """Retarget one arm from human keypoints (already expressed in the
    torso/arm-base-frame convention by frames.py) to 7 joint angles.
    q_prev (7,) or None seeds continuity and gimbal-branch selection."""
    try:
        upper_target = _normalize(elbow_local - shoulder_local)
        forearm_target = _normalize(wrist_local - elbow_local)
    except ValueError:
        return ArmTarget(side=side, q=_hold(q_prev), valid=False, reason=RejectReason.LOW_VISIBILITY)

    q_prev123 = q_prev[:3] if q_prev is not None else None
    seed_candidates = _shoulder_seed_candidates(side, upper_target, q_prev123)

    elbow_map = _ELBOW_MAPS[side]
    theta_target = float(np.arccos(np.clip(upper_target @ forearm_target, -1.0, 1.0)))
    q4_seed, elbow_reachable = elbow_map.invert(theta_target)

    q5_seed = q_prev[4] if q_prev is not None else 0.0
    q_prev5vec = q_prev[:5] if q_prev is not None else None
    lo7, hi7 = rm.LIMITS[side][:, 0], rm.LIMITS[side][:, 1]

    best_refined = None
    best_gimbal_band = False
    best_error = np.inf
    for q1, q2, q3, gimbal_band in seed_candidates:
        seed_vec = np.array([q1, q2, q3, q4_seed, q5_seed])
        refined = _refine(side, seed_vec, upper_target, forearm_target, gimbal_band, q_prev5vec)
        clipped = np.clip(refined, lo7[:5], hi7[:5])
        final_upper, final_forearm = _fk_directions(side, clipped)
        error = np.linalg.norm(final_upper - upper_target) + np.linalg.norm(final_forearm - forearm_target)
        if error < best_error:
            best_error = error
            best_refined = refined
            best_gimbal_band = gimbal_band

    refined = best_refined
    gimbal_band = best_gimbal_band
    q_full = np.zeros(7)
    within_limits = bool(np.all(refined >= lo7[:5] - 1e-6) and np.all(refined <= hi7[:5] + 1e-6))
    q_full[:5] = np.clip(refined, lo7[:5], hi7[:5])

    final_upper, final_forearm = _fk_directions(side, q_full[:5])
    residual_error = np.linalg.norm(final_upper - upper_target) + np.linalg.norm(final_forearm - forearm_target)

    if not elbow_reachable:
        reason = RejectReason.UNREACHABLE
        valid = False
    elif not within_limits:
        reason = RejectReason.JOINT_LIMIT
        valid = False
    elif residual_error > 0.15:
        reason = RejectReason.UNREACHABLE
        valid = False
    else:
        reason = RejectReason.NONE
        valid = True

    return ArmTarget(
        side=side, q=q_full, valid=valid, reason=reason,
        swivel=0.0, elbow_theta=theta_target, gimbal_band=gimbal_band,
    )


def _hold(q_prev) -> np.ndarray:
    return np.zeros(7) if q_prev is None else np.asarray(q_prev, dtype=np.float64)
