"""Frozen OpenArm v10 kinematic model: per-side joint geometry, per-side
joint limits, forward kinematics, and a capsule collision model.

Every number here was independently verified two ways before being frozen:
  1. Regenerated from the live robot's generated URDF
     (src/openarm_description/assets/robot/openarm_v1.0/urdf/v10.urdf,
     currently a single-arm demo file - the bimanual per-side numbers here
     were verified against the robot's own generated bimanual URDF, see
     below) via a rotation-matrix FK
     written from scratch for this package.
  2. Cross-checked against PyKDL's ChainFkSolverPos_recursive, built from the
     same URDF's joint origins/axes, run on the robot
     (tools/gen_golden_fk.py -> test/fixtures/fk_golden.json).
Both agree to double-precision (see test_robot_model.py::test_fk_against_golden).

Two facts drove the design and are worth restating because they are easy to
get wrong from the raw config alone:

  - Per-side joint limits are NOT symmetric and NOT the raw values in
    assets/robot/openarm_v1.0/config/arm/v10/joint_limits.yaml. openarm_macro.xacro's openarm-limits
    macro reflects and offsets j1/j2 per side and swaps bounds when the
    product inverts (see derive_limits below, which reimplements that
    macro exactly and is unit-tested against the checked-in yaml).
  - The generated URDF is stale for j4's lower bound (0.0 instead of the
    yaml's -0.02 safety margin - see the comment in joint_limits.yaml). This
    module uses the authoritative yaml value; derive_limits takes the raw
    per-joint bounds as a parameter precisely so a live /robot_description
    can be substituted at the call site (in teleop_node, not here - this
    module does no I/O) without ever falling back to the stale URDF number.

IMPORTANT CORRECTION vs. earlier design notes: this is *not* a zero-offset
spherical shoulder/wrist. j1/j2/j3's axes do NOT intersect at a common point
in general (residual ~5cm over random q - verified, see
test_robot_model.py::test_shoulder_is_not_exactly_concurrent) - there is a
genuine ~30mm mechanical offset baked into the link geometry. What IS exact,
to double precision, is the ROTATION identity used by retarget.py's closed
form:

    R_shoulder(q1,q2,q3) == Rz(q1) @ Rx(SHOULDER_REFLECT[side]*pi/2 - q2) @ Rz(q3)

(see test_robot_model.py::test_zxz_shoulder_identity). retarget.py uses this
exact rotation to build a closed-form SEED for q1,q2,q3 from a target upper-
arm direction, then runs a few Gauss-Newton iterations against this module's
real forward_kinematics (whose position IS exact, verified against PyKDL) to
correct for the offset. This is the standard, honest way to handle a
"near-spherical" mechanism: exact orientation algebra for the seed and
branch selection, numerical refinement against the true FK for the residual
position error. Never assume shoulder_reference()/wrist_reference() are true
concurrency points for position purposes - they are convenient fixed
reference frames (frame origins at q=0), nothing more.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIDES = ("left", "right")
N_JOINTS = 7

# ---------------------------------------------------------------------------
# Geometry: per-joint (origin_rpy, origin_xyz, axis), in the parent joint's
# frame, exactly as the URDF/xacro emits them. Frame 0 is the arm's own base
# link (openarm_<side>_link0); joint i's origin is applied before its own
# axis rotation, matching URDF semantics.
# ---------------------------------------------------------------------------

_HALF_PI = np.pi / 2.0

# (rpy, xyz, axis) per joint, left arm.
_GEOMETRY_LEFT = (
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0625), (0.0, 0.0, 1.0)),
    ((-_HALF_PI, 0.0, 0.0), (-0.0301, 0.0, 0.06), (-1.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.0301, 0.0, 0.06625), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 0.0), (0.0, 0.0315, 0.15375), (0.0, 1.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.0, -0.0315, 0.0955), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 0.0), (0.0375, 0.0, 0.1205), (1.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-0.0375, 0.0, 0.0), (0.0, -1.0, 0.0)),
)

# Right arm: identical except j2's origin roll flips sign and j7's axis
# flips sign (both are the `reflect` factor baked directly into the URDF).
_GEOMETRY_RIGHT = (
    _GEOMETRY_LEFT[0],
    ((_HALF_PI, 0.0, 0.0), (-0.0301, 0.0, 0.06), (-1.0, 0.0, 0.0)),
    _GEOMETRY_LEFT[2],
    _GEOMETRY_LEFT[3],
    _GEOMETRY_LEFT[4],
    _GEOMETRY_LEFT[5],
    ((0.0, 0.0, 0.0), (-0.0375, 0.0, 0.0), (0.0, 1.0, 0.0)),
)

GEOMETRY = {"left": _GEOMETRY_LEFT, "right": _GEOMETRY_RIGHT}

# `reflect` sign used by the shoulder ZXZ identity in retarget.py:
#   R_shoulder(q1,q2,q3) == Rz(q1) @ Rx(SHOULDER_REFLECT[side]*pi/2 - q2) @ Rz(q3)
SHOULDER_REFLECT = {"left": -1.0, "right": 1.0}

# Sagittal mirror sign vector - EXACT for the shoulder+elbow (indices 0-3)
# only: FK("right", MIRROR_SIGN[:4] * q_left[:4]) == mirror_y(FK("left", q_left))
# with q5=q6=q7=0 matches to double precision (see test_mirror.py). It does
# NOT extend to the wrist (indices 4-6): verified empirically that no single
# per-joint sign choice makes FK("right", MIRROR_SIGN * q_left) match
# mirror_y(FK("left", q_left)) once q5 is nonzero (error grows with q5,
# tens of mm) - the wrist frame's own accumulated orientation differs
# between sides in a way a diagonal sign matrix cannot undo.
# Consequence for retarget.py/pipeline.py: "mirror mode" is NOT implemented
# as a q-space transform. It is implemented by choosing which human arm's
# landmarks (left or right) feed which robot arm's independent closed-form
# solver - each side already retargets from body-frame direction vectors
# using its own SHOULDER_REFLECT/limits, so no cross-side sign trick is
# needed or correct. Keep MIRROR_SIGN only for what is verified exact:
# cross-checking the shoulder+elbow FK and mapping joint LIMITS between
# sides (which the xacro's own reflect/offset construction does exactly,
# independent of this residual, see derive_limits).
MIRROR_SIGN = np.array([-1.0, -1.0, -1.0, 1.0])  # applies to q[0:4] only

# ---------------------------------------------------------------------------
# Joint limits: raw (un-reflected) values from
# assets/robot/openarm_v1.0/config/arm/v10/joint_limits.yaml, and the exact per-joint (reflect, offset)
# rule openarm_macro.xacro's openarm-limits applies before swapping bounds
# if the product inverts. j4's lower bound uses the yaml's authoritative
# -0.02 safety margin, not the stale 0.0 baked into the generated URDF.
# ---------------------------------------------------------------------------

RAW_LIMITS = (
    (-1.396263, 3.490659),   # joint1
    (-1.745329, 1.745329),   # joint2
    (-1.570796, 1.570796),   # joint3
    (-0.02, 2.443461),       # joint4 (authoritative; URDF says 0.0 - stale)
    (-1.570796, 1.570796),   # joint5
    (-0.785398, 0.785398),   # joint6
    (-1.570796, 1.570796),   # joint7
)


def _limit_rule(side: str, joint_index: int) -> tuple[float, float]:
    """(reflect, offset) for joint `joint_index` (0-based) on `side`, exactly
    matching the xacro:openarm-limits invocations in openarm_arm.xacro."""
    if joint_index == 0:  # joint1
        return 1.0, (-2.094396 if side == "left" else 0.0)
    if joint_index == 1:  # joint2
        reflect = -1.0 if side == "left" else 1.0
        offset = -np.pi / 2 if side == "left" else np.pi / 2
        return reflect, offset
    return 1.0, 0.0  # joint3, joint5, joint6, joint7 pass through unchanged
    # (joint4's macro call also has no reflect/offset - handled by the
    # caller passing raw_limits with the authoritative lower bound already.)


def derive_limits(raw_limits=RAW_LIMITS) -> dict:
    """Reimplementation of xacro:openarm-limits, in Python, over all 7
    joints and both sides. Pure function of raw_limits so a live
    /robot_description can be substituted by the caller (teleop_node) without
    this module doing any file or ROS I/O. Returns {side: (7,2) array of
    [lower, upper]}."""
    out = {}
    for side in SIDES:
        bounds = np.empty((N_JOINTS, 2), dtype=np.float64)
        for i, (lower, upper) in enumerate(raw_limits):
            reflect, offset = _limit_rule(side, i)
            raw_lower = lower * reflect + offset
            raw_upper = upper * reflect + offset
            bounds[i] = (raw_lower, raw_upper) if raw_lower < raw_upper else (raw_upper, raw_lower)
        out[side] = bounds
    return out


LIMITS = derive_limits()  # frozen default, {side: (7,2) array}


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_rotation(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


@dataclass(frozen=True)
class FkFrames:
    """Per-joint frame origins/orientations for one arm, in the arm base
    frame. origins[i] / axes_world[i] are joint i's rotation-axis point
    and world-frame axis direction *before* joint i's own rotation is
    applied (i.e. the physical hinge line) - what capsule/collision geometry
    and the shoulder/wrist concurrency tests use. ee_position /
    ee_rotation are the final tool-frame pose after all 7 joints."""

    origins: np.ndarray  # (7, 3)
    axes_world: np.ndarray  # (7, 3), unit vectors
    ee_position: np.ndarray  # (3,)
    ee_rotation: np.ndarray  # (3, 3)


def forward_kinematics(side: str, q) -> FkFrames:
    """FK for one arm, in the arm's own base frame (openarm_<side>_link0).
    q is 7 joint angles in radians, in URDF joint order (joint1..joint7).

    Per-joint relative transform, verified against PyKDL segment-by-segment
    output (see tools/gen_golden_fk.py and test_robot_model.py):

        R_rel = Rot(axis_i, q_i) @ Rrpy_i
        p_rel = Rot(axis_i, q_i) @ xyz_i

    i.e. the joint's own variable rotation is applied FIRST, and the fixed
    <origin> (rpy, xyz) SECOND - matching KDL's Segment::pose(q) =
    joint.pose(q) * f_tip, where f_tip is the fixed origin frame. This is the
    opposite order from the naive "translate-then-rotate" reading of a URDF
    <origin> element, and it is easy to get backwards: with this robot's
    joint4 (axis=(0,1,0), xyz=(0,0.0315,0.15375), rpy=0) the two conventions
    diverge by tens of centimeters at the end effector, since xyz is not
    parallel to axis there. It happens to be unobservable on joint1 (xyz
    parallel to axis) and on joint2 (axis happens to be parallel to that
    joint's own rpy rotation axis, so the two Rrpy/Raxis compositions
    commute) - both of which matched under either convention during initial
    development, which is what made the joint3+ mismatch a genuine surprise
    rather than an obvious typo. Do not "simplify" this back to
    `p = p + r @ xyz; r = r @ rpy_matrix(*rpy); r = r @ axis_rotation(axis, qi)`
    without re-running test_fk_against_golden - that is exactly the bug this
    comment documents.

    `origins[i]` (the physical hinge point used for capsules/concurrency) is
    still q_i-INDEPENDENT, since Rot(axis_i, 0) = I: it is recorded using the
    frame accumulated through joint i-1 plus the unrotated xyz_i, before
    folding in joint i's own rotation.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (7,):
        raise ValueError(f"q must be shape (7,), got {q.shape}")
    geometry = GEOMETRY[side]

    r = np.eye(3)
    p = np.zeros(3)
    origins = np.empty((7, 3))
    axes_world = np.empty((7, 3))
    for i, ((rpy, xyz, axis), qi) in enumerate(zip(geometry, q)):
        xyz_arr = np.asarray(xyz, dtype=np.float64)
        axis_arr = np.asarray(axis, dtype=np.float64)
        origins[i] = p + r @ xyz_arr
        axes_world[i] = r @ (axis_arr / np.linalg.norm(axis_arr))
        raxis = _axis_rotation(axis, qi)
        p = p + r @ (raxis @ xyz_arr)
        r = r @ raxis @ _rpy_matrix(*rpy)
    return FkFrames(origins=origins, axes_world=axes_world, ee_position=p.copy(), ee_rotation=r.copy())


def ee_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (x, y, z, w) quaternion, matching PyKDL's
    Rotation.GetQuaternion() convention used in the golden fixtures."""
    m = rotation
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def shoulder_reference(side: str) -> np.ndarray:
    """Joint1's own origin, in the arm base frame - the fixed point
    retarget.py measures the upper-arm direction from. NOT a true
    concurrency point for j1/j2/j3 (see module docstring); it is exact and
    q-independent because joint1's own origin never moves."""
    return np.asarray(GEOMETRY[side][0][1], dtype=np.float64)


def wrist_reference(side: str) -> np.ndarray:
    """Joint5's hinge point at q1=q2=q3=q4=0, in the arm base frame - the
    fixed reference retarget.py measures the forearm direction from. Like
    shoulder_reference, this is a convenient fixed point, not a true
    concurrency point for j5/j6/j7."""
    frames = forward_kinematics(side, np.zeros(7))
    return frames.origins[4]


def canonical_upper_arm_vector(side: str) -> np.ndarray:
    """The upper-arm vector (shoulder_reference -> elbow) at q1=q2=q3=q4=0,
    in the arm base frame. retarget.py's closed-form seed approximates the
    upper-arm direction for arbitrary (q1,q2,q3) as
    R_shoulder(q1,q2,q3) @ canonical_upper_arm_vector(side) - exact for the
    ROTATION (confirmed to 2e-16, see module docstring), approximate for
    position (the real elbow also picks up a small translation-coupling
    term the closed form ignores; retarget.py's DLS refinement corrects for
    it against the true forward_kinematics)."""
    frames = forward_kinematics(side, np.zeros(7))
    return frames.origins[3] - shoulder_reference(side)


def canonical_forearm_vector(side: str, q4: float) -> np.ndarray:
    """The forearm vector (elbow -> wrist) at q1=q2=q3=0 and the given q4,
    in the frame established by q1=q2=q3=0 (i.e. Rx(SHOULDER_REFLECT*pi/2)).
    Used to build retarget.py's theta(q4) elbow-angle map: the angle between
    canonical_upper_arm_vector and this vector, as a function of q4 alone."""
    q = np.zeros(7)
    q[3] = q4
    frames = forward_kinematics(side, q)
    return frames.origins[4] - frames.origins[3]


def elbow_point(side: str, q4: float) -> np.ndarray:
    """Position of the physical elbow (joint4's own origin) for a given q4,
    holding q1=q2=q3=0. Used to build the theta(q4) elbow map in retarget.py."""
    q = np.zeros(7)
    q[3] = q4
    frames = forward_kinematics(side, q)
    return frames.origins[3]


# ---------------------------------------------------------------------------
# Capsule collision model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capsule:
    """A swept-sphere segment between two 3D points, in a common world
    frame, with `radius` already inflated for the safety margins documented
    in the plan (rate-limiter overshoot + one control period of travel +,
    for the elbow, the 31.5 mm inter-arm asymmetry)."""

    name: str
    p0: np.ndarray
    p1: np.ndarray
    radius: float


# Base radii from the arm's collision mesh bounding cylinders (measured on
# the STLs), inflated per plan Part "Self-collision without MoveIt": +6mm for
# one 100Hz control-period of travel at the teleop-stage rate-limiter's
# velocity cap, further +31.5mm on segments touching the elbow frame to cover
# the elbow's known left/right asymmetry (see MIRROR_SIGN docstring above).
UPPER_ARM_RADIUS = 0.055 + 0.006
FOREARM_RADIUS = 0.045 + 0.006 + 0.0315
# Sized relative to this arm's own (short) reach, not a human torso: total
# shoulder-to-wrist reach here is only ~0.32m (0.222 upper + 0.10 forearm -
# see robot_model's docstring / test_robot_model.py's segment-length
# tests), so a human-scale ~0.20m torso half-width would consume most of
# the arm's reach and falsely flag the home pose as colliding (caught by
# test_check_self_collision_no_collision_at_home). Kept in roughly the
# same proportion to arm length as a human torso is to a human arm.
TORSO_HALF_WIDTH = 0.07
TORSO_HALF_DEPTH = 0.06


def arm_capsules(side: str, q) -> list:
    """Two capsules per arm: shoulder->elbow and elbow->wrist. Always built
    from this side's OWN forward kinematics (never mirrored FK - the elbow
    offset is not symmetric between sides, see MIRROR_SIGN's docstring)."""
    frames = forward_kinematics(side, q)
    shoulder = shoulder_reference(side)
    elbow = frames.origins[3]
    wrist = frames.origins[4]
    return [
        Capsule(f"{side}_upper_arm", shoulder, elbow, UPPER_ARM_RADIUS),
        Capsule(f"{side}_forearm", elbow, wrist, FOREARM_RADIUS),
    ]


def torso_capsule() -> Capsule:
    """A single vertical capsule approximating the torso, in the robot base
    frame, spanning from the hip line to the shoulder line. Conservative
    substitute for a box: a capsule whose radius covers the torso's
    half-width.

    Both arms' own base frames (link0) coincide at the same physical
    mounting point (openarm_robot.xacro connects both to
    openarm_body_link0 at xyz="0 0 0"), which sits exactly on this
    capsule's central axis - i.e. each arm's shoulder mount is, by
    construction, always "at" the torso capsule regardless of q. The
    caller (validate.check_self_collision) must trim the proximal end of
    the upper-arm capsule away from the shoulder before checking against
    this, or every pose - including the home pose - reads as colliding."""
    p0 = np.array([0.0, 0.0, -0.15])
    p1 = np.array([0.0, 0.0, 0.25])
    return Capsule("torso", p0, p1, max(TORSO_HALF_WIDTH, TORSO_HALF_DEPTH))


def _closest_points_segments(p0, p1, q0, q1):
    """Closest points between two 3D segments (p0,p1) and (q0,q1); returns
    (dist, point_on_pq, point_on_qr). Standard segment-segment distance."""
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = d1 @ d1
    e = d2 @ d2
    f = d2 @ r
    if a <= 1e-12 and e <= 1e-12:
        return np.linalg.norm(r), p0, q0
    if a <= 1e-12:
        t = np.clip(f / e, 0.0, 1.0)
        s = 0.0
    else:
        c = d1 @ r
        if e <= 1e-12:
            s = np.clip(-c / a, 0.0, 1.0)
            t = 0.0
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    cp1 = p0 + s * d1
    cp2 = q0 + t * d2
    return np.linalg.norm(cp1 - cp2), cp1, cp2


def capsule_distance(a: Capsule, b: Capsule) -> float:
    dist, _, _ = _closest_points_segments(a.p0, a.p1, b.p0, b.p1)
    return dist - a.radius - b.radius


def capsules_collide(a: Capsule, b: Capsule) -> bool:
    return capsule_distance(a, b) < 0.0
