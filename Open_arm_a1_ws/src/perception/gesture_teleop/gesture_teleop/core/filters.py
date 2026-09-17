"""Smoothing and safety-clamping primitives: One Euro filter (ported from the
existing, already-tested
apps/robot-app/.../perception/face/util/OneEuroFilter.java) and a
velocity/acceleration/jerk-limited rate limiter.

Filtering choice (see the plan): One Euro at both the landmark stage and the
retargeted-joint-angle stage, backstopped by RateLimiter as the hard safety
clamp. Not a Kalman filter - a KF cannot GUARANTEE a velocity bound (a clamp
can, and that is a safety requirement here, not a smoothness preference),
and MediaPipe's landmark noise is dropout-prone and non-Gaussian (including
the depth-flip case in validate.py), so a KF's optimality assumption does
not hold and its innovation term would happily accelerate into a flip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _alpha(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Direct port of OneEuroFilter.java (Casiez et al., CHI 2012). Adaptive
    cutoff: low when slow (less jitter), higher when fast (less lag).
    Timestamps are in seconds and must be non-decreasing for a given
    instance; use reset() to start over and snap() to jump without a ramp
    (e.g. on re-engage after a track loss)."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = max(1e-4, min_cutoff)
        self.beta = max(0.0, beta)
        self.d_cutoff = max(1e-4, d_cutoff)
        self._initialized = False
        self._last_t = -1.0
        self._x_hat = 0.0
        self._dx_hat = 0.0

    def filter(self, value: float, t_sec: float) -> float:
        if not self._initialized:
            self._initialized = True
            self._last_t = t_sec
            self._x_hat = value
            self._dx_hat = 0.0
            return value
        dt = max(1e-3, t_sec - self._last_t)
        self._last_t = t_sec

        dx = (value - self._x_hat) / dt
        a_d = _alpha(dt, self.d_cutoff)
        self._dx_hat = a_d * dx + (1.0 - a_d) * self._dx_hat

        cutoff = self.min_cutoff + self.beta * abs(self._dx_hat)
        a = _alpha(dt, cutoff)
        self._x_hat = a * value + (1.0 - a) * self._x_hat
        return self._x_hat

    def reset(self) -> None:
        self._initialized = False
        self._last_t = -1.0
        self._x_hat = 0.0
        self._dx_hat = 0.0

    def snap(self, value: float, t_sec: float) -> None:
        self._initialized = True
        self._last_t = t_sec
        self._x_hat = value
        self._dx_hat = 0.0

    @property
    def initialized(self) -> bool:
        return self._initialized


class VectorOneEuro:
    """N independent OneEuroFilter instances (one per component), for
    filtering a 3D point or a joint-angle vector as a unit for reset/snap
    purposes while still filtering each axis independently (matching how
    FaceBoxStabilizer.java applies 4 independent filters to bbox corners)."""

    def __init__(self, n: int, min_cutoff: float, beta: float, d_cutoff: float):
        self._filters = [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(n)]

    def filter(self, value: np.ndarray, t_sec: float) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        return np.array([f.filter(v, t_sec) for f, v in zip(self._filters, value)])

    def reset(self) -> None:
        for f in self._filters:
            f.reset()

    def snap(self, value: np.ndarray, t_sec: float) -> None:
        value = np.asarray(value, dtype=np.float64)
        for f, v in zip(self._filters, value):
            f.snap(v, t_sec)

    @property
    def initialized(self) -> bool:
        return all(f.initialized for f in self._filters)


@dataclass
class RateLimiter:
    """Velocity/acceleration/jerk-limited rate limiter over an N-vector -
    the hard safety backstop (interlock 11): whatever the upstream retarget
    or filter stage produces, the command actually sent to the robot can
    never exceed these bounds per tick, regardless of the input. Internally
    a critically-damped control law (see _step_axis) tracks the target with
    no overshoot for a step input, hard-clamped to v/a/j caps at every tick.

    NOTE: `max_jerk` in
    robot_hardware_interface/config/joint_limits.yaml is documented there as
    a placeholder (1000.0), not a real limit - callers must pass their own
    teleop-specific caps (see plan: start 0.2 rad/s / 0.5 rad/s^2 / 10 rad/s^3),
    not that file's numbers.
    """

    n: int
    max_velocity: float
    max_acceleration: float
    max_jerk: float
    position: np.ndarray = field(init=False)
    velocity: np.ndarray = field(init=False)
    acceleration: np.ndarray = field(init=False)
    _initialized: bool = field(init=False, default=False)
    _omega: float = field(init=False)

    def __post_init__(self):
        self.position = np.zeros(self.n)
        self.velocity = np.zeros(self.n)
        self.acceleration = np.zeros(self.n)
        # Natural frequency for the critically-damped control law in
        # _step_axis, derived from the caps themselves (no extra tuning
        # parameter to expose): a_max/v_max is the characteristic
        # acceleration-per-unit-velocity scale the caps already imply.
        self._omega = self.max_acceleration / max(self.max_velocity, 1e-6)

    def reset(self, position) -> None:
        """Snap to `position` with zero velocity/acceleration - used on
        engage and on re-engage after a hold, never mid-motion."""
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.velocity = np.zeros(self.n)
        self.acceleration = np.zeros(self.n)
        self._initialized = True

    def step(self, target, dt: float) -> np.ndarray:
        """Advance one control tick toward `target`, respecting v/a/j caps.
        Returns the new position (also stored in self.position)."""
        if not self._initialized:
            self.reset(target)
            return self.position.copy()
        target = np.asarray(target, dtype=np.float64)
        dt = max(dt, 1e-4)

        for i in range(self.n):
            self.position[i], self.velocity[i], self.acceleration[i] = _step_axis(
                self.position[i], self.velocity[i], self.acceleration[i],
                target[i], dt, self.max_velocity, self.max_acceleration, self.max_jerk,
                self._omega,
            )
        return self.position.copy()

    def brake_to_stop(self, dt: float) -> np.ndarray:
        """One tick of decelerating toward zero velocity at the current
        position's own braking profile, ignoring any external target - used
        by the HOLD/disengage ramp (interlock 13): never an instant stop."""
        return self.step(self.position, dt)


def _step_axis(pos, vel, acc, target, dt, v_max, a_max, j_max, omega):
    """One rate-limited tick for a single scalar axis, using a critically
    damped (zeta=1) second-order control law as the desired acceleration,
    then clamping it to a hard v/a/j-limited cascade.

    Earlier revisions of this function tried a "braking distance" bang-bang
    scheme (accelerate at +a_max, then switch to -a_max once close enough)
    with a margin to account for the jerk-limited ramp between the two.
    That scheme persistently oscillated around the target instead of
    settling (caught by test_rate_limiter_converges_to_constant_target) -
    the jerk-limited transition between +a_max and -a_max takes real time
    (2*a_max/j_max), during which the discontinuous bang-bang decision
    flips back and forth around the switching point, chattering forever
    rather than converging. A critically damped law is continuous (no
    switching decision at all) and has NO overshoot for a step input in
    continuous time by construction - `omega` set so the acceleration
    response is naturally scaled by the same a_max/v_max ratio the caller's
    caps already imply (see RateLimiter.__post_init__), then hard-clamped so
    the caps are still exact bounds even where the smooth law alone
    wouldn't respect them (e.g. a very large initial error).
    """
    error = target - pos
    desired_accel = omega * omega * error - 2.0 * omega * vel
    desired_accel = float(np.clip(desired_accel, -a_max, a_max))

    max_dacc = j_max * dt
    new_acc = float(np.clip(desired_accel, acc - max_dacc, acc + max_dacc))
    new_acc = float(np.clip(new_acc, -a_max, a_max))

    new_vel = vel + new_acc * dt
    new_vel = float(np.clip(new_vel, -v_max, v_max))

    new_pos = pos + new_vel * dt
    return new_pos, new_vel, new_acc
