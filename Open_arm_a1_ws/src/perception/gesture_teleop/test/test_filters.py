"""Unit tests for core/filters.py."""
import numpy as np
import pytest

pytest.importorskip("hypothesis", reason="property-based tests need hypothesis")
from hypothesis import given, settings, strategies as st

from gesture_teleop.core.filters import OneEuroFilter, RateLimiter, VectorOneEuro


def test_one_euro_first_sample_passthrough():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.1, d_cutoff=1.0)
    assert f.filter(5.0, 0.0) == 5.0
    assert f.initialized


def test_one_euro_constant_input_stays_constant():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.1, d_cutoff=1.0)
    t = 0.0
    out = None
    for _ in range(50):
        out = f.filter(3.0, t)
        t += 1 / 30.0
    assert out == pytest.approx(3.0, abs=1e-6)


def test_one_euro_step_response_monotone_no_overshoot():
    """A step input should approach the new value monotonically, never
    overshooting past it (1-euro's exponential smoothing structurally
    cannot overshoot a step - this pins that down)."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5, d_cutoff=1.0)
    t = 0.0
    for _ in range(20):
        f.filter(0.0, t)
        t += 1 / 30.0
    prev = 0.0
    values = []
    for _ in range(60):
        v = f.filter(10.0, t)
        values.append(v)
        t += 1 / 30.0
    diffs = np.diff(values)
    assert np.all(diffs >= -1e-9)  # monotone non-decreasing
    assert np.all(np.array(values) <= 10.0 + 1e-6)  # never overshoots


def test_one_euro_reset_clears_state():
    f = OneEuroFilter(1.0, 0.1, 1.0)
    f.filter(5.0, 0.0)
    f.filter(5.0, 0.1)
    f.reset()
    assert not f.initialized
    assert f.filter(9.0, 1.0) == 9.0  # behaves like the first sample again


def test_one_euro_snap_jumps_without_ramp():
    f = OneEuroFilter(1.0, 0.1, 1.0)
    f.filter(0.0, 0.0)
    f.snap(100.0, 1.0)
    assert f.initialized
    # Immediately after snap, filtering the same value should not move away
    # from it (no residual velocity estimate pulling it back).
    out = f.filter(100.0, 1.033)
    assert out == pytest.approx(100.0, abs=1e-6)


def test_one_euro_dt_zero_is_safe():
    """Two samples at the identical timestamp must not divide by zero."""
    f = OneEuroFilter(1.0, 0.1, 1.0)
    f.filter(1.0, 5.0)
    out = f.filter(2.0, 5.0)  # same timestamp
    assert np.isfinite(out)


def test_vector_one_euro_independent_axes():
    v = VectorOneEuro(3, 1.0, 0.1, 1.0)
    out = v.filter(np.array([1.0, 2.0, 3.0]), 0.0)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])
    v.reset()
    assert not v.initialized
    v.snap(np.array([5.0, 5.0, 5.0]), 1.0)
    assert v.initialized


# --- RateLimiter -----------------------------------------------------------

_DT = 0.01
_VMAX = 1.0
_AMAX = 2.0
_JMAX = 20.0


def _make_limiter(n=1):
    return RateLimiter(n=n, max_velocity=_VMAX, max_acceleration=_AMAX, max_jerk=_JMAX)


def test_rate_limiter_reset_then_step_to_same_target_no_motion():
    rl = _make_limiter()
    rl.reset([0.0])
    out = rl.step([0.0], _DT)
    np.testing.assert_allclose(out, [0.0])


def test_rate_limiter_converges_to_constant_target():
    rl = _make_limiter()
    rl.reset([0.0])
    target = [3.0]
    for _ in range(2000):
        out = rl.step(target, _DT)
    assert out[0] == pytest.approx(3.0, abs=1e-3)


def test_rate_limiter_never_overshoots_a_step():
    """The critically-damped control law (see _step_axis) has no overshoot
    for a step input by construction; the v/a/j clamps on top of it can
    only slow the approach, never introduce one."""
    rl = _make_limiter()
    rl.reset([0.0])
    target = [1.0]
    max_pos = 0.0
    for _ in range(500):
        out = rl.step(target, _DT)
        max_pos = max(max_pos, out[0])
    assert max_pos <= 1.0 + 1e-6


@settings(max_examples=200, deadline=None)
@given(
    targets=st.lists(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False), min_size=1, max_size=50)
)
def test_rate_limiter_property_bounds_never_violated(targets):
    """For any sequence of (possibly wildly jumping) targets, the limiter's
    own internal velocity/acceleration/jerk must never exceed the configured
    caps, tick to tick - this is the safety guarantee interlock 11 depends
    on: whatever the upstream pipeline produces, the actual command is
    bounded."""
    rl = _make_limiter()
    rl.reset([0.0])
    prev_vel = 0.0
    prev_acc = 0.0
    for target in targets:
        rl.step([target], _DT)
        vel = rl.velocity[0]
        acc = rl.acceleration[0]
        assert abs(vel) <= _VMAX + 1e-9
        assert abs(acc) <= _AMAX + 1e-9
        assert abs(acc - prev_acc) <= _JMAX * _DT + 1e-6
        prev_vel, prev_acc = vel, acc


def test_rate_limiter_multi_axis_independent():
    rl = _make_limiter(n=2)
    rl.reset([0.0, 10.0])
    out = rl.step([0.0, 10.0], _DT)
    np.testing.assert_allclose(out, [0.0, 10.0], atol=1e-9)


def test_brake_to_stop_reduces_velocity_toward_zero():
    rl = _make_limiter()
    rl.reset([0.0])
    for _ in range(50):
        rl.step([5.0], _DT)  # build up some velocity
    assert abs(rl.velocity[0]) > 0.01
    prev_speed = abs(rl.velocity[0])
    for _ in range(300):
        rl.brake_to_stop(_DT)
    assert abs(rl.velocity[0]) < prev_speed
    assert abs(rl.velocity[0]) < 1e-3
