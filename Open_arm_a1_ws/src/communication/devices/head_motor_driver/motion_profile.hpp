#ifndef MOTION_PROFILE_HPP
#define MOTION_PROFILE_HPP

#include <algorithm>
#include <cmath>

// Velocity/acceleration-limited rate limiter ("trapezoidal profile": a step
// change in target produces an accel ramp, a max-velocity cruise, and a
// decel ramp that lands on target without overshoot). Call step() once per
// control-loop tick - this is what turns a raw ROS-commanded step change
// into motion the servo can actually track smoothly, instead of an instant
// PWM jump. See auto_calibrate.py's note on why the servo's own internal
// loop doesn't need this (it does, for POSITION accuracy) - this is a
// separate concern: shaping the target that internal loop chases.
class TrapezoidalProfile {
public:
    TrapezoidalProfile() = default;
    TrapezoidalProfile(double max_velocity, double max_acceleration)
        : max_velocity_(max_velocity), max_acceleration_(max_acceleration) {}

    void reset(double value) {
        position_ = value;
        velocity_ = 0.0;
    }

    double position() const { return position_; }

    // Advances position_ toward target by at most dt seconds of motion,
    // decelerating in time to land on target without overshoot (distance-
    // to-stop = v^2 / (2*a), the standard trapezoidal/kinematic relation).
    double step(double target, double dt) {
        if (dt <= 0.0) return position_;

        double distance = target - position_;
        if (std::abs(distance) < 1e-9 && std::abs(velocity_) < 1e-9) {
            position_ = target;
            velocity_ = 0.0;
            return position_;
        }

        double max_stop_speed = std::sqrt(2.0 * max_acceleration_ * std::abs(distance));
        double desired_velocity = std::copysign(std::min(max_velocity_, max_stop_speed), distance);

        double max_dv = max_acceleration_ * dt;
        double dv = std::clamp(desired_velocity - velocity_, -max_dv, max_dv);
        velocity_ += dv;

        double next_position = position_ + velocity_ * dt;
        bool overshoot = (distance > 0.0 && next_position > target) ||
                          (distance < 0.0 && next_position < target);
        if (overshoot) {
            next_position = target;
            velocity_ = 0.0;
        }
        position_ = next_position;
        return position_;
    }

private:
    double max_velocity_ = 0.0;
    double max_acceleration_ = 0.0;
    double position_ = 0.0;
    double velocity_ = 0.0;
};

#endif  // MOTION_PROFILE_HPP
