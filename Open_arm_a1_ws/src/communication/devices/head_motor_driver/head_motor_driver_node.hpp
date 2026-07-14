#ifndef HEAD_MOTOR_DRIVER_NODE_HPP
#define HEAD_MOTOR_DRIVER_NODE_HPP

#include <chrono>
#include <cstdint>
#include <string>

// ============================================================================
//  head_motor_driver_node
//
//  Bridges the ESP32 head firmware (head_api_protocol.h, raw ADC + PWM) to
//  the UDS/NDJSON contract in HEAD_DRIVER_SPEC.md consumed by openarm_hardware.
//
//  Responsibilities that exist ONLY because MG996R has no native feedback
//  loop (see: no is_healthy/error_code/velocity/load from the servo itself):
//    - ADC(raw) <-> radian conversion via per-unit calibration
//    - EMA filtering of the noisy analog tap
//    - Stall/health detection (servo gives us nothing for this natively)
//    - 250ms command watchdog + hold-last-position (spec section 6)
// ============================================================================

struct JointCalibration {
    uint16_t adc_min = 0;
    uint16_t adc_max = 4095;
    double   min_angle_rad = 0.0;
    double   max_angle_rad = 0.0;
    uint16_t pwm_min_us = 500;
    uint16_t pwm_max_us = 2500;
    bool     invert = false;
};

struct DriverConfig {
    JointCalibration pan;
    JointCalibration tilt;
    double   stall_error_threshold_rad = 0.15;
    std::chrono::milliseconds stall_timeout{400};
    double   position_filter_alpha = 0.3;
    std::chrono::milliseconds cmd_watchdog_timeout{250}; // spec section 6
};

enum class HeadErrorCode : int32_t {
    kNone            = 0,
    kStallDetected   = 1,
    kAdcOutOfRange   = 2,
    kCommandTimeout  = 3,
    kCalibrationMissing = 4,
};

// One instance per joint (pan / tilt). Owns calibration + filtering + stall
// state for that joint only. Does NOT own the socket or the ESP32 transport.
class HeadJointState {
public:
    explicit HeadJointState(JointCalibration calib) : calib_(calib) {}

    // raw ADC count -> filtered radians. Call once per control tick with the
    // latest HeadFeedbackPayload field for this joint.
    double updateFromRawAdc(uint16_t adc_raw);

    // target radians -> pulse width to send in HeadApiPayload. Clamps to
    // [min_angle_rad, max_angle_rad] per spec section 5 before converting.
    uint16_t commandToPulseUs(double target_rad);

    // compares last commanded vs filtered measured position over
    // stall_timeout; true once threshold exceeded that long.
    bool isStalled(std::chrono::steady_clock::time_point now) const;

    double filteredPositionRad() const { return filtered_rad_; }
    double lastCommandedRad() const { return last_commanded_rad_; }

private:
    JointCalibration calib_;
    double filtered_rad_ = 0.0;
    double last_commanded_rad_ = 0.0;
    bool   filter_initialized_ = false;
    std::chrono::steady_clock::time_point error_since_{};
    bool   error_active_ = false;

    double adcToAngleRad(uint16_t raw) const;
    double clampToLimits(double rad) const;
};

// Drives the two transports:
//   - UDS server, NDJSON, per HEAD_DRIVER_SPEC.md (server role - section 3)
//   - HTTP/serial client to ESP32, HeadApiPayload/HeadFeedbackPayload
// Runs the control loop on a dedicated thread per spec section 6
// (non-blocking I/O requirement) - socket thread never blocks on ESP32 I/O.
class HeadMotorDriverNode {
public:
    explicit HeadMotorDriverNode(DriverConfig config);

    // Binds UDS socket, starts control loop thread @ 100Hz state / handles
    // cmd messages as they arrive. Blocks until shutdown() called.
    void run();
    void shutdown();

private:
    // NDJSON handlers (spec section 4)
    void onCmdMessage(double pan_cmd_rad, double tilt_cmd_rad,
                       std::chrono::steady_clock::time_point recv_time);
    std::string buildStateMessageJson(uint32_t seq);

    // ESP32 transport round-trip: send HeadApiPayload, receive HeadFeedbackPayload
    bool pollEsp32(uint16_t pan_pulse_us, uint16_t tilt_pulse_us,
                   uint16_t& pan_adc_out, uint16_t& tilt_adc_out);

    // spec section 6: no cmd for > cmd_watchdog_timeout -> hold last position,
    // do not extrapolate, set error_code = kCommandTimeout in next state msg.
    void checkWatchdog(std::chrono::steady_clock::time_point now);

    DriverConfig config_;
    HeadJointState pan_joint_;
    HeadJointState tilt_joint_;

    std::chrono::steady_clock::time_point last_cmd_recv_time_{};
    bool watchdog_tripped_ = false;
    bool running_ = false;
};

#endif // HEAD_MOTOR_DRIVER_NODE_HPP
