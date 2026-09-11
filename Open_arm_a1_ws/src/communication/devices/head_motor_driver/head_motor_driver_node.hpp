#ifndef HEAD_MOTOR_DRIVER_NODE_HPP
#define HEAD_MOTOR_DRIVER_NODE_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "motion_profile.hpp"

// ============================================================================
//  head_motor_driver_node
//
//  Server side of HEAD_DRIVER_SPEC.md's UDS/NDJSON contract, consumed by
//  robot_hardware_interface's HeadHW plugin. Internally owns one persistent TCP
//  connection to the STM32 neck board (raw text protocol, SERVO/STATUS
//  commands - see robot-healthmate's NETWORK_PROTOCOL.md) and converts
//  radians <-> PWM us / ADC counts on this side only; the STM32 never sees
//  anything but PWM microseconds and replies with raw ADC counts.
// ============================================================================

struct CalibrationPoint {
    double adc = 0.0;
    double deg = 0.0;
};

struct JointCalibration {
    double min_angle_rad = 0.0;
    double max_angle_rad = 0.0;
    uint16_t pwm_min_us = 500;
    uint16_t pwm_max_us = 2500;
    bool invert = false;
    double center_deg = 0.0;
    double pwm_to_deg_slope = 0.0;
    double pwm_to_deg_intercept = 0.0;
    double adc_to_deg_slope = 0.0;
    double adc_to_deg_intercept = 0.0;

    // Multi-point ADC->degree lookup table (see auto_calibrate.py), sorted
    // by adc ascending. When non-empty, adc_to_rad() interpolates through
    // this instead of the single linear adc_to_deg_slope/intercept fit -
    // corrects potentiometer nonlinearity a 2-point fit can't capture.
    // Empty means "not calibrated yet with the new tool", falls back to
    // the linear fit.
    std::vector<CalibrationPoint> points;

    // Motion profile limits (rad/s, rad/s^2) for the trapezoidal rate
    // limiter in HeadMotorDriver::controlLoop(). Defaults are conservative
    // for a hobby servo; override per-joint in head_calibration.yaml.
    double max_velocity_rad_s = 3.0;
    double max_accel_rad_s2 = 8.0;

    /** Converts a calibrated joint angle (radians) to a PWM pulse width (us),
     *  applying `invert` and clamping to [pwm_min_us, pwm_max_us]. */
    uint16_t rad_to_pwm(double rad) const;
    /** Converts a raw ADC reading to a joint angle (radians): interpolates
     *  through `points` when calibrated with the multi-point tool, otherwise
     *  falls back to the linear adc_to_deg_slope/intercept fit. */
    double adc_to_rad(int adc) const;
};

/** Exponential moving average filter: value_ += alpha_ * (sample - value_),
 *  seeded with the first sample it sees. */
class EmaFilter {
public:
    /** `alpha` in (0, 1]; higher = less smoothing, more responsive. */
    explicit EmaFilter(double alpha) : alpha_(alpha) {}
    /** Folds in one new `sample`, returning the updated filtered value. */
    double update(double sample);

private:
    double alpha_;
    std::optional<double> value_;
};

/** Persistent TCP client to the STM32 board. Reconnects lazily on the next
 *  command() call after any failure - never lets a dead connection wedge
 *  the 100Hz loop for more than one round trip. */
class Stm32Link {
public:
    /** Stores the target `host`/`port`; the socket is opened lazily on first command(). */
    Stm32Link(std::string host, uint16_t port);
    /** Closes the socket if still open. */
    ~Stm32Link();

    /** Sends one line (CRLF appended), returns the reply line or nullopt on
     *  any I/O failure. Thread-safe. */
    std::optional<std::string> command(const std::string& line);

private:
    /** Opens the socket if not already connected; returns false on failure. */
    bool ensureConnected();
    /** Closes and invalidates the socket, if open. */
    void closeSocket();

    std::string host_;
    uint16_t port_;
    int sockfd_ = -1;
    std::mutex mutex_;
};

/** Server side of HEAD_DRIVER_SPEC.md's UDS/NDJSON contract, consumed by
 *  robot_hardware_interface's HeadHW plugin. Owns the persistent Stm32Link and
 *  runs the 100Hz control loop that converts commands to PWM and STM32 ADC
 *  replies back to filtered joint angles. */
class HeadMotorDriver {
public:
    /** Loads calibration from `calibration_path`, connects Stm32Link to
     *  `stm32_host:stm32_port`, and prepares (without yet binding) the UDS
     *  server at `uds_path`. */
    HeadMotorDriver(std::string calibration_path, std::string stm32_host,
                     uint16_t stm32_port, std::string uds_path);

    /** Binds the UDS socket, accepts one hardware_interface client at a
     *  time. Blocks until shutdown() is called from another thread. */
    void udsServerLoop();

    /** 100Hz round trip with the STM32: writes both servos' commanded PWM,
     *  reads back both ADC channels, filters/stall-checks, streams state. */
    void controlLoop();

    /** Signals udsServerLoop()/controlLoop() to exit and unblocks any pending accept(). */
    void shutdown();

private:
    /** Reads NDJSON command lines from `conn_fd` (the current UDS client)
     *  until it disconnects or shutdown() is called, updating pan_cmd_/tilt_cmd_. */
    void readCmdsFrom(int conn_fd);
    /** Writes one NDJSON state line (pan/tilt angle, health, sequence number,
     *  optional error) to the current UDS client, if any. */
    void sendState(double pan_rad, double tilt_rad, bool is_healthy,
                   const std::string& error_msg);
    /** Parses the STM32's ADC reply string into an integer count, or nullopt
     *  if `reply` is absent or not parseable. */
    static std::optional<int> parseAdc(const std::optional<std::string>& reply);

    JointCalibration pan_cal_;
    JointCalibration tilt_cal_;
    double stall_error_threshold_rad_ = 0.3;
    double stall_timeout_s_ = 0.8;
    EmaFilter pan_filter_;
    EmaFilter tilt_filter_;

    // Rate-limits the raw ROS-commanded angle before it's converted to PWM,
    // so a step change (e.g. a single-point RViz goal) becomes a smooth
    // accel/cruise/decel motion instead of an instant PWM jump.
    TrapezoidalProfile pan_profile_;
    TrapezoidalProfile tilt_profile_;
    double last_profile_tick_s_ = 0.0;
    bool profile_initialized_ = false;

    Stm32Link link_;
    std::string uds_path_;

    std::mutex cmd_mutex_;
    double pan_cmd_ = 0.0;
    double tilt_cmd_ = 0.0;
    double last_cmd_time_s_ = 0.0;

    std::mutex uds_client_mutex_;
    int uds_client_fd_ = -1;

    uint32_t seq_ = 0;
    std::optional<double> stall_since_s_;
    std::atomic<bool> running_{true};
    int listen_fd_ = -1;
};

#endif // HEAD_MOTOR_DRIVER_NODE_HPP
