#ifndef HEAD_MOTOR_DRIVER_NODE_HPP
#define HEAD_MOTOR_DRIVER_NODE_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

// ============================================================================
//  head_motor_driver_node
//
//  Server side of HEAD_DRIVER_SPEC.md's UDS/NDJSON contract, consumed by
//  openarm_hardware's HeadHW plugin. Internally owns one persistent TCP
//  connection to the STM32 neck board (raw text protocol, SERVO/STATUS
//  commands - see robot-healthmate's NETWORK_PROTOCOL.md) and converts
//  radians <-> PWM us / ADC counts on this side only; the STM32 never sees
//  anything but PWM microseconds and replies with raw ADC counts.
// ============================================================================

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

    uint16_t rad_to_pwm(double rad) const;
    double adc_to_rad(int adc) const;
};

class EmaFilter {
public:
    explicit EmaFilter(double alpha) : alpha_(alpha) {}
    double update(double sample);

private:
    double alpha_;
    std::optional<double> value_;
};

// Persistent TCP client to the STM32 board. Reconnects lazily on the next
// command() call after any failure - never lets a dead connection wedge
// the 100Hz loop for more than one round trip.
class Stm32Link {
public:
    Stm32Link(std::string host, uint16_t port);
    ~Stm32Link();

    // Sends one line (CRLF appended), returns the reply line or nullopt on
    // any I/O failure. Thread-safe.
    std::optional<std::string> command(const std::string& line);

private:
    bool ensureConnected();
    void closeSocket();

    std::string host_;
    uint16_t port_;
    int sockfd_ = -1;
    std::mutex mutex_;
};

class HeadMotorDriver {
public:
    HeadMotorDriver(std::string calibration_path, std::string stm32_host,
                     uint16_t stm32_port, std::string uds_path);

    // Binds the UDS socket, accepts one hardware_interface client at a
    // time. Blocks until shutdown() is called from another thread.
    void udsServerLoop();

    // 100Hz round trip with the STM32: writes both servos' commanded PWM,
    // reads back both ADC channels, filters/stall-checks, streams state.
    void controlLoop();

    void shutdown();

private:
    void readCmdsFrom(int conn_fd);
    void sendState(double pan_rad, double tilt_rad, bool is_healthy,
                   const std::string& error_msg);
    static std::optional<int> parseAdc(const std::optional<std::string>& reply);

    JointCalibration pan_cal_;
    JointCalibration tilt_cal_;
    double stall_error_threshold_rad_ = 0.3;
    double stall_timeout_s_ = 0.8;
    EmaFilter pan_filter_;
    EmaFilter tilt_filter_;

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
