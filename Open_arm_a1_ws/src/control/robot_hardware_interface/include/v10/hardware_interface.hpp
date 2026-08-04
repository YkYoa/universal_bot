#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <hardware_interface/system_interface.hpp>
#include <rclcpp_lifecycle/state.hpp>

#include "v10/visibility.hpp"

namespace openarm_hardware {

/**
 * @brief OpenArm V10 ros2_control hardware interface (one arm per instance).
 *
 * Bimanual setup uses two plugin instances with different:
 *   - can_interface (e.g. can0 / can1)
 *   - arm_prefix    (e.g. left_ / right_)
 */
class OpenArm_v10HW : public hardware_interface::SystemInterface
{
public:
  OpenArm_v10HW();

  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams& params) override;

  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State& previous_state) override;

  OPENARM_HARDWARE_PUBLIC
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  OPENARM_HARDWARE_PUBLIC
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State& previous_state) override;

  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State& previous_state) override;

  OPENARM_HARDWARE_PUBLIC
  hardware_interface::return_type read(
    const rclcpp::Time& time, const rclcpp::Duration& period) override;

  OPENARM_HARDWARE_PUBLIC
  hardware_interface::return_type write(
    const rclcpp::Time& time, const rclcpp::Duration& period) override;

private:
  static constexpr std::size_t ARM_DOF = 7;

  static constexpr double GRIPPER_JOINT_OPEN = 0.044;
  static constexpr double GRIPPER_MOTOR_CLOSED = 0.0;
  static constexpr double GRIPPER_MOTOR_OPEN = -1.0472;
  static constexpr double DEFAULT_GRIPPER_KP = 5.0;
  static constexpr double DEFAULT_GRIPPER_KD = 0.1;

  bool parse_config();
  void generate_joint_names();
  void return_to_zero();

  double joint_to_motor_radians(double joint_value) const;
  double motor_radians_to_joint(double motor_radians) const;

  std::string can_interface_{"can0"};
  std::string arm_prefix_;
  std::string ee_type_{"parallel_link"};
  bool hand_{false};
  bool can_fd_{true};
  // One of "mit" (default, unchanged behavior - position+velocity+torque
  // with local gains), "position" (damiao POS_VEL mode - q + dq feed-
  // forward, no torque/gain control), "velocity" (damiao VEL mode - dq
  // only). See https://wiki.seeedstudio.com/damiao_series/#4-control-settings.
  std::string control_mode_{"mit"};
  int shutdown_disable_retries_{3};
  int shutdown_retry_delay_ms_{100};

  std::array<double, ARM_DOF> kp_{70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0};
  std::array<double, ARM_DOF> kd_{2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5};
  double gripper_kp_{DEFAULT_GRIPPER_KP};
  double gripper_kd_{DEFAULT_GRIPPER_KD};

  std::vector<std::string> joint_names_;

  std::vector<double> pos_commands_;
  std::vector<double> vel_commands_;
  std::vector<double> tau_commands_;
  std::vector<double> pos_states_;
  std::vector<double> vel_states_;
  std::vector<double> tau_states_;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief ros2_control hardware interface for the OpenArm v2 body's
 *        articulated neck (pan/tilt), driven by an STM32 board over a
 *        raw TCP text protocol (see NETWORK_PROTOCOL.md in the
 *        robot-healthmate repo). Talks to a separate low-level driver
 *        process (head_motor_driver_node, package communication_devices)
 *        over a Unix Domain Socket, NDJSON framing, per
 *        HEAD_DRIVER_SPEC.md - this class never touches the TCP socket
 *        directly.
 *
 * The xacro's configure_arm_joint macro declares position/velocity/effort
 * command AND state interfaces for both joints (shared with the arm's own
 * joints). Only position is physically meaningful here (no velocity/effort
 * sensing or control on this hardware) - velocity/effort are exported as
 * inert 0.0 placeholders solely so this plugin satisfies what the URDF
 * declares, matching this codebase's existing habit of over-exporting
 * rather than trimming the shared macro.
 */
class HeadHW : public hardware_interface::SystemInterface
{
public:
  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams& params) override;

  OPENARM_HARDWARE_PUBLIC
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  OPENARM_HARDWARE_PUBLIC
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State& previous_state) override;

  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State& previous_state) override;

  OPENARM_HARDWARE_PUBLIC
  hardware_interface::return_type read(
    const rclcpp::Time& time, const rclcpp::Duration& period) override;

  OPENARM_HARDWARE_PUBLIC
  hardware_interface::return_type write(
    const rclcpp::Time& time, const rclcpp::Duration& period) override;

private:
  bool parse_config();
  bool connect_socket();
  void close_socket();
  bool send_line(const std::string& line);
  /// Non-blocking: pulls whatever bytes are available into rx_buffer_ and
  /// returns the LAST complete NDJSON line found (older ones are dropped -
  /// only the freshest state matters for a control loop), or empty if none.
  std::string poll_latest_line();

  std::string socket_path_{"/tmp/openarm_head_motor.sock"};
  double retry_interval_s_{0.5};
  double retry_timeout_s_{10.0};

  /// Exponential moving average applied to raw pan/tilt position feedback
  /// before it's exposed on the state interface. The raw socket feedback has
  /// no filtering at all, so the reported position visibly jitters even when
  /// the head sits physically still. 1.0 = no filtering (raw passthrough);
  /// lower values smooth harder but add more lag between an actual move and
  /// it showing up in joint_states/RViz - kept close to 1.0 by default since
  /// this is a reporting fix, not meant to mask a real motion problem.
  double position_filter_alpha_{0.4};
  std::vector<double> pos_filtered_;
  std::vector<bool> pos_filter_initialized_;

  std::vector<std::string> joint_names_;

  std::vector<double> pos_commands_;
  std::vector<double> vel_commands_;
  std::vector<double> eff_commands_;
  std::vector<double> pos_states_;
  std::vector<double> vel_states_;
  std::vector<double> eff_states_;

  int sockfd_{-1};
  bool connected_{false};
  std::string rx_buffer_;
  uint32_t cmd_seq_{0};
  bool is_healthy_{false};
};

}  // namespace openarm_hardware
