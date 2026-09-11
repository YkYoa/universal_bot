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

  /// Parses hardware params, generates joint names, sizes the command/state
  /// buffers, and (when built with openarm_can) opens the CAN interface and
  /// inits the arm motors (+ gripper motor if `hand`) in the configured
  /// damiao control mode.
  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams& params) override;

  /// Refreshes/receives once from the CAN bus so initial state is populated
  /// before activation.
  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State& previous_state) override;

  /// Exports position/velocity/effort state interfaces for every joint.
  OPENARM_HARDWARE_PUBLIC
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  /// Exports position/velocity/effort command interfaces for every joint.
  OPENARM_HARDWARE_PUBLIC
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  /// Enables the motors and calls return_to_zero() so the arm starts each
  /// activation from a known commanded position instead of wherever the
  /// stale command buffers last pointed.
  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State& previous_state) override;

  /// Drives the arm home (drive_home_blocking()) then disables the motors,
  /// retrying disable_all() shutdown_disable_retries_ times.
  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State& previous_state) override;

  /// Reads back position/velocity/torque for every arm motor (and, if
  /// present, the gripper/hand-rotate motor via motor_radians_to_joint())
  /// into the exported state interfaces.
  OPENARM_HARDWARE_PUBLIC
  hardware_interface::return_type read(
    const rclcpp::Time& time, const rclcpp::Duration& period) override;

  /// Sends pos_commands_/vel_commands_/tau_commands_ to the motors using
  /// whichever damiao control primitive matches control_mode_ (mit/position/
  /// torque; "velocity" is not implemented against the current openarm_can
  /// and logs an error instead of sending anything - see the .cpp for why).
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

  /// Reads hardware_parameters (can_interface, arm_prefix, ee_type, hand,
  /// can_fd, control_mode, per-joint kp/kd, gripper gains, timing/tolerance
  /// knobs) into this instance's members. Returns false on an invalid
  /// control_mode value.
  bool parse_config();
  /// Builds joint_names_ ("openarm_<prefix>joint1..7", plus a finger joint
  /// if `hand_`) from arm_prefix_/hand_.
  void generate_joint_names();
  /// Commands every arm (and gripper, if present) motor to its zero/closed
  /// reference position in whichever primitive control_mode_ supports
  /// (mit/position; a no-op for velocity/torque, which have no position
  /// reference).
  void return_to_zero();
  // Sends the zero-position command repeatedly (a single send_all() only
  // sets a target the motor's own firmware then chases - on_deactivate()
  // was cutting torque via disable_all() milliseconds later, before the
  // arm had any real time to get there) and waits, polling real position,
  // until all joints are within shutdown_home_tolerance_ or
  // shutdown_home_timeout_ms_ elapses. No-op in "velocity" mode (no
  // position reference to home to - see return_to_zero()).
  void drive_home_blocking();

  /// Converts a URDF joint value to the motor-shaft radians the end effector
  /// motor expects: passthrough for pinch_gripper, ratio-scaled passthrough
  /// for amazing_hand's direct-drive connector, or linear-stroke-to-angle
  /// scaling for openarm_hand's 0-0.044m gripper jaw.
  double joint_to_motor_radians(double joint_value) const;
  /// Inverse of joint_to_motor_radians().
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
  // Cruise speed (rad/s) used as POS_VEL mode's dq when control_mode_ ==
  // "position". The active joint_trajectory_controller config
  // (bimanual_controllers.yaml) only claims the position command interface,
  // so vel_commands_ is never written by anything - using it directly as dq
  // would command 0 velocity forever (motor accepts the position target but
  // never actually moves toward it). This fixed value stands in for that
  // missing live velocity command.
  double position_mode_velocity_{1.0};
  int shutdown_disable_retries_{3};
  int shutdown_retry_delay_ms_{100};
  // How long (max) and how close drive_home_blocking() waits/requires
  // before giving up and disabling torque anyway on shutdown.
  int shutdown_home_timeout_ms_{3000};
  double shutdown_home_tolerance_{0.05};

  std::array<double, ARM_DOF> kp_{70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0};
  std::array<double, ARM_DOF> kd_{2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5};
  double gripper_kp_{DEFAULT_GRIPPER_KP};
  double gripper_kd_{DEFAULT_GRIPPER_KD};
  // Motor-shaft-radians per joint-radian for "motor 8" when ee_type ==
  // amazing_hand (direct passthrough, unlike the openarm_hand gripper's
  // linear-stroke scaling below) - 1.0 unless the connector isn't direct
  // drive. See joint_to_motor_radians()/motor_radians_to_joint().
  double hand_rotate_ratio_{1.0};

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
  /// Parses hardware params (socket path, retry timing, filter alpha),
  /// builds joint_names_, and sizes the command/state/filter buffers.
  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams& params) override;

  /// Exports position (physically meaningful) plus inert 0.0 velocity/effort
  /// state interfaces for both neck joints (see class doc for why).
  OPENARM_HARDWARE_PUBLIC
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  /// Exports position (physically meaningful) plus inert 0.0 velocity/effort
  /// command interfaces for both neck joints.
  OPENARM_HARDWARE_PUBLIC
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  /// Connects to head_motor_driver_node's UDS socket (retrying up to
  /// retry_timeout_s_), failing activation if it never comes up.
  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State& previous_state) override;

  /// Closes the UDS socket.
  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State& previous_state) override;

  /// Polls the socket for the latest NDJSON state line (poll_latest_line())
  /// and updates the filtered position state (see position_filter_alpha_)
  /// plus is_healthy_.
  OPENARM_HARDWARE_PUBLIC
  hardware_interface::return_type read(
    const rclcpp::Time& time, const rclcpp::Duration& period) override;

  /// Sends the current pos_commands_ as one NDJSON command line to
  /// head_motor_driver_node.
  OPENARM_HARDWARE_PUBLIC
  hardware_interface::return_type write(
    const rclcpp::Time& time, const rclcpp::Duration& period) override;

private:
  /// Reads socket_path/retry_interval_s/retry_timeout_s/position_filter_alpha
  /// hardware parameters, falling back to their defaults when absent.
  bool parse_config();
  /// Opens the UDS socket at socket_path_, retrying every retry_interval_s_
  /// up to retry_timeout_s_ total; returns false on final failure.
  bool connect_socket();
  /// Closes and invalidates the socket, if open.
  void close_socket();
  /// Sends one line (newline-terminated) to the connected socket.
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
