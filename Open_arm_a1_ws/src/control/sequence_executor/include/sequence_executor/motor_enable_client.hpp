#pragma once
// -----------------------------------------------------------------------------
// motor_enable_client.hpp
//
// Re-activates the robot's physical (non-mock) ros2_control hardware
// components so they resend their "enable" command over CAN.
//
// Why this exists: a physical E-stop press+release cuts power at the motor
// driver, but ros2_control's hardware-component lifecycle never observes
// this - on_activate()'s enable_all() call (hardware_interface.cpp) only
// fires on an explicit activate transition, so `ros2 control
// list_hardware_components` keeps reporting the component "active" the
// whole time even though the physical motors dropped torque. The only way
// to make it resend "enable" over CAN is to force a fresh
// inactive -> active cycle through controller_manager - this class is
// exactly the manual recovery procedure (see terminal_command.md's
// zero-calibration doc for the CAN-level enable/disable pair this mirrors
// at the ros2_control layer) automated behind one call.
//
// Talks to controller_manager's list_hardware_components and
// set_hardware_component_state services, blocking on
// rclcpp::spin_until_future_complete - but NOT against the node passed in.
// ControlModeProbe (control_mode_probe.cpp) gets away with spinning the
// real node because it only ever probes once, at startup, before
// executor_app.cpp's executor.add_node(node)/spin(). enableAll() is called
// from inside RobotSupervisor::handleAccepted() and handleCommand(), which
// only run AT ALL because that same node is already registered with and
// being spun by the main executor - spin_until_future_complete() tries to
// add_node() it to a second, temporary executor, and rclcpp refuses a node
// already owned by one ("Node '...' has already been added to an
// executor.", an uncaught std::runtime_error that aborts the whole
// process). Found live in production 2026-09-16: qvic_fsm_node crash-
// looped on every accepted goal after this class's first deploy. The fix
// is to spin a second, throwaway node (internal_node_) that is never
// added to any other executor instead of node_ itself.
// -----------------------------------------------------------------------------
#include <cstdint>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <controller_manager_msgs/srv/list_hardware_components.hpp>
#include <controller_manager_msgs/srv/set_hardware_component_state.hpp>

namespace sequence_executor {

/// See file header comment. enableAll() is virtual, and the constructor
/// protected, so a test can subclass this to stub the re-enable cycle
/// without a real controller_manager to talk to - see
/// test_robot_supervisor.cpp's file header comment on why ControlModeProbe
/// couldn't get the same treatment without a bigger rework; this class was
/// designed testable from the start instead of retrofitted.
class MotorEnableClient
{
public:
  explicit MotorEnableClient(const rclcpp::Node::SharedPtr& node);
  virtual ~MotorEnableClient() = default;

  /// Cycles every physical (non-mock) hardware component through
  /// inactive -> active, forcing a fresh enable_all() over CAN on each.
  /// Blocking. Returns true iff every physical component ended the cycle
  /// "active"; `message` explains the outcome (component names and
  /// resulting states) either way, suitable for FsmCommand::Response or a
  /// failed goal's error_message.
  virtual bool enableAll(std::string& message);

private:
  using ListHardwareComponents = controller_manager_msgs::srv::ListHardwareComponents;
  using SetHardwareComponentState = controller_manager_msgs::srv::SetHardwareComponentState;

  /// Blocking single transition: `name` to the lifecycle state identified by
  /// `target_id`/`target_label` (both required by SetHardwareComponentState -
  /// see lifecycle_msgs/msg/State). Returns false with `error` set on
  /// timeout, an empty response, or the component refusing the transition.
  bool setComponentState(const std::string& name, uint8_t target_id,
                         const std::string& target_label, std::string& error);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_;
  // Never added to any other executor - see file header comment. Only used
  // to own the two clients below and as the argument to
  // spin_until_future_complete(); node_ is still what logger_ and the
  // caller's own node identity come from.
  rclcpp::Node::SharedPtr internal_node_;
  rclcpp::Client<ListHardwareComponents>::SharedPtr list_client_;
  rclcpp::Client<SetHardwareComponentState>::SharedPtr set_state_client_;
};

}  // namespace sequence_executor
