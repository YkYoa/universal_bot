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
// rclcpp::spin_until_future_complete the same way ControlModeProbe does
// (see control_mode_probe.cpp) - there is no other executor thread to hand
// this off to from inside a service/action callback on this node.
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
  rclcpp::Client<ListHardwareComponents>::SharedPtr list_client_;
  rclcpp::Client<SetHardwareComponentState>::SharedPtr set_state_client_;
};

}  // namespace sequence_executor
