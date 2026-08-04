#pragma once

#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <openarm_messages/srv/drive_command.hpp>

#include "control_def.h"
#include "drive_control.h"

namespace robot_control
{

/**
 * @brief Low-level safety + hand command server - servo on/off, halt,
 *        quick-stop, homing. Sits beside robot_skills' ExecuteSkill action
 *        server (MoveIt trajectory execution) and robot_hardware_interface's
 *        ros2_control loop, not on top of or instead of either - see
 *        drive_control.h's CAN dual-ownership warning before running this
 *        alongside robot_hardware_interface on the same CAN interface.
 */
class ControlServer : public rclcpp::Node
{
public:
  using DriveCommand = openarm_messages::srv::DriveCommand;

  explicit ControlServer(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  void handle_drive_command(
    const std::shared_ptr<DriveCommand::Request> request,
    std::shared_ptr<DriveCommand::Response> response);

  std::unique_ptr<DriveControl> drive_control_;
  rclcpp::Service<DriveCommand>::SharedPtr drive_command_service_;
};

}  // namespace robot_control
