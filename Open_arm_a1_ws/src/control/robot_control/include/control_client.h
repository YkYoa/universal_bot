#pragma once

#include <chrono>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <openarm_messages/srv/drive_command.hpp>

#include "control_def.h"

namespace robot_control
{

/**
 * @brief Thin synchronous wrapper around control_server's DriveCommand
 *        service - one blocking call per command, no persistent
 *        subscriptions/state. For future e-stop/safety callers; not wired
 *        into robot_skills in this plan.
 */
class ControlClient
{
public:
  using DriveCommand = openarm_messages::srv::DriveCommand;

  explicit ControlClient(
    const rclcpp::Node::SharedPtr& node,
    const std::string& server_name = "control_server");

  bool servoOn(int timeout_sec = 5);
  bool servoOff(int timeout_sec = 5);
  bool setHalt(int timeout_sec = 5);
  bool resetHalt(int timeout_sec = 5);
  bool setQuickStop(int timeout_sec = 5);
  bool resetQuickStop(int timeout_sec = 5);
  bool homeDrives(int timeout_sec = 5);

  /// Response message from the most recent call (success or failure).
  const std::string& lastMessage() const { return last_message_; }

private:
  bool sendCommand(DriveCommandType command, int timeout_sec);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Client<DriveCommand>::SharedPtr client_;
  std::string last_message_;
};

}  // namespace robot_control
