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

  /// Creates a DriveCommand service client for `server_name`, using `node`
  /// for the ROS graph (does not spin it).
  explicit ControlClient(
    const rclcpp::Node::SharedPtr& node,
    const std::string& server_name = "control_server");

  /// Enables the drives (servo on). See lastMessage() for the result reason.
  bool servoOn(int timeout_sec = 5);
  /// Disables the drives (servo off).
  bool servoOff(int timeout_sec = 5);
  /// Requests a halt (mapped onto DriveControl's quick-stop latch - see control_def.h).
  bool setHalt(int timeout_sec = 5);
  /// Clears a previously requested halt.
  bool resetHalt(int timeout_sec = 5);
  /// Requests an immediate quick-stop.
  bool setQuickStop(int timeout_sec = 5);
  /// Clears a previously requested quick-stop.
  bool resetQuickStop(int timeout_sec = 5);
  /// Requests the homing sequence.
  bool homeDrives(int timeout_sec = 5);

  /// Response message from the most recent call (success or failure).
  const std::string& lastMessage() const { return last_message_; }

private:
  /// Shared implementation behind every public method: blocks up to
  /// `timeout_sec` for the service and the response, storing the result
  /// message in last_message_.
  bool sendCommand(DriveCommandType command, int timeout_sec);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Client<DriveCommand>::SharedPtr client_;
  std::string last_message_;
};

}  // namespace robot_control
