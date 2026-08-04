#include "control_server.h"

namespace robot_control
{

ControlServer::ControlServer(const rclcpp::NodeOptions& options)
: rclcpp::Node("control_server", options)
{
  const std::string can_interface = declare_parameter<std::string>("can_interface", "can0");
  const bool can_fd = declare_parameter<bool>("can_fd", true);
  const bool hand = declare_parameter<bool>("hand", false);

  drive_control_ = std::make_unique<DriveControl>(can_interface, can_fd, hand);

  drive_command_service_ = create_service<DriveCommand>(
    "~/drive_command",
    std::bind(
      &ControlServer::handle_drive_command, this,
      std::placeholders::_1, std::placeholders::_2));

  RCLCPP_INFO(
    get_logger(), "control_server ready on %s (can_fd=%s, hand=%s)",
    can_interface.c_str(), can_fd ? "true" : "false", hand ? "true" : "false");
}

void ControlServer::handle_drive_command(
  const std::shared_ptr<DriveCommand::Request> request,
  std::shared_ptr<DriveCommand::Response> response)
{
  const auto command = static_cast<DriveCommandType>(request->command);

  bool ok = false;
  switch (command) {
    case DriveCommandType::SERVO_ON:
      ok = drive_control_->servoOn();
      break;
    case DriveCommandType::SERVO_OFF:
      ok = drive_control_->servoOff();
      break;
    // No separate hardware-level "halt" primitive exists (see control_def.h) -
    // both halt and quick-stop map onto DriveControl's one quick-stop latch.
    case DriveCommandType::SET_HALT:
    case DriveCommandType::SET_QUICK_STOP:
      ok = drive_control_->setQuickStop();
      break;
    case DriveCommandType::RESET_HALT:
    case DriveCommandType::RESET_QUICK_STOP:
      ok = drive_control_->resetQuickStop();
      break;
    case DriveCommandType::HOME_DRIVES:
      ok = drive_control_->homeDrives();
      break;
    default:
      response->success = false;
      response->message = "unknown DriveCommandType: " + std::to_string(request->command);
      RCLCPP_ERROR(get_logger(), "%s", response->message.c_str());
      return;
  }

  response->success = ok;
  response->message = drive_control_->getErrorCode();
}

}  // namespace robot_control

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<robot_control::ControlServer>());
  rclcpp::shutdown();
  return 0;
}
