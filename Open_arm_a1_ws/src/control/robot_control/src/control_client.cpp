#include "control_client.h"

#include <iostream>

namespace robot_control
{

ControlClient::ControlClient(const rclcpp::Node::SharedPtr& node, const std::string& server_name)
: node_(node)
{
  client_ = node_->create_client<DriveCommand>("/" + server_name + "/drive_command");
}

bool ControlClient::sendCommand(DriveCommandType command, int timeout_sec)
{
  if (!client_->wait_for_service(std::chrono::seconds(timeout_sec))) {
    last_message_ = "service unavailable";
    RCLCPP_ERROR(node_->get_logger(), "control_server service unavailable");
    return false;
  }

  auto request = std::make_shared<DriveCommand::Request>();
  request->command = static_cast<uint8_t>(command);

  auto future = client_->async_send_request(request);
  if (rclcpp::spin_until_future_complete(node_, future, std::chrono::seconds(timeout_sec)) !=
      rclcpp::FutureReturnCode::SUCCESS)
  {
    last_message_ = "request timed out";
    RCLCPP_ERROR(node_->get_logger(), "drive_command request timed out");
    return false;
  }

  auto response = future.get();
  last_message_ = response->message;
  return response->success;
}

bool ControlClient::servoOn(int timeout_sec)
{
  return sendCommand(DriveCommandType::SERVO_ON, timeout_sec);
}

bool ControlClient::servoOff(int timeout_sec)
{
  return sendCommand(DriveCommandType::SERVO_OFF, timeout_sec);
}

bool ControlClient::setHalt(int timeout_sec)
{
  return sendCommand(DriveCommandType::SET_HALT, timeout_sec);
}

bool ControlClient::resetHalt(int timeout_sec)
{
  return sendCommand(DriveCommandType::RESET_HALT, timeout_sec);
}

bool ControlClient::setQuickStop(int timeout_sec)
{
  return sendCommand(DriveCommandType::SET_QUICK_STOP, timeout_sec);
}

bool ControlClient::resetQuickStop(int timeout_sec)
{
  return sendCommand(DriveCommandType::RESET_QUICK_STOP, timeout_sec);
}

bool ControlClient::homeDrives(int timeout_sec)
{
  return sendCommand(DriveCommandType::HOME_DRIVES, timeout_sec);
}

}  // namespace robot_control

namespace
{

void print_usage(const char* prog)
{
  std::cerr << "Usage: " << prog
            << " <servo_on|servo_off|halt|reset_halt|quick_stop|reset_quick_stop|home>"
            << " [server_name]\n";
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("control_client");
  const std::string server_name = argc > 2 ? argv[2] : "control_server";
  robot_control::ControlClient client(node, server_name);

  const std::string cmd = argv[1];
  bool ok = false;
  if (cmd == "servo_on") {
    ok = client.servoOn();
  } else if (cmd == "servo_off") {
    ok = client.servoOff();
  } else if (cmd == "halt") {
    ok = client.setHalt();
  } else if (cmd == "reset_halt") {
    ok = client.resetHalt();
  } else if (cmd == "quick_stop") {
    ok = client.setQuickStop();
  } else if (cmd == "reset_quick_stop") {
    ok = client.resetQuickStop();
  } else if (cmd == "home") {
    ok = client.homeDrives();
  } else {
    print_usage(argv[0]);
    rclcpp::shutdown();
    return 1;
  }

  std::cout << (ok ? "OK: " : "FAILED: ") << client.lastMessage() << std::endl;
  rclcpp::shutdown();
  return ok ? 0 : 1;
}
