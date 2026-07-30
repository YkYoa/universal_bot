// Plain ROS2 service node for the neck board's two WS2812 RGB LED strips.
//
// Deliberately NOT part of ros2_control - LED commands aren't in the
// real-time control loop and don't need calibration/watchdog/filtering, so
// this just opens a fresh direct TCP connection per call to the STM32
// board and thinly wraps the RGB/MODE/BRIGHTNESS commands documented in
// robot-healthmate's NETWORK_PROTOCOL.md.
//
// Uses its own dedicated port (9878, separate from head_motor_driver_node's
// persistent 9877 connection) - the W5500 only exposes one TCP client slot
// per opened socket, so this needs its own socket/port to connect
// concurrently with the neck driver. See NETWORK_PROTOCOL.md.
//
// Run: ros2 run communication_devices head_led_node

#include <cstdlib>
#include <memory>
#include <string>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <rclcpp/rclcpp.hpp>

#include "openarm_messages/srv/set_brightness.hpp"
#include "openarm_messages/srv/set_color.hpp"
#include "openarm_messages/srv/set_mode.hpp"

namespace {

std::string sendLine(const std::string& host, uint16_t port, const std::string& line)
{
  struct addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* res = nullptr;
  if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0) {
    return "ERROR: DNS/address resolution failed";
  }

  int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (fd < 0) {
    freeaddrinfo(res);
    return "ERROR: socket() failed";
  }

  struct timeval tv{2, 0};
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  if (::connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
    freeaddrinfo(res);
    ::close(fd);
    return "ERROR: connect() failed";
  }
  freeaddrinfo(res);

  std::string out = line + "\r\n";
  if (::send(fd, out.c_str(), out.size(), MSG_NOSIGNAL) < 0) {
    ::close(fd);
    return "ERROR: send() failed";
  }

  char buf[256];
  ssize_t n = ::recv(fd, buf, sizeof(buf) - 1, 0);
  ::close(fd);
  if (n <= 0) {
    return "ERROR: no reply from STM32";
  }
  buf[n] = '\0';
  std::string reply(buf);
  while (!reply.empty() && (reply.back() == '\n' || reply.back() == '\r')) reply.pop_back();
  return reply;
}

}  // namespace

class HeadLedNode : public rclcpp::Node
{
public:
  HeadLedNode() : Node("head_led_node")
  {
    if (const char* env = std::getenv("HEAD_STM32_HOST")) host_ = env;
    if (const char* env = std::getenv("HEAD_STM32_LED_PORT")) port_ = static_cast<uint16_t>(std::stoi(env));

    set_color_srv_ = create_service<openarm_messages::srv::SetColor>(
      "head_led/set_color",
      std::bind(&HeadLedNode::handleSetColor, this, std::placeholders::_1, std::placeholders::_2));
    set_mode_srv_ = create_service<openarm_messages::srv::SetMode>(
      "head_led/set_mode",
      std::bind(&HeadLedNode::handleSetMode, this, std::placeholders::_1, std::placeholders::_2));
    set_brightness_srv_ = create_service<openarm_messages::srv::SetBrightness>(
      "head_led/set_brightness",
      std::bind(&HeadLedNode::handleSetBrightness, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(get_logger(), "head_led_node ready, board at %s:%u", host_.c_str(), port_);
  }

private:
  void handleSetColor(
    const std::shared_ptr<openarm_messages::srv::SetColor::Request> request,
    std::shared_ptr<openarm_messages::srv::SetColor::Response> response)
  {
    std::string cmd = "RGB " + std::to_string(request->led) + " " +
      std::to_string(request->r) + " " + std::to_string(request->g) + " " + std::to_string(request->b);
    std::string reply = sendLine(host_, port_, cmd);
    response->success = reply.rfind("SUCCESS", 0) == 0;
    response->message = reply;
  }

  void handleSetMode(
    const std::shared_ptr<openarm_messages::srv::SetMode::Request> request,
    std::shared_ptr<openarm_messages::srv::SetMode::Response> response)
  {
    std::string cmd = "MODE " + request->mode;
    if (request->speed_ms) cmd += " " + std::to_string(request->speed_ms);
    std::string reply = sendLine(host_, port_, cmd);
    response->success = reply.rfind("SUCCESS", 0) == 0;
    response->message = reply;
  }

  void handleSetBrightness(
    const std::shared_ptr<openarm_messages::srv::SetBrightness::Request> request,
    std::shared_ptr<openarm_messages::srv::SetBrightness::Response> response)
  {
    std::string cmd = "BRIGHTNESS " + request->led + " " + std::to_string(request->value);
    std::string reply = sendLine(host_, port_, cmd);
    response->success = reply.rfind("SUCCESS", 0) == 0;
    response->message = reply;
  }

  std::string host_ = "192.168.10.101";
  uint16_t port_ = 9878;
  rclcpp::Service<openarm_messages::srv::SetColor>::SharedPtr set_color_srv_;
  rclcpp::Service<openarm_messages::srv::SetMode>::SharedPtr set_mode_srv_;
  rclcpp::Service<openarm_messages::srv::SetBrightness>::SharedPtr set_brightness_srv_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HeadLedNode>());
  rclcpp::shutdown();
  return 0;
}
