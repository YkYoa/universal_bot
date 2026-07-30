#include "head/hardware_interface.hpp"

#include <chrono>
#include <cstring>
#include <optional>
#include <sstream>
#include <thread>

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/logging.hpp>

namespace openarm_hardware {

namespace {

/* Minimal flat-JSON field extraction. Both ends of this socket (this
 * plugin and head_motor_driver_node) are written together for this one
 * fixed schema, so a full JSON library is unnecessary overhead - this is
 * NOT a general-purpose parser. */

std::optional<double> extract_number(const std::string& line, const std::string& key)
{
  std::string needle = "\"" + key + "\":";
  auto pos = line.find(needle);
  if (pos == std::string::npos) return std::nullopt;
  pos += needle.size();
  auto end = line.find_first_of(",}", pos);
  if (end == std::string::npos) return std::nullopt;
  try {
    return std::stod(line.substr(pos, end - pos));
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<bool> extract_bool(const std::string& line, const std::string& key)
{
  std::string needle = "\"" + key + "\":";
  auto pos = line.find(needle);
  if (pos == std::string::npos) return std::nullopt;
  pos += needle.size();
  if (line.compare(pos, 4, "true") == 0) return true;
  if (line.compare(pos, 5, "false") == 0) return false;
  return std::nullopt;
}

}  // namespace

bool HeadHW::parse_config()
{
  const auto& params = info_.hardware_parameters;

  if (auto it = params.find("socket_path"); it != params.end()) {
    socket_path_ = it->second;
  }
  if (auto it = params.find("retry_interval_s"); it != params.end()) {
    retry_interval_s_ = std::stod(it->second);
  }
  if (auto it = params.find("retry_timeout_s"); it != params.end()) {
    retry_timeout_s_ = std::stod(it->second);
  }

  joint_names_.clear();
  for (const auto& joint : info_.joints) {
    joint_names_.push_back(joint.name);
  }

  RCLCPP_INFO(
    rclcpp::get_logger("HeadHW"),
    "Config: socket_path=%s joints=%zu", socket_path_.c_str(), joint_names_.size());

  return true;
}

hardware_interface::CallbackReturn HeadHW::on_init(
  const hardware_interface::HardwareComponentInterfaceParams& params)
{
  if (hardware_interface::SystemInterface::on_init(params) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }

  if (!parse_config()) {
    return CallbackReturn::ERROR;
  }

  const std::size_t n = joint_names_.size();
  pos_commands_.assign(n, 0.0);
  vel_commands_.assign(n, 0.0);
  eff_commands_.assign(n, 0.0);
  pos_states_.assign(n, 0.0);
  vel_states_.assign(n, 0.0);
  eff_states_.assign(n, 0.0);

  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> HeadHW::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  state_interfaces.reserve(joint_names_.size() * 3);

  for (std::size_t i = 0; i < joint_names_.size(); ++i) {
    state_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_POSITION, &pos_states_[i]);
    state_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_VELOCITY, &vel_states_[i]);
    state_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_EFFORT, &eff_states_[i]);
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> HeadHW::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  command_interfaces.reserve(joint_names_.size() * 3);

  for (std::size_t i = 0; i < joint_names_.size(); ++i) {
    command_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_POSITION, &pos_commands_[i]);
    command_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_VELOCITY, &vel_commands_[i]);
    command_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_EFFORT, &eff_commands_[i]);
  }

  return command_interfaces;
}

bool HeadHW::connect_socket()
{
  close_socket();

  sockfd_ = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sockfd_ < 0) {
    return false;
  }

  struct sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

  if (::connect(sockfd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
    close_socket();
    return false;
  }

  // Non-blocking so read() never stalls the control loop waiting on the driver.
  int flags = fcntl(sockfd_, F_GETFL, 0);
  fcntl(sockfd_, F_SETFL, flags | O_NONBLOCK);

  connected_ = true;
  rx_buffer_.clear();
  return true;
}

void HeadHW::close_socket()
{
  if (sockfd_ >= 0) {
    ::close(sockfd_);
    sockfd_ = -1;
  }
  connected_ = false;
}

bool HeadHW::send_line(const std::string& line)
{
  if (!connected_) return false;
  ssize_t sent = ::send(sockfd_, line.c_str(), line.size(), MSG_NOSIGNAL);
  if (sent < 0 || static_cast<size_t>(sent) != line.size()) {
    connected_ = false;
    return false;
  }
  return true;
}

std::string HeadHW::poll_latest_line()
{
  if (!connected_) return "";

  char chunk[4096];
  while (true) {
    ssize_t n = ::recv(sockfd_, chunk, sizeof(chunk), 0);
    if (n > 0) {
      rx_buffer_.append(chunk, static_cast<size_t>(n));
      continue;
    }
    if (n == 0) {
      // Peer closed the connection.
      connected_ = false;
      break;
    }
    // n < 0: EAGAIN/EWOULDBLOCK means no more data right now (expected on
    // a non-blocking socket); anything else is a real error.
    if (errno != EAGAIN && errno != EWOULDBLOCK) {
      connected_ = false;
    }
    break;
  }

  std::string latest;
  auto newline = rx_buffer_.find('\n');
  while (newline != std::string::npos) {
    latest = rx_buffer_.substr(0, newline);
    rx_buffer_.erase(0, newline + 1);
    newline = rx_buffer_.find('\n');
  }
  return latest;
}

hardware_interface::CallbackReturn HeadHW::on_activate(
  const rclcpp_lifecycle::State& /*previous_state*/)
{
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::milliseconds(static_cast<int>(retry_timeout_s_ * 1000));

  while (std::chrono::steady_clock::now() < deadline) {
    if (connect_socket()) {
      RCLCPP_INFO(rclcpp::get_logger("HeadHW"), "Connected to %s", socket_path_.c_str());
      return CallbackReturn::SUCCESS;
    }
    RCLCPP_WARN(
      rclcpp::get_logger("HeadHW"),
      "Retrying connection to %s ...", socket_path_.c_str());
    std::this_thread::sleep_for(
      std::chrono::milliseconds(static_cast<int>(retry_interval_s_ * 1000)));
  }

  RCLCPP_ERROR(
    rclcpp::get_logger("HeadHW"),
    "Could not connect to %s after %.1fs - is head_motor_driver_node running?",
    socket_path_.c_str(), retry_timeout_s_);
  return CallbackReturn::FAILURE;
}

hardware_interface::CallbackReturn HeadHW::on_deactivate(
  const rclcpp_lifecycle::State& /*previous_state*/)
{
  close_socket();
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type HeadHW::read(
  const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/)
{
  if (!connected_) {
    // Per spec: freeze last known state and let write()/on_activate's
    // retry loop handle reconnection - don't fail the whole controller.
    return hardware_interface::return_type::OK;
  }

  std::string line = poll_latest_line();
  if (line.empty()) {
    return hardware_interface::return_type::OK;
  }

  // joint order matches xacro: [0]=neck_joint (pan), [1]=head_joint (tilt)
  if (joint_names_.size() >= 1) {
    if (auto v = extract_number(line, "pan_pos")) pos_states_[0] = *v;
    if (auto v = extract_number(line, "pan_vel")) vel_states_[0] = *v;
  }
  if (joint_names_.size() >= 2) {
    if (auto v = extract_number(line, "tilt_pos")) pos_states_[1] = *v;
    if (auto v = extract_number(line, "tilt_vel")) vel_states_[1] = *v;
  }
  if (auto v = extract_bool(line, "is_healthy")) {
    if (*v != is_healthy_) {
      is_healthy_ = *v;
      RCLCPP_WARN_EXPRESSION(
        rclcpp::get_logger("HeadHW"), !is_healthy_,
        "Driver reports unhealthy state");
    }
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type HeadHW::write(
  const rclcpp::Time& time, const rclcpp::Duration& /*period*/)
{
  if (!connected_) {
    return hardware_interface::return_type::OK;
  }

  double pan_cmd = pos_commands_.size() >= 1 ? pos_commands_[0] : 0.0;
  double tilt_cmd = pos_commands_.size() >= 2 ? pos_commands_[1] : 0.0;

  std::ostringstream oss;
  oss << "{\"type\":\"cmd\",\"seq\":" << cmd_seq_++
      << ",\"timestamp\":" << time.seconds()
      << ",\"pan_cmd\":" << pan_cmd
      << ",\"tilt_cmd\":" << tilt_cmd
      << "}\n";

  send_line(oss.str());
  return hardware_interface::return_type::OK;
}

}  // namespace openarm_hardware

PLUGINLIB_EXPORT_CLASS(openarm_hardware::HeadHW, hardware_interface::SystemInterface)
