#include "v10/hardware_interface.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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

#if defined(OPENARM_HARDWARE_HAS_OPENARMCAN)
#include <openarm/can/socket/openarm.hpp>
#include <openarm/damiao_motor/dm_motor_constants.hpp>
#endif

namespace openarm_hardware {

struct OpenArm_v10HW::Impl
{
#if defined(OPENARM_HARDWARE_HAS_OPENARMCAN)
  std::unique_ptr<openarm::can::socket::OpenArm> openarm;
#endif
};

#if defined(OPENARM_HARDWARE_HAS_OPENARMCAN)
namespace {
openarm::damiao_motor::ControlMode damiaoControlModeFor(const std::string& control_mode)
{
  if (control_mode == "position") {
    return openarm::damiao_motor::ControlMode::POS_VEL;
  }
  if (control_mode == "velocity") {
    return openarm::damiao_motor::ControlMode::VEL;
  }
  return openarm::damiao_motor::ControlMode::MIT;
}
}  // namespace
#endif

OpenArm_v10HW::OpenArm_v10HW()
: impl_(std::make_unique<Impl>())
{
}

bool OpenArm_v10HW::parse_config()
{
  const auto& params = info_.hardware_parameters;

  if (auto it = params.find("can_interface"); it != params.end()) {
    can_interface_ = it->second;
  }
  if (auto it = params.find("arm_prefix"); it != params.end()) {
    arm_prefix_ = it->second;
  }
  if (auto it = params.find("ee_type"); it != params.end()) {
    ee_type_ = it->second;
  }
  if (auto it = params.find("hand"); it != params.end()) {
    std::string v = it->second;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    hand_ = (v == "true" || v == "1");
  }
  if (auto it = params.find("can_fd"); it != params.end()) {
    std::string v = it->second;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    can_fd_ = (v == "true" || v == "1");
  }
  if (auto it = params.find("control_mode"); it != params.end()) {
    control_mode_ = it->second;
    if (control_mode_ != "mit" && control_mode_ != "position" && control_mode_ != "velocity" &&
        control_mode_ != "torque") {
      RCLCPP_ERROR(
        rclcpp::get_logger("OpenArm_v10HW"),
        "control_mode must be 'mit', 'position', 'velocity', or 'torque', got '%s'",
        control_mode_.c_str());
      return false;
    }
  }
  if (auto it = params.find("position_mode_velocity"); it != params.end()) {
    position_mode_velocity_ = std::stod(it->second);
  }
  if (auto it = params.find("shutdown_home_timeout_ms"); it != params.end()) {
    shutdown_home_timeout_ms_ = std::stoi(it->second);
  }
  if (auto it = params.find("shutdown_home_tolerance"); it != params.end()) {
    shutdown_home_tolerance_ = std::stod(it->second);
  }
  if (auto it = params.find("shutdown_disable_retries"); it != params.end()) {
    shutdown_disable_retries_ = std::stoi(it->second);
  }
  if (auto it = params.find("shutdown_retry_delay_ms"); it != params.end()) {
    shutdown_retry_delay_ms_ = std::stoi(it->second);
  }

  for (std::size_t i = 1; i <= ARM_DOF; ++i) {
    if (auto it = params.find("kp" + std::to_string(i)); it != params.end()) {
      kp_[i - 1] = std::stod(it->second);
    }
    if (auto it = params.find("kd" + std::to_string(i)); it != params.end()) {
      kd_[i - 1] = std::stod(it->second);
    }
  }

  if (auto it = params.find("kp_hand"); it != params.end()) {
    gripper_kp_ = std::stod(it->second);
  }
  if (auto it = params.find("kd_hand"); it != params.end()) {
    gripper_kd_ = std::stod(it->second);
  }
  if (auto it = params.find("hand_rotate_ratio"); it != params.end()) {
    hand_rotate_ratio_ = std::stod(it->second);
  }

  RCLCPP_INFO(
    rclcpp::get_logger("OpenArm_v10HW"),
    "Config: can=%s prefix=%s hand=%s can_fd=%s ee_type=%s control_mode=%s "
    "shutdown_disable_retries=%d shutdown_retry_delay_ms=%d",
    can_interface_.c_str(),
    arm_prefix_.c_str(),
    hand_ ? "true" : "false",
    can_fd_ ? "true" : "false",
    ee_type_.c_str(),
    control_mode_.c_str(),
    shutdown_disable_retries_,
    shutdown_retry_delay_ms_);

  return true;
}

void OpenArm_v10HW::generate_joint_names()
{
  joint_names_.clear();

  for (std::size_t i = 1; i <= ARM_DOF; ++i) {
    joint_names_.push_back("openarm_" + arm_prefix_ + "joint" + std::to_string(i));
  }

  if (hand_) {
    joint_names_.push_back("openarm_" + arm_prefix_ + "finger_joint1");
  }
}

double OpenArm_v10HW::joint_to_motor_radians(double joint_value) const
{
  if (ee_type_ == "pinch_gripper") {
    return joint_value;
  }
  if (ee_type_ == "amazing_hand") {
    // "Motor 8" here spins the amazing_hand connector directly (a real
    // revolute joint in radians), not a linear gripper stroke - passthrough
    // (with an optional gear ratio) instead of the finger_joint1 stroke
    // scaling below, which is specific to openarm_hand's 0-0.044m jaw travel.
    return joint_value * hand_rotate_ratio_;
  }
  return (joint_value / GRIPPER_JOINT_OPEN) * GRIPPER_MOTOR_OPEN;
}

double OpenArm_v10HW::motor_radians_to_joint(double motor_radians) const
{
  if (ee_type_ == "pinch_gripper") {
    return motor_radians;
  }
  if (ee_type_ == "amazing_hand") {
    return motor_radians / hand_rotate_ratio_;
  }
  return GRIPPER_JOINT_OPEN * (motor_radians / GRIPPER_MOTOR_OPEN);
}

hardware_interface::CallbackReturn OpenArm_v10HW::on_init(
  const hardware_interface::HardwareComponentInterfaceParams& params)
{
  if (hardware_interface::SystemInterface::on_init(params) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }

  if (!parse_config()) {
    return CallbackReturn::ERROR;
  }

  generate_joint_names();

  const std::size_t total_joints = joint_names_.size();
  pos_commands_.assign(total_joints, 0.0);
  vel_commands_.assign(total_joints, 0.0);
  tau_commands_.assign(total_joints, 0.0);
  pos_states_.assign(total_joints, 0.0);
  vel_states_.assign(total_joints, 0.0);
  tau_states_.assign(total_joints, 0.0);

#if defined(OPENARM_HARDWARE_HAS_OPENARMCAN)
  impl_->openarm = std::make_unique<openarm::can::socket::OpenArm>(can_interface_, can_fd_);

  static const std::vector<openarm::damiao_motor::MotorType> motor_types = {
    openarm::damiao_motor::MotorType::DM8009,
    openarm::damiao_motor::MotorType::DM8009,
    openarm::damiao_motor::MotorType::DM4340,
    openarm::damiao_motor::MotorType::DM4340,
    openarm::damiao_motor::MotorType::DM4310,
    openarm::damiao_motor::MotorType::DM4310,
    openarm::damiao_motor::MotorType::DM4310,
  };
  static const std::vector<uint32_t> send_ids = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
  static const std::vector<uint32_t> recv_ids = {0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17};

  // Set per motor at init time (matches openarm_can's own examples, e.g.
  // python/examples/test_posvel.py) - not a runtime-switchable register.
  const openarm::damiao_motor::ControlMode dm_mode = damiaoControlModeFor(control_mode_);
  const std::vector<openarm::damiao_motor::ControlMode> control_modes(motor_types.size(), dm_mode);

  impl_->openarm->init_arm_motors(motor_types, send_ids, recv_ids, control_modes);

  if (hand_) {
    impl_->openarm->init_gripper_motor(
      openarm::damiao_motor::MotorType::DM4310, 0x08, 0x18, dm_mode);
  }
#endif

  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn OpenArm_v10HW::on_configure(
  const rclcpp_lifecycle::State& /*previous_state*/)
{
#if defined(OPENARM_HARDWARE_HAS_OPENARMCAN)
  if (impl_ && impl_->openarm) {
    impl_->openarm->refresh_all();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    impl_->openarm->recv_all();
  }
#endif
  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> OpenArm_v10HW::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  state_interfaces.reserve(joint_names_.size() * 3);

  for (std::size_t i = 0; i < joint_names_.size(); ++i) {
    state_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_POSITION, &pos_states_[i]);
    state_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_VELOCITY, &vel_states_[i]);
    state_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_EFFORT, &tau_states_[i]);
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> OpenArm_v10HW::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  command_interfaces.reserve(joint_names_.size() * 3);

  for (std::size_t i = 0; i < joint_names_.size(); ++i) {
    command_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_POSITION, &pos_commands_[i]);
    command_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_VELOCITY, &vel_commands_[i]);
    command_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_EFFORT, &tau_commands_[i]);
  }

  return command_interfaces;
}

void OpenArm_v10HW::return_to_zero()
{
#if defined(OPENARM_HARDWARE_HAS_OPENARMCAN)
  if (!impl_ || !impl_->openarm) {
    return;
  }

  impl_->openarm->refresh_all();

  if (control_mode_ == "mit") {
    std::vector<openarm::damiao_motor::MITParam> arm_params;
    for (std::size_t i = 0; i < ARM_DOF; ++i) {
      arm_params.push_back({kp_[i], kd_[i], 0.0, 0.0, 0.0});
    }
    impl_->openarm->get_arm().mit_control_all(arm_params);
    if (hand_) {
      impl_->openarm->get_gripper().mit_control_all(
        {{gripper_kp_, gripper_kd_, GRIPPER_MOTOR_CLOSED, 0.0, 0.0}});
    }
  } else if (control_mode_ == "position") {
    std::vector<openarm::damiao_motor::PosVelParam> arm_params(ARM_DOF, {0.0, position_mode_velocity_});
    impl_->openarm->get_arm().posvel_control_all(arm_params);
    if (hand_) {
      impl_->openarm->get_gripper().posvel_control_all({{GRIPPER_MOTOR_CLOSED, position_mode_velocity_}});
    }
  }
  // "velocity"/"torque" modes have no position reference to return to -
  // skip (write() will command 0 velocity, or whatever tau_commands_ is,
  // on the first cycle after activation regardless).

  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  impl_->openarm->recv_all();
#endif
}

void OpenArm_v10HW::drive_home_blocking()
{
#if defined(OPENARM_HARDWARE_HAS_OPENARMCAN)
  if (!impl_ || !impl_->openarm || control_mode_ == "velocity" || control_mode_ == "torque") {
    return;
  }

  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::milliseconds(shutdown_home_timeout_ms_);

  while (std::chrono::steady_clock::now() < deadline) {
    return_to_zero();  // re-sends the zero-position command each iteration

    impl_->openarm->refresh_all();
    impl_->openarm->recv_all();

    const auto& arm_motors = impl_->openarm->get_arm().get_motors();
    bool all_home = true;
    for (std::size_t i = 0; i < ARM_DOF && i < arm_motors.size(); ++i) {
      if (std::abs(arm_motors[i].get_position()) > shutdown_home_tolerance_) {
        all_home = false;
        break;
      }
    }
    if (all_home) {
      RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"), "Reached home position, disabling.");
      return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  RCLCPP_WARN(
    rclcpp::get_logger("OpenArm_v10HW"),
    "Did not reach home within %dms, disabling torque anyway.", shutdown_home_timeout_ms_);
#endif
}

hardware_interface::CallbackReturn OpenArm_v10HW::on_activate(
  const rclcpp_lifecycle::State& /*previous_state*/)
{
#if !defined(OPENARM_HARDWARE_HAS_OPENARMCAN)
  RCLCPP_ERROR(
    rclcpp::get_logger("OpenArm_v10HW"),
    "OpenArmCAN not available (stub build). Install libopenarm-can-dev and rebuild.");
  return CallbackReturn::ERROR;
#else
  if (!impl_ || !impl_->openarm) {
    RCLCPP_ERROR(rclcpp::get_logger("OpenArm_v10HW"), "OpenArm instance not initialized");
    return CallbackReturn::ERROR;
  }

  impl_->openarm->set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);
  impl_->openarm->enable_all();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  impl_->openarm->recv_all();

  return_to_zero();

  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"), "Activated on %s", can_interface_.c_str());
  return CallbackReturn::SUCCESS;
#endif
}

hardware_interface::CallbackReturn OpenArm_v10HW::on_deactivate(
  const rclcpp_lifecycle::State& /*previous_state*/)
{
#if defined(OPENARM_HARDWARE_HAS_OPENARMCAN)
  if (impl_ && impl_->openarm) {
    drive_home_blocking();

    for (int i = 0; i < shutdown_disable_retries_; ++i) {
      impl_->openarm->disable_all();
      std::this_thread::sleep_for(std::chrono::milliseconds(shutdown_retry_delay_ms_));
      impl_->openarm->recv_all();
    }
  }
#endif
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type OpenArm_v10HW::read(
  const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/)
{
#if defined(OPENARM_HARDWARE_HAS_OPENARMCAN)
  if (!impl_ || !impl_->openarm) {
    return hardware_interface::return_type::ERROR;
  }

  impl_->openarm->refresh_all();
  impl_->openarm->recv_all();

  const auto& arm_motors = impl_->openarm->get_arm().get_motors();
  for (std::size_t i = 0; i < ARM_DOF && i < arm_motors.size(); ++i) {
    pos_states_[i] = arm_motors[i].get_position();
    vel_states_[i] = arm_motors[i].get_velocity();
    tau_states_[i] = arm_motors[i].get_torque();
  }

  if (hand_ && joint_names_.size() > ARM_DOF) {
    const auto& gripper_motors = impl_->openarm->get_gripper().get_motors();
    if (!gripper_motors.empty()) {
      pos_states_[ARM_DOF] = motor_radians_to_joint(gripper_motors[0].get_position());
      vel_states_[ARM_DOF] = 0.0;
      tau_states_[ARM_DOF] = 0.0;
    }
  }
#endif

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type OpenArm_v10HW::write(
  const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/)
{
#if defined(OPENARM_HARDWARE_HAS_OPENARMCAN)
  if (!impl_ || !impl_->openarm) {
    return hardware_interface::return_type::ERROR;
  }

  const bool has_gripper = hand_ && joint_names_.size() > ARM_DOF;
  const double gripper_motor_cmd = has_gripper ? joint_to_motor_radians(pos_commands_[ARM_DOF]) : 0.0;

  if (control_mode_ == "mit") {
    std::vector<openarm::damiao_motor::MITParam> arm_params;
    arm_params.reserve(ARM_DOF);
    for (std::size_t i = 0; i < ARM_DOF; ++i) {
      arm_params.push_back({kp_[i], kd_[i], pos_commands_[i], vel_commands_[i], tau_commands_[i]});
    }
    impl_->openarm->get_arm().mit_control_all(arm_params);
    if (has_gripper) {
      impl_->openarm->get_gripper().mit_control_all(
        {{gripper_kp_, gripper_kd_, gripper_motor_cmd, 0.0, 0.0}});
    }
  } else if (control_mode_ == "position") {
    // dq intentionally NOT sourced from vel_commands_ - the active
    // joint_trajectory_controller config only claims the position command
    // interface, so vel_commands_ is always 0 (see position_mode_velocity_'s
    // doc comment in the header).
    std::vector<openarm::damiao_motor::PosVelParam> arm_params;
    arm_params.reserve(ARM_DOF);
    for (std::size_t i = 0; i < ARM_DOF; ++i) {
      arm_params.push_back({pos_commands_[i], position_mode_velocity_});
    }
    impl_->openarm->get_arm().posvel_control_all(arm_params);
    if (has_gripper) {
      impl_->openarm->get_gripper().posvel_control_all({{gripper_motor_cmd, position_mode_velocity_}});
    }
  } else if (control_mode_ == "velocity") {
    // The installed libopenarm-can does not expose a dedicated velocity
    // control API (only mit/posvel/posforce_control_all) - this mode was
    // never actually exercised on real hardware (stub-mode builds skipped
    // this whole #ifdef block, so the mismatch never surfaced at compile
    // time until real CAN support was linked in). Sending nothing is safer
    // than guessing at a replacement without verifying against real
    // hardware; use "mit" (with vel_commands_ as MITParam's dq, kp=0) if
    // velocity-only motion is actually needed.
    RCLCPP_ERROR_THROTTLE(
      rclcpp::get_logger("OpenArm_v10HW"), *rclcpp::Clock::make_shared(), 5000,
      "control_mode=velocity is not implemented against the current libopenarm-can - no commands sent");
  } else {  // "torque" - kp=kd=0 always, tau_commands_ as pure feedforward.
    // Only meaningful when something is actually writing tau_commands_ (e.g.
    // gravity_compensation_controller) - no controller in
    // bimanual_controllers.yaml claims the effort interface by default.
    std::vector<openarm::damiao_motor::MITParam> arm_params;
    arm_params.reserve(ARM_DOF);
    for (std::size_t i = 0; i < ARM_DOF; ++i) {
      arm_params.push_back({0.0, 0.0, 0.0, 0.0, tau_commands_[i]});
    }
    impl_->openarm->get_arm().mit_control_all(arm_params);
    if (has_gripper) {
      impl_->openarm->get_gripper().mit_control_all({{0.0, 0.0, 0.0, 0.0, tau_commands_[ARM_DOF]}});
    }
  }

  impl_->openarm->recv_all(100);
#endif

  return hardware_interface::return_type::OK;
}

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
  if (auto it = params.find("position_filter_alpha"); it != params.end()) {
    position_filter_alpha_ = std::stod(it->second);
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
  pos_filtered_.assign(n, 0.0);
  pos_filter_initialized_.assign(n, false);

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

  // Smooths raw position feedback with an EMA so joint_states/RViz don't
  // show jitter that isn't real head motion (see position_filter_alpha_).
  // First sample per joint is taken as-is so the filter doesn't ramp up
  // from a 0.0 default on startup.
  auto filter_position = [this](std::size_t i, double raw) {
    if (!pos_filter_initialized_[i]) {
      pos_filtered_[i] = raw;
      pos_filter_initialized_[i] = true;
    } else {
      pos_filtered_[i] = position_filter_alpha_ * raw +
                          (1.0 - position_filter_alpha_) * pos_filtered_[i];
    }
    return pos_filtered_[i];
  };

  // joint order matches xacro: [0]=neck_joint, [1]=head_joint. Swapped which
  // driver key feeds which index - physical wiring has neck_joint (index 0)
  // driven by the board's "tilt" motor and head_joint (index 1) driven by
  // its "pan" motor, confirmed by commanding each joint from RViz one at a
  // time and observing which motor actually moved. Must match write()'s
  // tilt_cmd/pan_cmd assignment below exactly - this used to read the
  // opposite fields (pan_pos for index 0, tilt_pos for index 1), so each
  // joint's reported state was actually the OTHER joint's commanded
  // position (confirmed on real hardware 2026-08-16: commanding head_joint
  // updated neck_joint's displayed value in RViz).
  if (joint_names_.size() >= 1) {
    if (auto v = extract_number(line, "tilt_pos")) pos_states_[0] = filter_position(0, *v);
    if (auto v = extract_number(line, "tilt_vel")) vel_states_[0] = *v;
  }
  if (joint_names_.size() >= 2) {
    if (auto v = extract_number(line, "pan_pos")) pos_states_[1] = filter_position(1, *v);
    if (auto v = extract_number(line, "pan_vel")) vel_states_[1] = *v;
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

  // index 0 (neck_joint) -> board's "tilt" motor, index 1 (head_joint) ->
  // board's "pan" motor - see matching swap note in read().
  double tilt_cmd = pos_commands_.size() >= 1 ? pos_commands_[0] : 0.0;
  double pan_cmd = pos_commands_.size() >= 2 ? pos_commands_[1] : 0.0;

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

PLUGINLIB_EXPORT_CLASS(openarm_hardware::OpenArm_v10HW, hardware_interface::SystemInterface)
PLUGINLIB_EXPORT_CLASS(openarm_hardware::HeadHW, hardware_interface::SystemInterface)
