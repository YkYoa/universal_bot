#include "v10/openarm_v10_hardware.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

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

  RCLCPP_INFO(
    rclcpp::get_logger("OpenArm_v10HW"),
    "Config: can=%s prefix=%s hand=%s can_fd=%s ee_type=%s",
    can_interface_.c_str(),
    arm_prefix_.c_str(),
    hand_ ? "true" : "false",
    can_fd_ ? "true" : "false",
    ee_type_.c_str());

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
  return (joint_value / GRIPPER_JOINT_OPEN) * GRIPPER_MOTOR_OPEN;
}

double OpenArm_v10HW::motor_radians_to_joint(double motor_radians) const
{
  if (ee_type_ == "pinch_gripper") {
    return motor_radians;
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

  impl_->openarm->init_arm_motors(motor_types, send_ids, recv_ids);

  if (hand_) {
    impl_->openarm->init_gripper_motor(
      openarm::damiao_motor::MotorType::DM4310, 0x08, 0x18);
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

  std::vector<openarm::damiao_motor::MITParam> arm_params;
  for (std::size_t i = 0; i < ARM_DOF; ++i) {
    arm_params.push_back({kp_[i], kd_[i], 0.0, 0.0, 0.0});
  }
  impl_->openarm->get_arm().mit_control_all(arm_params);

  if (hand_) {
    impl_->openarm->get_gripper().mit_control_all(
      {{gripper_kp_, gripper_kd_, GRIPPER_MOTOR_CLOSED, 0.0, 0.0}});
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  impl_->openarm->recv_all();
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
    for (int i = 0; i < 3; ++i) {
      impl_->openarm->disable_all();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
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

  std::vector<openarm::damiao_motor::MITParam> arm_params;
  arm_params.reserve(ARM_DOF);
  for (std::size_t i = 0; i < ARM_DOF; ++i) {
    arm_params.push_back({kp_[i], kd_[i], pos_commands_[i], vel_commands_[i], tau_commands_[i]});
  }
  impl_->openarm->get_arm().mit_control_all(arm_params);

  if (hand_ && joint_names_.size() > ARM_DOF) {
    const double motor_cmd = joint_to_motor_radians(pos_commands_[ARM_DOF]);
    impl_->openarm->get_gripper().mit_control_all(
      {{gripper_kp_, gripper_kd_, motor_cmd, 0.0, 0.0}});
  }

  impl_->openarm->recv_all(100);
#endif

  return hardware_interface::return_type::OK;
}

}  // namespace openarm_hardware

PLUGINLIB_EXPORT_CLASS(openarm_hardware::OpenArm_v10HW, hardware_interface::SystemInterface)
