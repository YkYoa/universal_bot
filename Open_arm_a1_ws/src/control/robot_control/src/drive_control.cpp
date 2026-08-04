#include "drive_control.h"

#include <cstdint>
#include <vector>

#include <rclcpp/logging.hpp>

#if defined(ROBOT_CONTROL_HAS_OPENARMCAN)
#include <openarm/can/socket/openarm.hpp>
#include <openarm/damiao_motor/dm_motor_constants.hpp>
#endif

namespace robot_control
{

struct DriveControl::Impl
{
#if defined(ROBOT_CONTROL_HAS_OPENARMCAN)
  std::unique_ptr<openarm::can::socket::OpenArm> openarm;
#endif
};

DriveControl::DriveControl(std::string can_interface, bool can_fd, bool hand)
: impl_(std::make_unique<Impl>()),
  can_interface_(std::move(can_interface)),
  can_fd_(can_fd),
  hand_(hand)
{
#if defined(ROBOT_CONTROL_HAS_OPENARMCAN)
  impl_->openarm = std::make_unique<openarm::can::socket::OpenArm>(can_interface_, can_fd_);

  // Same physical arm as robot_hardware_interface's OpenArm_v10HW - motor
  // types/CAN IDs must be kept in sync with
  // robot_hardware_interface/src/v10/hardware_interface.cpp if the hardware
  // ever changes.
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
    impl_->openarm->init_gripper_motor(openarm::damiao_motor::MotorType::DM4310, 0x08, 0x18);
  }
#else
  RCLCPP_WARN(
    rclcpp::get_logger("DriveControl"),
    "OpenArmCAN not available (stub build). Install libopenarm-can-dev and rebuild.");
#endif
}

DriveControl::~DriveControl() = default;

bool DriveControl::servoOn()
{
  if (quick_stopped_) {
    last_error_ = "refused: quick-stop latched, call resetQuickStop() first";
    RCLCPP_ERROR(rclcpp::get_logger("DriveControl"), "%s", last_error_.c_str());
    return false;
  }

#if !defined(ROBOT_CONTROL_HAS_OPENARMCAN)
  last_error_ = "OpenArmCAN not available (stub build)";
  return false;
#else
  if (!impl_->openarm) {
    last_error_ = "OpenArm instance not initialized";
    return false;
  }
  impl_->openarm->set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);
  impl_->openarm->enable_all();
  impl_->openarm->recv_all();
  last_error_ = "ok";
  return true;
#endif
}

bool DriveControl::servoOff()
{
#if !defined(ROBOT_CONTROL_HAS_OPENARMCAN)
  last_error_ = "OpenArmCAN not available (stub build)";
  return false;
#else
  if (!impl_->openarm) {
    last_error_ = "OpenArm instance not initialized";
    return false;
  }
  impl_->openarm->disable_all();
  impl_->openarm->recv_all();
  last_error_ = "ok";
  return true;
#endif
}

bool DriveControl::setQuickStop()
{
  quick_stopped_ = true;

#if !defined(ROBOT_CONTROL_HAS_OPENARMCAN)
  last_error_ = "OpenArmCAN not available (stub build); quick-stop latched but nothing to disable";
  return false;
#else
  if (!impl_->openarm) {
    last_error_ = "OpenArm instance not initialized; quick-stop latched but nothing to disable";
    return false;
  }
  impl_->openarm->disable_all();
  impl_->openarm->recv_all();
  last_error_ = "quick-stopped";
  return true;
#endif
}

bool DriveControl::resetQuickStop()
{
  quick_stopped_ = false;
  last_error_ = "ok";
  return true;
}

bool DriveControl::homeDrives()
{
  if (quick_stopped_) {
    last_error_ = "refused: quick-stop latched, call resetQuickStop() first";
    RCLCPP_ERROR(rclcpp::get_logger("DriveControl"), "%s", last_error_.c_str());
    return false;
  }

#if !defined(ROBOT_CONTROL_HAS_OPENARMCAN)
  last_error_ = "OpenArmCAN not available (stub build)";
  return false;
#else
  if (!impl_->openarm) {
    last_error_ = "OpenArm instance not initialized";
    return false;
  }
  impl_->openarm->set_zero_all();
  impl_->openarm->recv_all();
  last_error_ = "ok";
  return true;
#endif
}

std::string DriveControl::getErrorCode() const
{
  if (quick_stopped_) {
    return "quick-stopped";
  }
  return last_error_;
}

}  // namespace robot_control
