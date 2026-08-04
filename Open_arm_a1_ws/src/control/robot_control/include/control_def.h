#pragma once

#include <cstdint>

namespace robot_control
{

/// Matches DriveCommand.srv's uint8 command field (communication/messages).
///
/// SET_HALT/RESET_HALT and SET_QUICK_STOP/RESET_QUICK_STOP are distinct at
/// the service/client API level (matching the CiA402-style halt-vs-quick-
/// stop distinction), but openarm_can/damiao motors only expose a single
/// immediate enable/disable primitive - no separate controlled-decel "halt"
/// exists at the hardware level. control_server maps both pairs onto
/// DriveControl's one quick-stop latch (see drive_control.h); the two
/// command names stay separate here so the ROS service surface doesn't
/// need to change if a real halt primitive becomes available later.
enum class DriveCommandType : uint8_t
{
  SERVO_ON = 0,
  SERVO_OFF = 1,
  SET_HALT = 2,
  RESET_HALT = 3,
  SET_QUICK_STOP = 4,
  RESET_QUICK_STOP = 5,
  HOME_DRIVES = 6,
};

}  // namespace robot_control
