#pragma once

#include <memory>
#include <string>

namespace robot_control
{

/**
 * @brief Direct openarm_can wrapper for the low-level drive commands
 *        control_server exposes (servo on/off, quick-stop, homing) - the
 *        safety/hand layer that sits outside MoveIt's trajectory-execution
 *        path (robot_skills' ExecuteSkill action, ros2_control,
 *        robot_hardware_interface), not a replacement for it.
 *
 * WARNING: must not run concurrently with robot_hardware_interface's
 * ros2_control plugin on the same CAN interface. Both own an independent
 * openarm::can::socket::OpenArm instance and issue enable_all()/
 * disable_all() directly - running both at once means two uncoordinated
 * processes commanding the same motors, a real hardware-safety hazard, not
 * just a style issue. Use one or the other per CAN interface at a time.
 */
class DriveControl
{
public:
  explicit DriveControl(std::string can_interface, bool can_fd = true, bool hand = false);
  ~DriveControl();

  DriveControl(const DriveControl&) = delete;
  DriveControl& operator=(const DriveControl&) = delete;

  /// Enables all motors on this interface (openarm_can enable_all()).
  bool servoOn();

  /// Gracefully disables all motors (openarm_can disable_all()). Does not
  /// latch - servoOn() can be called again directly afterward.
  bool servoOff();

  /// Immediately disables all motors and latches a quick-stop fault state -
  /// servoOn() refuses until resetQuickStop() is called, mirroring an
  /// e-stop's "must be explicitly cleared" behavior.
  bool setQuickStop();

  /// Clears the quick-stop latch set by setQuickStop(). Does not itself
  /// re-enable motors - call servoOn() afterward.
  bool resetQuickStop();

  /// Zeroes the current position as each motor's reference zero
  /// (openarm_can set_zero_all()).
  bool homeDrives();

  /// Human-readable status/last-error string. openarm_can does not expose a
  /// queryable motor fault code today, so this currently reflects local
  /// servo/quick-stop latch state and the last operation's outcome, not a
  /// real motor-side fault code.
  std::string getErrorCode() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  std::string can_interface_;
  bool can_fd_;
  bool hand_;
  bool quick_stopped_{false};
  std::string last_error_{"ok"};
};

}  // namespace robot_control
