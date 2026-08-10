#pragma once
// -----------------------------------------------------------------------------
// control_mode_probe.hpp
//
// What control mode did the arm actually come up in?
//
// This matters because the mode is not switchable at runtime: the damiao
// register is written once during hardware init from hardware_config.yaml's
// `control_mode` (robot_hardware_interface/src/v10/hardware_interface.cpp).
// In torque mode the arm ignores position commands outright - a trajectory
// replay looks like it ran and nothing moves. So every step declares the mode
// it needs and the FSM refuses the run up front instead of discovering it as
// silence.
//
// Two sources, cross-checked:
//   1. hardware_config.yaml's `control_mode` - the declared intent.
//   2. /controller_manager/list_controllers - which controllers are actually
//      active. A live gravity-comp controller means torque; a live arm
//      trajectory controller means position/mit.
//
// The controllers win when they disagree, because they describe the running
// system rather than a file that may have been edited since bringup.
// -----------------------------------------------------------------------------
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <controller_manager_msgs/srv/list_controllers.hpp>

namespace sequence_executor {

class ControlModeProbe
{
public:
  // `hardware_config_path` empty = resolve robot_hardware_interface's share
  // directory. A missing file is not an error; the probe just falls back to
  // the controller list.
  ControlModeProbe(rclcpp::Node::SharedPtr node, const std::string& hardware_config_path);

  // Queries both sources and caches the answer. Call once, at startup, before
  // anything starts publishing state.
  //
  // It blocks - it spins the node waiting on a service response - which is
  // exactly why it is separate from mode(). mode() is read on every FSM
  // transition, and doing a service round-trip there would stall the executor
  // mid-sequence for as long as controller_manager took to answer.
  void probe();

  // "position" | "mit" | "velocity" | "torque" | "unknown".
  // Pure getter, no I/O. Returns "unknown" until probe() has run - the mode
  // cannot change without a hardware restart, so one probe is enough.
  std::string mode() const;

private:
  std::string modeFromConfig() const;
  std::string modeFromControllers();

  rclcpp::Node::SharedPtr node_;
  std::string hardware_config_path_;
  std::string cached_;
  rclcpp::Client<controller_manager_msgs::srv::ListControllers>::SharedPtr controller_client_;
  rclcpp::Logger logger_;
};

}  // namespace sequence_executor
