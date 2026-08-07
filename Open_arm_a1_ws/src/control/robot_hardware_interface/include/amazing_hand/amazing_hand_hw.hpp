#pragma once

#include <memory>
#include <string>
#include <vector>

#include <hardware_interface/system_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>

#include "amazing_hand_kinematics/amazing_hand_kinematics.hpp"
#include "v10/visibility.hpp"

namespace openarm_hardware {

/**
 * @brief ros2_control hardware interface for the amazing_hand end effector
 *        (one instance per side in a bimanual robot_description, or one
 *        for the standalone build).
 *
 * Replaces the topic_based_ros2_control + hand_kinematics_node.py bridge:
 * the closed-loop rod/gimbal linkage solve
 * (control/amazing_hand_kinematics::HandSolver, a structural C++ port of
 * hand_kinematics_node.py's math) runs directly in read(), in-process, on
 * whatever's currently in the command interfaces - no topics, no separate
 * node, no echo-race.
 *
 * The 8 alias command joints (j11-j14 yaw, j21-j24 flex) are exactly what
 * the URDF's <ros2_control> block declares (info_.joints) - unchanged from
 * today. The passive joints HandSolver also reports (rod/gimbal mechanism
 * links not commandable, previously published via a side-channel topic)
 * are exported here as proper state-only interfaces instead, so
 * joint_state_broadcaster picks them up the normal way.
 *
 * Needs the full robot_description URDF (for FK across the hand's whole
 * subtree) that HardwareInfo does NOT provide (HardwareInfo::original_xml
 * is only this component's own <ros2_control> XML fragment) - fetched via a
 * SyncParametersClient against robot_state_publisher. This happens in
 * on_init(), not on_configure(): the resource manager calls
 * export_state_interfaces() right after on_init() returns, so the passive
 * joint list (which depends on the solver, which depends on the fetched
 * URDF) must already be known by then.
 */
class AmazingHandHW : public hardware_interface::SystemInterface
{
public:
  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams& params) override;

  OPENARM_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;

  OPENARM_HARDWARE_PUBLIC
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  OPENARM_HARDWARE_PUBLIC
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  OPENARM_HARDWARE_PUBLIC
  hardware_interface::return_type read(const rclcpp::Time& time, const rclcpp::Duration& period) override;

  OPENARM_HARDWARE_PUBLIC
  hardware_interface::return_type write(const rclcpp::Time& time, const rclcpp::Duration& period) override;

private:
  bool parse_config();
  /// Blocks (bounded by robot_description_timeout_s_) until robot_description
  /// is fetched from robot_state_publisher via a SyncParametersClient on an
  /// internal node added to the controller_manager's executor.
  bool fetchRobotDescription(const rclcpp::Executor::WeakPtr& executor, std::string& out_urdf);

  std::string link_prefix_;
  std::string alias_prefix_;
  std::string command_space_{"knuckle"};
  double robot_description_timeout_s_{10.0};

  // Alias command joints, from info_.joints (URDF-declared, unchanged from
  // today's topic_based_ros2_control wiring).
  std::vector<std::string> command_joint_names_;
  std::vector<double> command_positions_;
  // Parsed from each joint's command_interfaces[].initial_value in
  // parse_config() - this plugin's manual export_command_interfaces()
  // override bypasses the framework's automatic initial_value seeding, so
  // on_init() applies it to command_positions_ itself.
  std::vector<double> command_initial_positions_;

  // command_joint_names_ followed by HandSolver::allMovable() (passive,
  // state-only) - what export_state_interfaces() actually exports.
  std::vector<std::string> state_joint_names_;
  std::vector<double> state_positions_;

  std::unique_ptr<amazing_hand_kinematics::HandSolver> solver_;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_{rclcpp::get_logger("AmazingHandHW")};
};

}  // namespace openarm_hardware
