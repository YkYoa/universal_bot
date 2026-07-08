#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <hardware_interface/system_interface.hpp>
#include <rclcpp_lifecycle/state.hpp>

namespace openarm_hardware {

class OpenArm_v10HW : public hardware_interface::SystemInterface
{
public:
  OpenArm_v10HW();

  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams& params) override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State& previous_state) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State& previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State& previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time& time, const rclcpp::Duration& period) override;

  hardware_interface::return_type write(
    const rclcpp::Time& time, const rclcpp::Duration& period) override;

private:
  static constexpr std::size_t ARM_DOF = 7;

  bool parse_config();
  void generate_joint_names();

  // Params from URDF ros2_control <hardware><param ...>
  std::string can_interface_{"can0"};
  std::string arm_prefix_{""};     // e.g. "left_" / "right_" (already includes underscore)
  bool hand_{false};
  bool can_fd_{true};

  std::array<double, ARM_DOF> kp_{};
  std::array<double, ARM_DOF> kd_{};
  double gripper_kp_{0.0};
  double gripper_kd_{0.0};

  std::vector<std::string> joint_names_;

  std::vector<double> pos_commands_;
  std::vector<double> vel_commands_;
  std::vector<double> tau_commands_;

  std::vector<double> pos_states_;
  std::vector<double> vel_states_;
  std::vector<double> tau_states_;

  // PImpl so we can build without OpenArmCAN installed.
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace openarm_hardware

