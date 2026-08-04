#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <kdl/chain.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/jntarray.hpp>
#include <std_srvs/srv/set_bool.hpp>

namespace gravity_compensation_controller
{

/**
 * @brief Joint-space gravity compensation for free-drive/hand-guiding.
 *        Claims the `effort` command interface (only meaningful when
 *        OpenArm_v10HW's control_mode is "torque" - see hardware_interface.cpp,
 *        which forces kp=kd=0 in that mode so this controller's commanded
 *        torque is the only thing driving the motor). Static gravity torque
 *        only - no Coriolis/inertia compensation, no wrench estimation.
 *
 *        SAFETY: writes 0.0 torque until explicitly enabled via the
 *        ~/enable (std_srvs/SetBool) service, then ramps to full gravity
 *        torque over ramp_duration_sec instead of snapping on. Disabling
 *        ramps back down the same way. Activating this controller is never
 *        by itself sufficient to make the arm move - see the plan's Step 3
 *        staged validation procedure before ever enabling on real hardware.
 */
class GravityCompensationController : public controller_interface::ControllerInterface
{
public:
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::return_type update(const rclcpp::Time& time, const rclcpp::Duration& period) override;

private:
  std::vector<std::string> joint_names_;
  std::string base_link_;
  std::string tip_link_;
  double ramp_duration_sec_{2.0};

  KDL::Chain kdl_chain_;
  std::unique_ptr<KDL::ChainDynParam> dyn_param_;
  KDL::JntArray q_;
  KDL::JntArray gravity_torques_;

  // Set by the ~/enable service callback (a non-realtime executor thread);
  // update() (the realtime control loop) only ever reads it and owns the
  // actual ramp progress (current_scale_) itself - no shared mutable ramp
  // state crosses threads.
  std::atomic<bool> enable_requested_{false};
  double current_scale_{0.0};

  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_service_;
};

}  // namespace gravity_compensation_controller
