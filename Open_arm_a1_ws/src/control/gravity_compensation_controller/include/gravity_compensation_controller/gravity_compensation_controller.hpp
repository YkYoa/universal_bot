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
 *
 *        Also adds velocity damping (`joint_damping_nm_per_rad_s`,
 *        tau_damp = -damping[i] * qd) plus a "virtual detent" position hold
 *        (`detent_kp_nm_per_rad`, tau_detent = kp[i] * (hold_position_[i] -
 *        q[i])) on top of the gravity feedforward, both ramped in/out with
 *        it. Needed because pure damping only slows a joint's drift toward
 *        its true gravity equilibrium, it never stops it - a joint whose
 *        gravity torque is ~0 across its whole range (mass near its own
 *        rotation axis - true for this rig's joint7, see 2026-08-17
 *        real-hardware investigation) has no equilibrium of its own to slow
 *        down TOWARD, so any small residual model error or friction
 *        imbalance just accumulates into slow, undamped drift forever.
 *
 *        hold_position_[i] is NOT a fixed setpoint: it continuously chases
 *        q[i] every cycle, but at a capped rate
 *        (`detent_max_slew_rad_s`) - fast intentional hand motion keeps the
 *        anchor pinned close (tau_detent stays small, doesn't fight the
 *        user), while ANY sustained drift, however slow, eventually
 *        outruns the anchor's slew cap and the growing position error
 *        supplies a restoring torque that arrests it. An earlier version
 *        used an instantaneous-velocity threshold to snap-freeze the
 *        anchor instead of slew-limiting it; that failed on real hardware
 *        (2026-08-17) for slow continuous drift that never actually
 *        dropped below the threshold, so the anchor chased it 1:1 forever
 *        and supplied ~zero restoring force - the whole point of this
 *        control is to catch exactly that case, so the rate cap (which
 *        can't be evaded by a drift simply being slow) replaces it
 *        entirely. */
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
  std::vector<double> joint_damping_;  // one entry per joint_names_[i]
  std::vector<double> detent_kp_;      // one entry per joint_names_[i]
  double detent_max_slew_{0.02};  // rad/s - how fast hold_position_ may chase q

  KDL::Chain kdl_chain_;
  std::unique_ptr<KDL::ChainDynParam> dyn_param_;
  KDL::JntArray q_;
  KDL::JntArray qd_;
  KDL::JntArray gravity_torques_;
  KDL::JntArray hold_position_;
  bool hold_initialized_{false};

  // Set by the ~/enable service callback (a non-realtime executor thread);
  // update() (the realtime control loop) only ever reads it and owns the
  // actual ramp progress (current_scale_) itself - no shared mutable ramp
  // state crosses threads.
  std::atomic<bool> enable_requested_{false};
  double current_scale_{0.0};

  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_service_;
};

}  // namespace gravity_compensation_controller
