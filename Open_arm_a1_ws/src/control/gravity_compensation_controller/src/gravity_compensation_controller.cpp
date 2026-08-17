#include "gravity_compensation_controller/gravity_compensation_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <sstream>

#include <kdl_parser/kdl_parser.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

namespace gravity_compensation_controller
{

controller_interface::CallbackReturn GravityCompensationController::on_init()
{
  auto node = get_node();
  joint_names_ = node->declare_parameter<std::vector<std::string>>("joints", std::vector<std::string>{});
  base_link_ = node->declare_parameter<std::string>("base_link", "openarm_body_link0");
  tip_link_ = node->declare_parameter<std::string>("tip_link", "");
  ramp_duration_sec_ = node->declare_parameter<double>("ramp_duration_sec", 2.0);
  joint_damping_ = node->declare_parameter<std::vector<double>>(
    "joint_damping_nm_per_rad_s", std::vector<double>{});
  detent_kp_ = node->declare_parameter<std::vector<double>>(
    "detent_kp_nm_per_rad", std::vector<double>{});
  detent_max_slew_ = node->declare_parameter<double>(
    "detent_max_slew_rad_s", 0.02);

  if (joint_names_.empty()) {
    RCLCPP_ERROR(node->get_logger(), "'joints' parameter is empty");
    return controller_interface::CallbackReturn::ERROR;
  }
  if (tip_link_.empty()) {
    RCLCPP_ERROR(node->get_logger(), "'tip_link' parameter is required (e.g. openarm_left_link7)");
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration GravityCompensationController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto& joint : joint_names_) {
    config.names.push_back(joint + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration GravityCompensationController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto& joint : joint_names_) {
    config.names.push_back(joint + "/position");
    config.names.push_back(joint + "/velocity");
  }
  return config;
}

controller_interface::CallbackReturn GravityCompensationController::on_configure(
  const rclcpp_lifecycle::State& /*previous_state*/)
{
  auto node = get_node();

  // robot_state_publisher publishes robot_description as a latched
  // (transient_local) topic - a fresh subscriber gets the last value
  // immediately, no need to wait for a live republish.
  //
  // Uses a separate, throwaway plain rclcpp::Node (not get_node(), the
  // controller's own LifecycleNode) for this one-shot fetch: the
  // controller's node is already owned by controller_manager's executor,
  // and adding it to a second local executor here throws ("Node ... has
  // already been added to an executor").
  auto urdf_waiter = std::make_shared<rclcpp::Node>(std::string(node->get_name()) + "_urdf_waiter");
  std::promise<std::string> urdf_promise;
  auto urdf_future = urdf_promise.get_future();
  auto sub = urdf_waiter->create_subscription<std_msgs::msg::String>(
    "/robot_description", rclcpp::QoS(1).transient_local(),
    [&urdf_promise](const std_msgs::msg::String::SharedPtr msg) { urdf_promise.set_value(msg->data); });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(urdf_waiter);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (urdf_future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
    if (std::chrono::steady_clock::now() > deadline) {
      RCLCPP_ERROR(node->get_logger(), "Timed out waiting for /robot_description");
      return controller_interface::CallbackReturn::ERROR;
    }
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  executor.remove_node(urdf_waiter);
  const std::string urdf_xml = urdf_future.get();

  KDL::Tree tree;
  if (!kdl_parser::treeFromString(urdf_xml, tree)) {
    RCLCPP_ERROR(node->get_logger(), "Failed to parse robot_description into a KDL tree");
    return controller_interface::CallbackReturn::ERROR;
  }
  if (!tree.getChain(base_link_, tip_link_, kdl_chain_)) {
    RCLCPP_ERROR(
      node->get_logger(), "No KDL chain from '%s' to '%s' - check base_link/tip_link params",
      base_link_.c_str(), tip_link_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  // Verify the chain's actual movable-joint order matches joint_names_
  // exactly - torques are written by index, so a silent mismatch here
  // would send the wrong torque to the wrong joint.
  std::vector<std::string> chain_joint_names;
  for (unsigned int i = 0; i < kdl_chain_.getNrOfSegments(); ++i) {
    const KDL::Joint& joint = kdl_chain_.getSegment(i).getJoint();
    if (joint.getType() != KDL::Joint::Fixed) {
      chain_joint_names.push_back(joint.getName());
    }
  }
  if (chain_joint_names != joint_names_) {
    RCLCPP_ERROR(
      node->get_logger(),
      "KDL chain '%s'->'%s' joint order does not match the 'joints' parameter - "
      "refusing to configure (would send torques to the wrong joints)",
      base_link_.c_str(), tip_link_.c_str());
    std::ostringstream chain_list, param_list;
    for (const auto& n : chain_joint_names) chain_list << n << " ";
    for (const auto& n : joint_names_) param_list << n << " ";
    RCLCPP_ERROR(node->get_logger(), "  chain order: %s", chain_list.str().c_str());
    RCLCPP_ERROR(node->get_logger(), "  joints param: %s", param_list.str().c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (joint_damping_.empty()) {
    joint_damping_.assign(joint_names_.size(), 0.0);
  } else if (joint_damping_.size() != joint_names_.size()) {
    RCLCPP_ERROR(
      node->get_logger(),
      "'joint_damping_nm_per_rad_s' has %zu entries, expected %zu (one per 'joints' entry)",
      joint_damping_.size(), joint_names_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (detent_kp_.empty()) {
    detent_kp_.assign(joint_names_.size(), 0.0);
  } else if (detent_kp_.size() != joint_names_.size()) {
    RCLCPP_ERROR(
      node->get_logger(),
      "'detent_kp_nm_per_rad' has %zu entries, expected %zu (one per 'joints' entry)",
      detent_kp_.size(), joint_names_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  dyn_param_ = std::make_unique<KDL::ChainDynParam>(kdl_chain_, KDL::Vector(0.0, 0.0, -9.81));
  q_ = KDL::JntArray(kdl_chain_.getNrOfJoints());
  qd_ = KDL::JntArray(kdl_chain_.getNrOfJoints());
  gravity_torques_ = KDL::JntArray(kdl_chain_.getNrOfJoints());
  hold_position_ = KDL::JntArray(kdl_chain_.getNrOfJoints());

  RCLCPP_INFO(
    node->get_logger(), "Configured gravity compensation for %zu joints ('%s' -> '%s'), ramp=%.1fs",
    joint_names_.size(), base_link_.c_str(), tip_link_.c_str(), ramp_duration_sec_);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn GravityCompensationController::on_activate(
  const rclcpp_lifecycle::State& /*previous_state*/)
{
  enable_requested_ = false;
  current_scale_ = 0.0;
  hold_initialized_ = false;

  auto node = get_node();
  enable_service_ = node->create_service<std_srvs::srv::SetBool>(
    "~/enable",
    [this](
      const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
      std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
      enable_requested_ = request->data;
      response->success = true;
      response->message = request->data ? "ramping up" : "ramping down";
    });

  RCLCPP_INFO(
    node->get_logger(),
    "Activated (disabled, writing 0 torque). Call ~/enable (std_srvs/SetBool) to ramp up - "
    "see the staged real-hardware validation procedure before enabling.");

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn GravityCompensationController::on_deactivate(
  const rclcpp_lifecycle::State& /*previous_state*/)
{
  enable_service_.reset();
  for (auto& command_interface : command_interfaces_) {
    static_cast<void>(command_interface.set_value<double>(0.0));
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type GravityCompensationController::update(
  const rclcpp::Time& /*time*/, const rclcpp::Duration& period)
{
  for (std::size_t i = 0; i < joint_names_.size(); ++i) {
    // state_interface_configuration() lists position then velocity per
    // joint, in that order, so they land at consecutive indices here.
    const auto position = state_interfaces_[2 * i].get_optional<double>();
    if (position.has_value()) {
      q_(i) = position.value();
    }
    // else: keep the previous q_(i) - a single missed read shouldn't zero
    // out the gravity model for that joint.
    const auto velocity = state_interfaces_[2 * i + 1].get_optional<double>();
    qd_(i) = velocity.value_or(0.0);
  }

  if (!hold_initialized_) {
    // First cycle after activation - anchor the detent to wherever the arm
    // actually is right now, not some stale/zero default.
    for (std::size_t i = 0; i < joint_names_.size(); ++i) hold_position_(i) = q_(i);
    hold_initialized_ = true;
  }
  {
    // Anchor chases q_ every cycle, capped at detent_max_slew_ rad/s - see
    // the class doc comment for why this replaced a velocity-threshold
    // snap-freeze (that failed to ever lock onto slow continuous drift).
    const double max_step = detent_max_slew_ * period.seconds();
    for (std::size_t i = 0; i < joint_names_.size(); ++i) {
      const double error = q_(i) - hold_position_(i);
      hold_position_(i) += std::clamp(error, -max_step, max_step);
    }
  }

  dyn_param_->JntToGravity(q_, gravity_torques_);

  // Ramp current_scale_ toward 0.0 or 1.0 depending on enable_requested_,
  // at a constant rate so a full ramp takes ramp_duration_sec_ regardless
  // of the controller's actual update rate.
  const double target = enable_requested_ ? 1.0 : 0.0;
  const double step = period.seconds() / std::max(ramp_duration_sec_, 0.01);
  if (current_scale_ < target) {
    current_scale_ = std::min(current_scale_ + step, target);
  } else if (current_scale_ > target) {
    current_scale_ = std::max(current_scale_ - step, target);
  }

  for (std::size_t i = 0; i < joint_names_.size(); ++i) {
    const double tau_detent = detent_kp_[i] * (hold_position_(i) - q_(i));
    const double tau = (gravity_torques_(i) + tau_detent - joint_damping_[i] * qd_(i)) * current_scale_;
    static_cast<void>(command_interfaces_[i].set_value<double>(tau));
  }

  return controller_interface::return_type::OK;
}

}  // namespace gravity_compensation_controller

PLUGINLIB_EXPORT_CLASS(
  gravity_compensation_controller::GravityCompensationController, controller_interface::ControllerInterface)
