#include "bt_executor/nodes/actions/plan_to_joint_target.hpp"
#include "rclcpp/rclcpp.hpp"

namespace bt_executor {

bool PlanToJointTarget::setGoal(Goal & goal)
{
  std::string arm;
  std::vector<double> joint_targets;
  if (!getInput("arm", arm)) {
    RCLCPP_ERROR(logger(), "[PlanToJointTarget] arm port not set");
    return false;
  }
  if (!getInput("joint_targets", joint_targets)) {
    RCLCPP_ERROR(logger(), "[PlanToJointTarget] joint_targets port not set");
    return false;
  }
  if (joint_targets.empty()) {
    RCLCPP_ERROR(logger(), "[PlanToJointTarget] joint_targets is empty");
    return false;
  }

  // Read config from blackboard
  std::string profile_name = "";
  std::string planning_mode = "normal";

  (void)config().blackboard->get(BB_PLANNER_PROFILE, profile_name);
  (void)config().blackboard->get(BB_PLANNING_MODE, planning_mode);

  double velocity_scaling = 0.0;
  getInput("velocity_scaling", velocity_scaling);
  double acceleration_scaling = 0.0;
  getInput("acceleration_scaling", acceleration_scaling);

  // Set ExecuteSkill goal fields
  goal.skill_name = "move_to_joint";
  goal.arm = arm;
  goal.planner_profile = profile_name;
  goal.planning_mode = planning_mode;
  goal.joint_targets = joint_targets;
  goal.velocity_override = velocity_scaling; // 0.0 = skill's default profile speed
  goal.acceleration_override = acceleration_scaling; // 0.0 = skill's default profile
  goal.position_only = false;

  std::string targets_str;
  for (size_t i = 0; i < joint_targets.size(); ++i) {
    if (i > 0) targets_str += ", ";
    targets_str += std::to_string(joint_targets[i]);
  }
  RCLCPP_INFO(logger(),
    "[PlanToJointTarget] ExecuteSkill: %s | arm: %s | profile: %s | mode: %s | velocity_scaling: %.2f | "
    "acceleration_scaling: %.2f | joint_targets: [%s]",
    goal.skill_name.c_str(), goal.arm.c_str(), goal.planner_profile.c_str(),
    goal.planning_mode.c_str(), velocity_scaling, acceleration_scaling, targets_str.c_str());

  return true;
}

BT::NodeStatus PlanToJointTarget::onResultReceived(const WrappedResult & result)
{
  if (result.result->success) {
    config().blackboard->set(BB_STATUS_MSG, std::string("plan_to_joint_target: success"));

    // Save a dummy trajectory to output port for compatibility with existing BT trees
    moveit_msgs::msg::RobotTrajectory dummy_traj;
    setOutput("output_trajectory", dummy_traj);

    return BT::NodeStatus::SUCCESS;
  }

  const std::string msg = std::string("plan_to_joint_target skill failed: ") + result.result->error_message;
  config().blackboard->set(BB_STATUS_MSG, msg);
  config().blackboard->set(BB_REPLAN_NEEDED, true);
  RCLCPP_WARN(logger(), "[PlanToJointTarget] %s", msg.c_str());
  return BT::NodeStatus::FAILURE;
}

BT::NodeStatus PlanToJointTarget::onFailure(BT::ActionNodeErrorCode error)
{
  RCLCPP_ERROR(logger(), "[PlanToJointTarget] action failed: %s", BT::toStr(error));
  config().blackboard->set(BB_REPLAN_NEEDED, true);
  return BT::NodeStatus::FAILURE;
}

}  // namespace bt_executor
