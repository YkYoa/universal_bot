#include "bt_executor/nodes/actions/plan_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"

namespace bt_executor {

bool PlanToPose::setGoal(Goal & goal)
{
  std::string arm;
  if (!getInput("arm", arm)) {
    RCLCPP_ERROR(logger(), "[PlanToPose] arm port not set");
    return false;
  }

  geometry_msgs::msg::PoseStamped target;
  if (!getInput("target_pose", target)) {
    RCLCPP_ERROR(logger(), "[PlanToPose] target_pose port not set");
    return false;
  }

  double vel = 0.3;
  bool position_only = false;
  getInput("velocity_scaling", vel);
  getInput("position_only", position_only);

  // Read config from blackboard
  std::string profile_name = "";
  std::string planning_mode = "normal";

  config().blackboard->get(BB_PLANNER_PROFILE, profile_name);
  config().blackboard->get(BB_PLANNING_MODE, planning_mode);

  // Set ExecuteSkill goal fields
  goal.skill_name = "move_to_pose";
  goal.arm = arm;
  goal.planner_profile = profile_name;
  goal.planning_mode = planning_mode;
  goal.target_pose = target;
  goal.velocity_override = vel;
  goal.position_only = position_only;

  RCLCPP_INFO(logger(), 
    "[PlanToPose] ExecuteSkill: %s | arm: %s | profile: %s | mode: %s | target: (%.3f, %.3f, %.3f)",
    goal.skill_name.c_str(), goal.arm.c_str(), goal.planner_profile.c_str(), 
    goal.planning_mode.c_str(), target.pose.position.x, target.pose.position.y, target.pose.position.z);

  return true;
}

BT::NodeStatus PlanToPose::onResultReceived(const WrappedResult & result)
{
  if (result.result->success) {
    config().blackboard->set(BB_STATUS_MSG, std::string("plan_to_pose: success"));
    
    // Save a dummy trajectory to output port for compatibility with existing BT trees
    moveit_msgs::msg::RobotTrajectory dummy_traj;
    setOutput("output_trajectory", dummy_traj);

    return BT::NodeStatus::SUCCESS;
  }

  const std::string msg = std::string("plan_to_pose skill failed: ") + result.result->error_message;
  config().blackboard->set(BB_STATUS_MSG, msg);
  config().blackboard->set(BB_REPLAN_NEEDED, true);
  RCLCPP_WARN(logger(), "[PlanToPose] %s", msg.c_str());
  return BT::NodeStatus::FAILURE;
}

BT::NodeStatus PlanToPose::onFailure(BT::ActionNodeErrorCode error)
{
  RCLCPP_ERROR(logger(), "[PlanToPose] action failed: %s", BT::toStr(error));
  config().blackboard->set(BB_REPLAN_NEEDED, true);
  return BT::NodeStatus::FAILURE;
}

}  // namespace bt_executor
