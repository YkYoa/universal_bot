#include "sequence_executor/skill_client.hpp"

#include <algorithm>

namespace sequence_executor {

SkillClient::SkillClient(const rclcpp::Node::SharedPtr& node, const std::string& action_name)
: node_(node), logger_(rclcpp::get_logger("SkillClient"))
{
  client_ = rclcpp_action::create_client<ExecuteSkill>(node_, action_name);
}

void SkillClient::moveToJoint(
  const std::string& arm, const std::vector<double>& joint_targets, const std::string& planner_profile,
  double velocity_scaling, double acceleration_scaling, ResultCallback callback)
{
  ExecuteSkill::Goal goal;
  goal.skill_name = "move_to_joint";
  goal.arm = arm;
  goal.planner_profile = planner_profile;
  goal.planning_mode = "normal";
  goal.joint_targets = joint_targets;
  goal.velocity_override = velocity_scaling;
  goal.acceleration_override = acceleration_scaling;
  goal.position_only = false;
  sendGoal(std::move(goal), std::move(callback));
}

void SkillClient::moveToJointSequence(
  const std::string& arm, const std::vector<double>& joint_sequence, const std::string& planner_profile,
  double velocity_scaling, double acceleration_scaling, ResultCallback callback)
{
  ExecuteSkill::Goal goal;
  goal.skill_name = "move_to_joint_sequence";
  goal.arm = arm;
  goal.planner_profile = planner_profile;
  goal.planning_mode = "normal";
  goal.joint_sequence = joint_sequence;
  goal.velocity_override = velocity_scaling;
  goal.acceleration_override = acceleration_scaling;
  goal.position_only = false;
  sendGoal(std::move(goal), std::move(callback));
}

void SkillClient::moveToPose(
  const std::string& arm, const geometry_msgs::msg::PoseStamped& target, const std::string& planner_profile,
  double velocity_scaling, double acceleration_scaling, bool position_only, ResultCallback callback)
{
  ExecuteSkill::Goal goal;
  goal.skill_name = "move_to_pose";
  goal.arm = arm;
  goal.planner_profile = planner_profile;
  goal.planning_mode = "normal";
  goal.target_pose = target;
  goal.velocity_override = velocity_scaling;
  goal.acceleration_override = acceleration_scaling;
  goal.position_only = position_only;
  sendGoal(std::move(goal), std::move(callback));
}

void SkillClient::cartesianMove(
  const std::string& arm, const std::vector<geometry_msgs::msg::PoseStamped>& waypoints,
  const std::string& planner_profile, double velocity_scaling, double acceleration_scaling,
  bool position_only, ResultCallback callback)
{
  ExecuteSkill::Goal goal;
  goal.skill_name = "cartesian_move";
  goal.arm = arm;
  goal.planner_profile = planner_profile;
  goal.planning_mode = "normal";
  goal.waypoints = waypoints;
  goal.velocity_override = velocity_scaling;
  goal.acceleration_override = acceleration_scaling;
  goal.position_only = position_only;
  sendGoal(std::move(goal), std::move(callback));
}

void SkillClient::sendGoal(ExecuteSkill::Goal goal, ResultCallback callback)
{
  // robot_skills_node hosts this server behind a MoveItCpp stack (planning
  // scene, occupancy map monitor, etc.) that can take well over 5s to
  // finish initializing on a loaded machine - the old BT.CPP system never
  // hit this because its 50Hz tick loop kept retrying the connection
  // indefinitely across many ticks; this single-shot event-driven client
  // needs an explicitly generous wait instead of BT.CPP's incidental retry.
  if (!client_->wait_for_action_server(std::chrono::seconds(30))) {
    RCLCPP_ERROR(logger_, "ExecuteSkill action server not available after 30s");
    callback(false, "ExecuteSkill action server not available after 30s");
    return;
  }

  RCLCPP_INFO(
    logger_, "ExecuteSkill: %s | arm: %s | profile: %s", goal.skill_name.c_str(), goal.arm.c_str(),
    goal.planner_profile.c_str());

  rclcpp_action::Client<ExecuteSkill>::SendGoalOptions options;
  options.result_callback = [this, callback](const rclcpp_action::ClientGoalHandle<ExecuteSkill>::WrappedResult& result) {
    forgetGoal(result.goal_id);
    if (result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result->success) {
      callback(true, "");
      return;
    }
    if (result.code == rclcpp_action::ResultCode::CANCELED) {
      callback(false, "cancelled");
      return;
    }
    const std::string msg = result.result ? result.result->error_message : "action did not succeed";
    RCLCPP_WARN(logger_, "ExecuteSkill failed: %s", msg.c_str());
    callback(false, msg);
  };
  options.goal_response_callback =
    [this, callback](const rclcpp_action::ClientGoalHandle<ExecuteSkill>::SharedPtr& handle) {
      if (!handle) {
        RCLCPP_ERROR(logger_, "ExecuteSkill goal rejected");
        callback(false, "goal rejected");
        return;
      }
      active_goals_.push_back(handle);
    };

  client_->async_send_goal(goal, options);
}

bool SkillClient::cancelActiveGoal()
{
  if (active_goals_.empty()) {
    return false;
  }
  RCLCPP_INFO(logger_, "Cancelling %zu active ExecuteSkill goal(s)", active_goals_.size());
  // Copy first: async_cancel_goal can complete synchronously and reach back
  // into forgetGoal(), which mutates the vector being iterated.
  const auto goals = active_goals_;
  for (const auto& goal : goals) {
    client_->async_cancel_goal(goal);
  }
  return true;
}

void SkillClient::forgetGoal(const rclcpp_action::GoalUUID& goal_id)
{
  active_goals_.erase(
    std::remove_if(active_goals_.begin(), active_goals_.end(),
                   [&goal_id](const auto& handle) {
                     return !handle || handle->get_goal_id() == goal_id;
                   }),
    active_goals_.end());
}

}  // namespace sequence_executor
