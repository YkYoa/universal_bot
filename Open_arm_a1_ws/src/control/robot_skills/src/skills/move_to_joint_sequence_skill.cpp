#include "robot_skills/skills/move_to_joint_sequence_skill.hpp"

namespace robot_skills
{

bool MoveToJointSequenceSkill::initialize(
    const std::shared_ptr<rclcpp::Node>& node,
    const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner)
{
    node_ = node;
    planner_ = planner;
    return true;
}

SkillResult MoveToJointSequenceSkill::execute(
    const SkillRequest& req,
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<openarm_messages::action::ExecuteSkill>>& goal_handle)
{
    SkillResult result;

    if (!planner_ || !planner_->getMoveItCpp()) {
        result.success = false;
        result.error_message = "Planner/MoveItCpp is not initialized.";
        return result;
    }

    RCLCPP_INFO(node_->get_logger(), "[MoveToJointSequenceSkill] Planning joint sequence for arm: %s", req.arm.c_str());

    if (req.joint_sequence.empty() || req.joint_sequence.size() % 7 != 0) {
        result.success = false;
        result.error_message = "joint_sequence must be a non-empty, flat array with a multiple-of-7 length.";
        RCLCPP_ERROR(node_->get_logger(), "[MoveToJointSequenceSkill] %s", result.error_message.c_str());
        return result;
    }
    if (req.joint_sequence.size() < 14) {
        result.success = false;
        result.error_message = "joint_sequence needs at least 2 waypoints (use move_to_joint for a single target).";
        RCLCPP_ERROR(node_->get_logger(), "[MoveToJointSequenceSkill] %s", result.error_message.c_str());
        return result;
    }

    std::vector<std::vector<double>> waypoints;
    waypoints.reserve(req.joint_sequence.size() / 7);
    for (size_t i = 0; i < req.joint_sequence.size(); i += 7) {
        waypoints.emplace_back(req.joint_sequence.begin() + static_cast<long>(i), req.joint_sequence.begin() + static_cast<long>(i + 7));
    }

    // 1. Build planning request
    planning_interface::PlannerRequest plan_req;
    plan_req.setGroupName(req.arm);
    plan_req.setProfileName(req.planner_profile);
    plan_req.setJointSequence(waypoints);

    // Apply scaling overrides
    planning_interface::PlanRequestParameters params;
    if (req.velocity_override > 0.0 || req.acceleration_override > 0.0) {
        params.velocity_scaling = req.velocity_override;
        params.acceleration_scaling = req.acceleration_override;
        plan_req.setParameters(params);
    }

    // 2. Perform planning
    auto plan_resp = planner_->plan(plan_req);
    result.planning_time_sec = plan_resp.planning_time;

    if (!plan_resp.success) {
        result.success = false;
        result.error_message = plan_resp.error_message;
        RCLCPP_ERROR(node_->get_logger(), "[MoveToJointSequenceSkill] Planning failed: %s", result.error_message.c_str());
        return result;
    }

    RCLCPP_INFO(node_->get_logger(), "[MoveToJointSequenceSkill] Planning succeeded (%zu waypoints). Executing trajectory...",
                waypoints.size());

    // 3. Dispatch execution to server helper
    return server_->execute_trajectory(req.arm, plan_resp.trajectory, req.mode, goal_handle, req);
}

} // namespace robot_skills
