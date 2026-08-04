#include "robot_skills/skills/move_to_pose_skill.hpp"

namespace robot_skills
{

bool MoveToPoseSkill::initialize(
    const std::shared_ptr<rclcpp::Node>& node,
    const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner)
{
    node_ = node;
    planner_ = planner;
    return true;
}

SkillResult MoveToPoseSkill::execute(
    const SkillRequest& req,
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<openarm_messages::action::ExecuteSkill>>& goal_handle)
{
    SkillResult result;

    if (!planner_) {
        result.success = false;
        result.error_message = "Planner is not initialized.";
        return result;
    }

    RCLCPP_INFO(node_->get_logger(), "[MoveToPoseSkill] Planning to pose for arm: %s", req.arm.c_str());

    // 1. Build planning request
    planning_interface::PlannerRequest plan_req;
    plan_req.setGroupName(req.arm);
    plan_req.setProfileName(req.planner_profile);
    plan_req.setTargetPose(req.target_pose);
    plan_req.setPositionOnly(req.position_only);

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
        RCLCPP_ERROR(node_->get_logger(), "[MoveToPoseSkill] Planning failed: %s", result.error_message.c_str());
        return result;
    }

    RCLCPP_INFO(node_->get_logger(), "[MoveToPoseSkill] Planning succeeded. Executing trajectory...");

    // 3. Dispatch execution to server helper
    return server_->execute_trajectory(req.arm, plan_resp.trajectory, req.mode, goal_handle, req);
}

} // namespace robot_skills
