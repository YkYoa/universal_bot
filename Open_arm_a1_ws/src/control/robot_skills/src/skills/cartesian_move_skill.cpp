#include "robot_skills/skills/cartesian_move_skill.hpp"

namespace robot_skills
{

bool CartesianMoveSkill::initialize(
    const std::shared_ptr<rclcpp::Node>& node,
    const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner)
{
    node_ = node;
    planner_ = planner;
    return true;
}

SkillResult CartesianMoveSkill::execute(
    const SkillRequest& req,
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<openarm_messages::action::ExecuteSkill>>& goal_handle)
{
    SkillResult result;

    if (!planner_) {
        result.success = false;
        result.error_message = "Planner is not initialized.";
        return result;
    }

    if (req.waypoints.empty()) {
        result.success = false;
        result.error_message = "Cartesian waypoints list is empty.";
        return result;
    }

    RCLCPP_INFO(node_->get_logger(), "[CartesianMoveSkill] Planning Cartesian move for arm: %s with %zu waypoints", 
                req.arm.c_str(), req.waypoints.size());

    // In a future release, we can use computeCartesianPath on the moveit_cpp RobotState.
    // For now, we will plan to the first waypoint in the list using the requested planner profile.
    planning_interface::PlannerRequest plan_req;
    plan_req.setGroupName(req.arm);
    plan_req.setProfileName(req.planner_profile.empty() ? "linear_approach" : req.planner_profile);
    plan_req.setWaypoints(req.waypoints);
    plan_req.setPositionOnly(req.position_only);

    planning_interface::PlanRequestParameters params;
    if (req.velocity_override > 0.0) {
        params.velocity_scaling = req.velocity_override;
        plan_req.setParameters(params);
    }

    auto plan_resp = planner_->plan(plan_req);
    result.planning_time_sec = plan_resp.planning_time;

    if (!plan_resp.success) {
        result.success = false;
        result.error_message = plan_resp.error_message;
        RCLCPP_ERROR(node_->get_logger(), "[CartesianMoveSkill] Cartesian planning failed: %s", result.error_message.c_str());
        return result;
    }

    return server_->execute_trajectory(req.arm, plan_resp.trajectory, req.mode, goal_handle, req);
}

} // namespace robot_skills
