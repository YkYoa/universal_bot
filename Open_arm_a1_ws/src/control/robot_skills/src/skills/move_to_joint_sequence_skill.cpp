#include "robot_skills/skills/move_to_joint_sequence_skill.hpp"
#include <moveit/robot_state/robot_state.hpp>

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

    auto robot_model = planner_->getMoveItCpp()->getRobotModel();
    const auto* jmg = robot_model->getJointModelGroup(req.arm);
    if (!jmg) {
        result.success = false;
        result.error_message = "Unknown planning group: " + req.arm;
        RCLCPP_ERROR(node_->get_logger(), "[MoveToJointSequenceSkill] %s", result.error_message.c_str());
        return result;
    }

    if (req.joint_sequence.size() < 2) {
        result.success = false;
        result.error_message = "joint_sequence needs at least 2 waypoints (use move_to_joint for a single target).";
        RCLCPP_ERROR(node_->get_logger(), "[MoveToJointSequenceSkill] %s", result.error_message.c_str());
        return result;
    }

    // Resolve each waypoint's name->position pairs against the live robot
    // model independently (same pattern as MoveToJointSkill) - a waypoint
    // recorded under one ee_type (e.g. wavePoses' 7-value openarm_hand data)
    // still resolves correctly against a group booted under another, since
    // there is no positional stride/DOF to get wrong.
    std::vector<std::vector<double>> waypoints;
    waypoints.reserve(req.joint_sequence.size());
    for (std::size_t w = 0; w < req.joint_sequence.size(); ++w) {
        const auto& js = req.joint_sequence[w];
        if (js.name.size() != js.position.size()) {
            result.success = false;
            result.error_message = "waypoint " + std::to_string(w) + ": name/position length mismatch (" +
                std::to_string(js.name.size()) + " names, " + std::to_string(js.position.size()) + " positions).";
            RCLCPP_ERROR(node_->get_logger(), "[MoveToJointSequenceSkill] %s", result.error_message.c_str());
            return result;
        }

        moveit::core::RobotState wp_state(robot_model);
        wp_state.setToDefaultValues();
        std::size_t matched = 0;
        for (std::size_t i = 0; i < js.name.size(); ++i) {
            if (robot_model->hasJointModel(js.name[i])) {
                wp_state.setVariablePosition(js.name[i], js.position[i]);
                ++matched;
            }
        }
        if (matched == 0) {
            result.success = false;
            result.error_message = "waypoint " + std::to_string(w) +
                ": none of its joint names exist on the live robot model.";
            RCLCPP_ERROR(node_->get_logger(), "[MoveToJointSequenceSkill] %s", result.error_message.c_str());
            return result;
        }
        wp_state.update();

        std::vector<double> group_positions;
        wp_state.copyJointGroupPositions(jmg, group_positions);
        waypoints.push_back(std::move(group_positions));
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
