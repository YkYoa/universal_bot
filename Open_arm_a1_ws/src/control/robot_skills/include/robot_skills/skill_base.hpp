#ifndef __ROBOT_SKILLS_SKILL_BASE_HPP__
#define __ROBOT_SKILLS_SKILL_BASE_HPP__

#include "robot_skills/planning_mode.hpp"
#include "motion_planner/moveit_cpp_planner_manager.hpp"
#include "openarm_messages/action/execute_skill.hpp"
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <memory>
#include <string>

namespace robot_skills
{
    /// Everything one ExecuteSkill goal needs, decoded from the action goal
    /// into whichever target fields the requested skill actually reads
    /// (target_pose / named_pose / waypoints / joint_targets / joint_sequence
    /// are mutually exclusive in practice, per skill).
    struct SkillRequest
    {
        std::string arm;
        std::string planner_profile;
        PlanningMode mode = PlanningMode::NORMAL;
        
        // Target fields
        geometry_msgs::msg::PoseStamped target_pose;
        std::string named_pose;
        std::vector<geometry_msgs::msg::PoseStamped> waypoints;
        double velocity_override = 0.0;
        double acceleration_override = 0.0;
        bool position_only = false;
        std::vector<double> joint_targets;
        std::vector<double> joint_sequence;  // flat, stride-7 chunks
    };

    /// Outcome of one execute() call, reported back as the ExecuteSkill result.
    struct SkillResult
    {
        bool success = false;
        std::string error_message;
        double planning_time_sec = 0.0;
        double execution_time_sec = 0.0;
    };

    /// Interface every concrete skill (CartesianMoveSkill, GripperSkill, ...)
    /// implements; SkillServer dispatches ExecuteSkill goals to whichever
    /// skill matches the request by name().
    class RobotSkill
    {
    public:
        virtual ~RobotSkill() = default;
        /// One-time setup against the shared node/planner; returns false on
        /// unrecoverable init failure.
        virtual bool initialize(
            const std::shared_ptr<rclcpp::Node>& node,
            const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner) = 0;

        /// Runs the skill for `req`, reporting feedback/cancellation through
        /// `goal_handle`, and returns the final SkillResult.
        virtual SkillResult execute(
            const SkillRequest& req,
            const std::shared_ptr<rclcpp_action::ServerGoalHandle<openarm_messages::action::ExecuteSkill>>& goal_handle) = 0;

        /// The skill name SkillServer dispatches on (matches the ExecuteSkill
        /// goal's skill_name field).
        virtual std::string name() const = 0;
    };
}

#endif // __ROBOT_SKILLS_SKILL_BASE_HPP__
