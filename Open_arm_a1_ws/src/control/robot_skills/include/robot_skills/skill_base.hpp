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
    };

    struct SkillResult
    {
        bool success = false;
        std::string error_message;
        double planning_time_sec = 0.0;
        double execution_time_sec = 0.0;
    };

    class RobotSkill
    {
    public:
        virtual ~RobotSkill() = default;
        virtual bool initialize(
            const std::shared_ptr<rclcpp::Node>& node,
            const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner) = 0;
        
        virtual SkillResult execute(
            const SkillRequest& req,
            const std::shared_ptr<rclcpp_action::ServerGoalHandle<openarm_messages::action::ExecuteSkill>>& goal_handle) = 0;
        
        virtual std::string name() const = 0;
    };
}

#endif // __ROBOT_SKILLS_SKILL_BASE_HPP__
