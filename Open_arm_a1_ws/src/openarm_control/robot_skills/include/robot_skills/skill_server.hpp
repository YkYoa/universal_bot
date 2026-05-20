#ifndef __ROBOT_SKILLS_SKILL_SERVER_HPP__
#define __ROBOT_SKILLS_SKILL_SERVER_HPP__

#include "robot_skills/skill_base.hpp"
#include "motion_planner/moveit_cpp_planner_manager.hpp"
#include "openarm_messages/action/execute_skill.hpp"
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <memory>
#include <string>
#include <map>
#include <thread>
#include <atomic>
#include <mutex>

namespace robot_skills
{
    class SkillServer
    {
    public:
        using ExecuteSkill = openarm_messages::action::ExecuteSkill;
        using GoalHandleExecuteSkill = rclcpp_action::ServerGoalHandle<ExecuteSkill>;

        SkillServer(const std::shared_ptr<rclcpp::Node>& node,
                    const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner);
        ~SkillServer();

        bool start();
        void register_skill(const std::shared_ptr<RobotSkill>& skill);

        // Helper to execute trajectories in both Normal (blocking) and Realtime modes
        SkillResult execute_trajectory(
            const std::string& group_name,
            const moveit_msgs::msg::RobotTrajectory& trajectory,
            PlanningMode mode,
            const std::shared_ptr<GoalHandleExecuteSkill>& goal_handle,
            const SkillRequest& original_req);

    private:
        rclcpp_action::GoalResponse handle_goal(
            const rclcpp_action::GoalUUID& uuid,
            std::shared_ptr<const ExecuteSkill::Goal> goal);
        
        rclcpp_action::CancelResponse handle_cancel(
            const std::shared_ptr<GoalHandleExecuteSkill>& goal_handle);
        
        void handle_accepted(const std::shared_ptr<GoalHandleExecuteSkill>& goal_handle);
        void execute_goal(const std::shared_ptr<GoalHandleExecuteSkill>& goal_handle);

        std::shared_ptr<rclcpp::Node> node_;
        std::shared_ptr<motion_planner::MoveItCppPlannerManager> planner_;
        std::map<std::string, std::shared_ptr<RobotSkill>> skills_;
        
        rclcpp_action::Server<ExecuteSkill>::SharedPtr action_server_;

        // Realtime execution tracking variables
        std::atomic<bool> execution_active_{false};
        std::atomic<bool> goal_changed_{false};
        std::mutex req_mutex_;
        SkillRequest updated_request_;
        std::thread replan_thread_;
    };
}

#endif // __ROBOT_SKILLS_SKILL_SERVER_HPP__
