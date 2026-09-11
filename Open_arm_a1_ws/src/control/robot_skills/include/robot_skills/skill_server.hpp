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
    /// Hosts the ExecuteSkill action server: dispatches goals to registered
    /// RobotSkill implementations by name, and gives those skills a shared
    /// execute_trajectory() helper for actually running a planned trajectory.
    class SkillServer
    {
    public:
        using ExecuteSkill = openarm_messages::action::ExecuteSkill;
        using GoalHandleExecuteSkill = rclcpp_action::ServerGoalHandle<ExecuteSkill>;

        /// Stores `node`/`planner`; call start() to advertise the action server.
        SkillServer(const std::shared_ptr<rclcpp::Node>& node,
                    const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner);
        /// Stops execution and joins execute_thread_/replan_thread_ (both
        /// capture `this`, so must be joined before destruction).
        ~SkillServer();

        /// Advertises robot_skills_server/execute_skill.
        bool start();
        /// Initializes `skill` (node/planner) and registers it under its name()
        /// for goal dispatch.
        void register_skill(const std::shared_ptr<RobotSkill>& skill);

        /// Runs `trajectory` on `group_name`'s controllers via
        /// TrajectoryExecutionManager. NORMAL mode pushes and blocks
        /// (executeAndWait()) until done. REALTIME mode starts execution
        /// non-blocking and spins a background thread that watches
        /// updated_request_/goal_changed_ for a mid-motion goal update,
        /// replanning and swapping the running trajectory when one arrives,
        /// until execution finishes.
        SkillResult execute_trajectory(
            const std::string& group_name,
            const moveit_msgs::msg::RobotTrajectory& trajectory,
            PlanningMode mode,
            const std::shared_ptr<GoalHandleExecuteSkill>& goal_handle,
            const SkillRequest& original_req);

    private:
        /// Action server goal callback: rejects unknown skill names, otherwise
        /// accepts and executes immediately.
        rclcpp_action::GoalResponse handle_goal(
            const rclcpp_action::GoalUUID& uuid,
            std::shared_ptr<const ExecuteSkill::Goal> goal);

        /// Action server cancel callback: stops execution_active_ and any
        /// in-flight TrajectoryExecutionManager execution.
        rclcpp_action::CancelResponse handle_cancel(
            const std::shared_ptr<GoalHandleExecuteSkill>& goal_handle);

        /// Action server accepted callback: joins any previous execute_thread_
        /// then launches execute_goal() on a new one.
        void handle_accepted(const std::shared_ptr<GoalHandleExecuteSkill>& goal_handle);
        /// Runs on execute_thread_: builds a SkillRequest from the goal,
        /// dispatches it to the matching registered skill's execute(), and
        /// reports the result back through `goal_handle`.
        void execute_goal(const std::shared_ptr<GoalHandleExecuteSkill>& goal_handle);

        std::shared_ptr<rclcpp::Node> node_;
        std::shared_ptr<motion_planner::MoveItCppPlannerManager> planner_;
        std::map<std::string, std::shared_ptr<RobotSkill>> skills_;
        
        rclcpp_action::Server<ExecuteSkill>::SharedPtr action_server_;

        // Goal execution thread — must be joined (never detached) because
        // it captures 'this'.  execute_thread_ is joined in the destructor
        // and re-joined in handle_accepted before launching the next goal.
        std::thread execute_thread_;

        // Realtime execution tracking variables
        std::atomic<bool> execution_active_{false};
        std::atomic<bool> goal_changed_{false};
        std::mutex req_mutex_;
        SkillRequest updated_request_;
        std::thread replan_thread_;
    };
}

#endif // __ROBOT_SKILLS_SKILL_SERVER_HPP__
