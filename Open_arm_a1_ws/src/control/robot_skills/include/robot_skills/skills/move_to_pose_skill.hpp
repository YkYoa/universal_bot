#ifndef __ROBOT_SKILLS_MOVE_TO_POSE_SKILL_HPP__
#define __ROBOT_SKILLS_MOVE_TO_POSE_SKILL_HPP__

#include "robot_skills/skill_base.hpp"
#include "robot_skills/skill_server.hpp"

namespace robot_skills
{
    /// "move_to_pose": plans a Cartesian end-effector target (req.target_pose,
    /// optionally position-only) for req.arm and dispatches it via
    /// SkillServer::execute_trajectory().
    class MoveToPoseSkill : public RobotSkill
    {
    public:
        /// `server` is used to dispatch the planned trajectory for execution.
        MoveToPoseSkill(SkillServer* server) : server_(server) {}
        ~MoveToPoseSkill() override = default;

        bool initialize(
            const std::shared_ptr<rclcpp::Node>& node,
            const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner) override;
        
        SkillResult execute(
            const SkillRequest& req,
            const std::shared_ptr<rclcpp_action::ServerGoalHandle<openarm_messages::action::ExecuteSkill>>& goal_handle) override;
        
        std::string name() const override { return "move_to_pose"; }

    private:
        SkillServer* server_;
        std::shared_ptr<rclcpp::Node> node_;
        std::shared_ptr<motion_planner::MoveItCppPlannerManager> planner_;
    };
}

#endif // __ROBOT_SKILLS_MOVE_TO_POSE_SKILL_HPP__
