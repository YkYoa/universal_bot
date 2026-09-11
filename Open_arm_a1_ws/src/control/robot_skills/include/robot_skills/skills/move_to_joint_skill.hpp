#ifndef __ROBOT_SKILLS_MOVE_TO_JOINT_SKILL_HPP__
#define __ROBOT_SKILLS_MOVE_TO_JOINT_SKILL_HPP__

#include "robot_skills/skill_base.hpp"
#include "robot_skills/skill_server.hpp"

namespace robot_skills
{
    /// "move_to_joint": plans a single joint-space target (req.joint_targets)
    /// for req.arm and dispatches it via SkillServer::execute_trajectory().
    class MoveToJointSkill : public RobotSkill
    {
    public:
        /// `server` is used to dispatch the planned trajectory for execution.
        MoveToJointSkill(SkillServer* server) : server_(server) {}
        ~MoveToJointSkill() override = default;

        bool initialize(
            const std::shared_ptr<rclcpp::Node>& node,
            const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner) override;
        
        SkillResult execute(
            const SkillRequest& req,
            const std::shared_ptr<rclcpp_action::ServerGoalHandle<openarm_messages::action::ExecuteSkill>>& goal_handle) override;
        
        std::string name() const override { return "move_to_joint"; }

    private:
        SkillServer* server_;
        std::shared_ptr<rclcpp::Node> node_;
        std::shared_ptr<motion_planner::MoveItCppPlannerManager> planner_;
    };
}

#endif // __ROBOT_SKILLS_MOVE_TO_JOINT_SKILL_HPP__
