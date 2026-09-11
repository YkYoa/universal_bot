#ifndef __ROBOT_SKILLS_GRIPPER_SKILL_HPP__
#define __ROBOT_SKILLS_GRIPPER_SKILL_HPP__

#include "robot_skills/skill_base.hpp"
#include "robot_skills/skill_server.hpp"

namespace robot_skills
{
    /// "open_gripper"/"close_gripper" (per `skill_name`): resolves req.arm to
    /// its left_gripper/right_gripper planning group, looks up that group's
    /// "open"/"close" SRDF group_state, and plans/executes a joint-target
    /// move to it via SkillServer::execute_trajectory().
    class GripperSkill : public RobotSkill
    {
    public:
        /// `skill_name` is "open_gripper" or "close_gripper" - what name() reports.
        GripperSkill(SkillServer* server, const std::string& skill_name)
          : server_(server), skill_name_(skill_name) {}
        ~GripperSkill() override = default;

        bool initialize(
            const std::shared_ptr<rclcpp::Node>& node,
            const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner) override;
        
        SkillResult execute(
            const SkillRequest& req,
            const std::shared_ptr<rclcpp_action::ServerGoalHandle<openarm_messages::action::ExecuteSkill>>& goal_handle) override;
        
        std::string name() const override { return skill_name_; }

    private:
        SkillServer* server_;
        std::string skill_name_; // "open_gripper" or "close_gripper"
        std::shared_ptr<rclcpp::Node> node_;
        std::shared_ptr<motion_planner::MoveItCppPlannerManager> planner_;
    };
}

#endif // __ROBOT_SKILLS_GRIPPER_SKILL_HPP__
