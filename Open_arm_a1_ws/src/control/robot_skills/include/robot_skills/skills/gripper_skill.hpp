#ifndef __ROBOT_SKILLS_GRIPPER_SKILL_HPP__
#define __ROBOT_SKILLS_GRIPPER_SKILL_HPP__

#include "robot_skills/skill_base.hpp"
#include "robot_skills/skill_server.hpp"

namespace robot_skills
{
    class GripperSkill : public RobotSkill
    {
    public:
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
