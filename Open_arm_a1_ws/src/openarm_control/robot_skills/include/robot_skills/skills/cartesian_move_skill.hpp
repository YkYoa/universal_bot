#ifndef __ROBOT_SKILLS_CARTESIAN_MOVE_SKILL_HPP__
#define __ROBOT_SKILLS_CARTESIAN_MOVE_SKILL_HPP__

#include "robot_skills/skill_base.hpp"
#include "robot_skills/skill_server.hpp"

namespace robot_skills
{
    class CartesianMoveSkill : public RobotSkill
    {
    public:
        CartesianMoveSkill(SkillServer* server) : server_(server) {}
        ~CartesianMoveSkill() override = default;

        bool initialize(
            const std::shared_ptr<rclcpp::Node>& node,
            const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner) override;
        
        SkillResult execute(
            const SkillRequest& req,
            const std::shared_ptr<rclcpp_action::ServerGoalHandle<openarm_messages::action::ExecuteSkill>>& goal_handle) override;
        
        std::string name() const override { return "cartesian_move"; }

    private:
        SkillServer* server_;
        std::shared_ptr<rclcpp::Node> node_;
        std::shared_ptr<motion_planner::MoveItCppPlannerManager> planner_;
    };
}

#endif // __ROBOT_SKILLS_CARTESIAN_MOVE_SKILL_HPP__
