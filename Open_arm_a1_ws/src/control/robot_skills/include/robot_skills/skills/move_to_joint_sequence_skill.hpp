#ifndef __ROBOT_SKILLS_MOVE_TO_JOINT_SEQUENCE_SKILL_HPP__
#define __ROBOT_SKILLS_MOVE_TO_JOINT_SEQUENCE_SKILL_HPP__

#include "robot_skills/skill_base.hpp"
#include "robot_skills/skill_server.hpp"

namespace robot_skills
{
    /// "move_to_joint_sequence": plans a flat, stride-DOF list of joint-space
    /// waypoints (req.joint_sequence, at least 2) as ONE continuous blended
    /// trajectory (see motion_planner::MoveItCppPlannerManager's
    /// getJointSequence() branch) instead of N independent
    /// plan+execute+full-stop cycles like MoveToJointSkill.
    class MoveToJointSequenceSkill : public RobotSkill
    {
    public:
        /// `server` is used to dispatch the planned trajectory for execution.
        MoveToJointSequenceSkill(SkillServer* server) : server_(server) {}
        ~MoveToJointSequenceSkill() override = default;

        bool initialize(
            const std::shared_ptr<rclcpp::Node>& node,
            const std::shared_ptr<motion_planner::MoveItCppPlannerManager>& planner) override;

        SkillResult execute(
            const SkillRequest& req,
            const std::shared_ptr<rclcpp_action::ServerGoalHandle<openarm_messages::action::ExecuteSkill>>& goal_handle) override;

        std::string name() const override { return "move_to_joint_sequence"; }

    private:
        SkillServer* server_;
        std::shared_ptr<rclcpp::Node> node_;
        std::shared_ptr<motion_planner::MoveItCppPlannerManager> planner_;
    };
}

#endif // __ROBOT_SKILLS_MOVE_TO_JOINT_SEQUENCE_SKILL_HPP__
