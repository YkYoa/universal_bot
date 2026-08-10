#include <rclcpp/rclcpp.hpp>
#include "motion_planner/moveit_cpp_planner_manager.hpp"
#include "robot_skills/skill_server.hpp"
#include "robot_skills/skills/move_to_pose_skill.hpp"
#include "robot_skills/skills/move_to_named_pose_skill.hpp"
#include "robot_skills/skills/move_to_joint_skill.hpp"
#include "robot_skills/skills/move_to_joint_sequence_skill.hpp"
#include "robot_skills/skills/cartesian_move_skill.hpp"
#include "robot_skills/skills/gripper_skill.hpp"
#include "robot_skills/scene_manager.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    // Create node with options to load robot_description parameters
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);
    auto node = std::make_shared<rclcpp::Node>("robot_skills_node", node_options);

    RCLCPP_INFO(node->get_logger(), "Starting Robot Skills Node...");

    // Initialize planner manager
    auto planner = std::make_shared<motion_planner::MoveItCppPlannerManager>();
    if (!planner->initialize(node)) {
        RCLCPP_FATAL(node->get_logger(), "Failed to initialize MoveItCppPlannerManager.");
        return 1;
    }

    // Initialize and start skills server
    auto server = std::make_shared<robot_skills::SkillServer>(node, planner);

    // Register all skills
    server->register_skill(std::make_shared<robot_skills::MoveToPoseSkill>(server.get()));
    server->register_skill(std::make_shared<robot_skills::MoveToNamedPoseSkill>(server.get()));
    server->register_skill(std::make_shared<robot_skills::MoveToJointSkill>(server.get()));
    server->register_skill(std::make_shared<robot_skills::MoveToJointSequenceSkill>(server.get()));
    server->register_skill(std::make_shared<robot_skills::CartesianMoveSkill>(server.get()));
    server->register_skill(std::make_shared<robot_skills::GripperSkill>(server.get(), "open_gripper"));
    server->register_skill(std::make_shared<robot_skills::GripperSkill>(server.get(), "close_gripper"));

    if (!server->start()) {
        RCLCPP_FATAL(node->get_logger(), "Failed to start SkillServer.");
        return 1;
    }

    // Planning-scene edits (collision objects, grasp attach/detach, ACM) live
    // here because this process owns the PlanningSceneMonitor. Not fatal if it
    // fails to start - motion skills still work without scene editing.
    auto scene_manager = std::make_shared<robot_skills::SceneManager>(node, planner);
    scene_manager->start();

    // Use multi-threaded executor to support concurrent action goals and planning tasks
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    // Reset shared pointers before shutdown to ensure correct destruction order while ROS is still active
    scene_manager.reset();
    server.reset();
    planner.reset();
    node.reset();

    rclcpp::shutdown();
    return 0;
}
