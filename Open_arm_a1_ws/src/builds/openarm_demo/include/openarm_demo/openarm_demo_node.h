#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <openarm_messages/action/execute_skill.hpp>
#include <std_msgs/msg/string.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>

namespace openarm_demo
{
    enum class DemoState {
        IDLE,
        WAVING_JOINT,
        WAVING_CARTESIAN,
        GREETING_JOINT,
        GREETING_CARTESIAN,
        HOMING,
        OPENING_GRIPPER,
        CLOSING_GRIPPER
    };

    class OpenArmDemoNode : public rclcpp::Node
    {
    public:
        using ExecuteSkill = openarm_messages::action::ExecuteSkill;
        using GoalHandleExecuteSkill = rclcpp_action::ClientGoalHandle<ExecuteSkill>;

        explicit OpenArmDemoNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
        virtual ~OpenArmDemoNode();

        // Member function to retrieve a waypoint parameter
        std::vector<double> getWaypoint(const std::string & key) const;

    private:
        void loadAndSketchSequence();
        void runConsole();
        void printMenu();
        bool sendSkillGoal(const ExecuteSkill::Goal& goal);

        // Core movement helpers
        bool moveArmToJoints(const std::string& arm, const std::string& key);
        bool moveArmToPose(const std::string& arm, const std::string& key);
        bool controlGripper(const std::string& arm, bool open);

        // State Machine Action Handlers
        void runHomingSequence();
        void runWavingJointSequence();
        void runWavingCartesianSequence();
        void runGreetingJointSequence();
        void runGreetingCartesianSequence();
        void runGripperSequence(bool open);

        void handleCommand(const std::string& cmd);
        void executeSingleCommand(const std::string& cmd);

        std::string sequence_yaml_path_;
        YAML::Node config_;

        // ROS 2 Action client for robot_skills execution
        rclcpp_action::Client<ExecuteSkill>::SharedPtr action_client_;

        // Command subscriber for non-blocking trigger support
        rclcpp::Subscription<std_msgs::msg::String>::SharedPtr command_sub_;

        // Status publisher to publish execution status
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;

        // Threading & state control
        std::thread console_thread_;
        bool running_;
        std::mutex state_mutex_;
        DemoState current_state_;
        std::atomic<bool> cancel_flag_{false};
        std::atomic<bool> executing_{false};
        std::thread execution_thread_;
    };

    // Global helper function to retrieve waypoints from anywhere
    std::vector<double> getGlobalWaypoint(const std::string & key);
} // namespace openarm_demo