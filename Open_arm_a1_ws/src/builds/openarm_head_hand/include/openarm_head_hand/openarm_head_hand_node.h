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
#include <mutex>

namespace openarm_head_hand
{
    class OpenArmHeadHandNode : public rclcpp::Node
    {
    public:
        using ExecuteSkill = openarm_messages::action::ExecuteSkill;
        using GoalHandleExecuteSkill = rclcpp_action::ClientGoalHandle<ExecuteSkill>;

        explicit OpenArmHeadHandNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
        virtual ~OpenArmHeadHandNode();

    private:
        void loadAndSketchSequence();
        std::vector<double> getWaypoint(const std::string & key) const;

        void runConsole();
        void printMenu();
        bool sendSkillGoal(const ExecuteSkill::Goal& goal);

        // Core movement helpers
        bool moveGroupToJoints(const std::string& group, const std::string& key);
        bool moveGroupToNamedPose(const std::string& group, const std::string& named_pose);

        // Sequences
        bool runHeadHome();
        bool runHand(const std::string& side, const std::string& named_pose);

        void handleCommand(const std::string& cmd);
        void executeSingleCommand(const std::string& cmd);

        std::string sequence_yaml_path_;
        YAML::Node config_;

        rclcpp_action::Client<ExecuteSkill>::SharedPtr action_client_;

        rclcpp::Subscription<std_msgs::msg::String>::SharedPtr command_sub_;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;

        std::thread console_thread_;
        bool running_;
        std::mutex state_mutex_;
        std::atomic<bool> cancel_flag_{false};
        std::atomic<bool> executing_{false};
        std::thread execution_thread_;
    };
} // namespace openarm_head_hand
