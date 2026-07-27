#include <openarm_head_hand/openarm_head_hand_node.h>
#include <common/yaml_parser.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <iostream>
#include <sstream>
#include <chrono>

namespace openarm_head_hand
{
    auto LOG = rclcpp::get_logger("openarm_head_hand_node");

    OpenArmHeadHandNode::OpenArmHeadHandNode(const rclcpp::NodeOptions & options)
    : Node("openarm_head_hand_node", options), running_(true)
    {
        std::string default_yaml_path =
            ament_index_cpp::get_package_share_directory("openarm_head_hand") + "/config/sequence.yaml";

        this->declare_parameter<std::string>("sequence_yaml_path", default_yaml_path);
        this->get_parameter("sequence_yaml_path", sequence_yaml_path_);

        RCLCPP_INFO(LOG, "Initializing OpenArm Head+Hand Node");
        RCLCPP_INFO(LOG, "Loading sequence config from: %s", sequence_yaml_path_.c_str());

        loadAndSketchSequence();

        action_client_ = rclcpp_action::create_client<ExecuteSkill>(this, "robot_skills_server/execute_skill");

        command_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/openarm_head_hand/command", 10,
            [this](const std_msgs::msg::String::SharedPtr msg) {
                RCLCPP_INFO(LOG, "Received topic command: '%s'", msg->data.c_str());
                std::thread(&OpenArmHeadHandNode::handleCommand, this, msg->data).detach();
            });

        status_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/openarm_head_hand/status", 10);

        console_thread_ = std::thread(&OpenArmHeadHandNode::runConsole, this);
    }

    OpenArmHeadHandNode::~OpenArmHeadHandNode()
    {
        running_ = false;
        if (console_thread_.joinable()) {
            console_thread_.join();
        }
        RCLCPP_INFO(LOG, "OpenArm Head+Hand Node stopped.");
    }

    void OpenArmHeadHandNode::loadAndSketchSequence()
    {
        config_ = common::loadAndSketchSequence(this->get_logger(), sequence_yaml_path_);
    }

    std::vector<double> OpenArmHeadHandNode::getWaypoint(const std::string & key) const
    {
        return common::getWaypointFromConfig(config_, key);
    }

    bool OpenArmHeadHandNode::sendSkillGoal(const ExecuteSkill::Goal& goal)
    {
        if (!action_client_->wait_for_action_server(std::chrono::seconds(5))) {
            RCLCPP_ERROR(LOG, "Action server 'robot_skills_server/execute_skill' not available!");
            return false;
        }

        auto send_goal_options = rclcpp_action::Client<ExecuteSkill>::SendGoalOptions();
        auto goal_handle_future = action_client_->async_send_goal(goal, send_goal_options);

        if (goal_handle_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            RCLCPP_ERROR(LOG, "Action goal submission timed out!");
            return false;
        }

        auto goal_handle = goal_handle_future.get();
        if (!goal_handle) {
            RCLCPP_ERROR(LOG, "Goal was rejected by robot_skills_server!");
            return false;
        }

        auto result_future = action_client_->async_get_result(goal_handle);
        while (rclcpp::ok() && running_ && !cancel_flag_) {
            if (result_future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
                break;
            }
        }

        if (cancel_flag_) {
            RCLCPP_INFO(LOG, "Cancellation requested. Cancelling active goal on server...");
            action_client_->async_cancel_goal(goal_handle);
            return false;
        }

        if (!rclcpp::ok() || !running_) {
            return false;
        }

        auto wrapped_result = result_future.get();
        if (wrapped_result.code != rclcpp_action::ResultCode::SUCCEEDED) {
            RCLCPP_ERROR(LOG, "Skill execution failed or was aborted! Code: %d", static_cast<int>(wrapped_result.code));
            return false;
        }

        if (!wrapped_result.result->success) {
            RCLCPP_ERROR(LOG, "Skill reported failure: %s", wrapped_result.result->error_message.c_str());
        }

        return wrapped_result.result->success;
    }

    bool OpenArmHeadHandNode::moveGroupToJoints(const std::string& group, const std::string& key)
    {
        std::vector<double> joints = getWaypoint(key);
        if (joints.empty()) {
            RCLCPP_ERROR(LOG, "Could not find waypoint joint angles for key: %s", key.c_str());
            return false;
        }

        ExecuteSkill::Goal goal;
        goal.skill_name = "move_to_joint";
        goal.arm = group;
        goal.planner_profile = "safe_rrt";
        goal.planning_mode = "normal";
        goal.joint_targets = joints;
        goal.velocity_override = 0.0;
        goal.position_only = false;

        RCLCPP_INFO(LOG, "Moving group '%s' to joint waypoint '%s'", group.c_str(), key.c_str());
        return sendSkillGoal(goal);
    }

    bool OpenArmHeadHandNode::moveGroupToNamedPose(const std::string& group, const std::string& named_pose)
    {
        ExecuteSkill::Goal goal;
        goal.skill_name = "move_to_named_pose";
        goal.arm = group;
        goal.named_pose = named_pose;
        goal.planner_profile = "safe_rrt";
        goal.planning_mode = "normal";
        goal.velocity_override = 0.0;
        goal.position_only = false;

        RCLCPP_INFO(LOG, "Moving group '%s' to named pose '%s'", group.c_str(), named_pose.c_str());
        return sendSkillGoal(goal);
    }

    bool OpenArmHeadHandNode::runHeadHome()
    {
        RCLCPP_INFO(LOG, "=== Sending head to home ===");
        bool success = moveGroupToJoints("head", "headHome");
        if (success) {
            RCLCPP_INFO(LOG, "=== Head home completed successfully! ===");
        } else {
            RCLCPP_ERROR(LOG, "=== Head home failed! ===");
        }
        return success;
    }

    bool OpenArmHeadHandNode::runHand(const std::string& side, const std::string& named_pose)
    {
        RCLCPP_INFO(LOG, "=== %s hand -> '%s' ===", side.c_str(), named_pose.c_str());
        bool success = true;
        if (side == "left" || side == "both") {
            success &= moveGroupToNamedPose("left_hand_fingers", named_pose);
        }
        if (side == "right" || side == "both") {
            success &= moveGroupToNamedPose("right_hand_fingers", named_pose);
        }
        if (success) {
            RCLCPP_INFO(LOG, "=== %s hand '%s' completed successfully! ===", side.c_str(), named_pose.c_str());
        } else {
            RCLCPP_ERROR(LOG, "=== %s hand '%s' failed! ===", side.c_str(), named_pose.c_str());
        }
        return success;
    }

    void OpenArmHeadHandNode::executeSingleCommand(const std::string& cmd)
    {
        if (cmd == "h") {
            runHeadHome();
        } else if (cmd == "lo") {
            runHand("left", "open");
        } else if (cmd == "lc") {
            runHand("left", "close");
        } else if (cmd == "lh") {
            runHand("left", "home");
        } else if (cmd == "ro") {
            runHand("right", "open");
        } else if (cmd == "rc") {
            runHand("right", "close");
        } else if (cmd == "rh") {
            runHand("right", "home");
        } else if (cmd == "bo") {
            runHand("both", "open");
        } else if (cmd == "bc") {
            runHand("both", "close");
        } else if (cmd == "bh") {
            runHand("both", "home");
        } else {
            std::cout << "\033[1;31mInvalid command: '" << cmd << "'\033[0m\n";
        }
    }

    void OpenArmHeadHandNode::handleCommand(const std::string& raw_cmd)
    {
        std::string cmd = raw_cmd;
        cmd.erase(0, cmd.find_first_not_of(" \t\r\n"));
        cmd.erase(cmd.find_last_not_of(" \t\r\n") + 1);

        if (cmd.empty()) return;

        if (cmd == "q") {
            running_ = false;
            cancel_flag_ = true;
            if (execution_thread_.joinable()) {
                execution_thread_.join();
            }
            rclcpp::shutdown();
            return;
        }

        std::lock_guard<std::mutex> lock(state_mutex_);
        if (cmd == "s" || cmd == "stop") {
            if (executing_) {
                RCLCPP_INFO(LOG, "Stop command received! Cancelling current execution...");
                cancel_flag_ = true;
                if (execution_thread_.joinable()) {
                    execution_thread_.join();
                }
                executing_ = false;
                cancel_flag_ = false;
                RCLCPP_INFO(LOG, "Execution successfully stopped.");
            } else {
                std::cout << "\033[1;33mNo active execution to stop.\033[0m\n";
                printMenu();
            }
            return;
        }

        if (executing_) {
            std::cout << "\033[1;33mExecution is currently running. Please type 's' or 'stop' to stop it first.\033[0m\n";
            return;
        }

        if (execution_thread_.joinable()) {
            execution_thread_.join();
        }

        cancel_flag_ = false;
        executing_ = true;

        {
            std_msgs::msg::String status_msg;
            status_msg.data = "executing:" + cmd;
            status_pub_->publish(status_msg);
        }

        execution_thread_ = std::thread([this, cmd]() {
            executeSingleCommand(cmd);
            executing_ = false;

            std_msgs::msg::String status_msg;
            status_msg.data = "idle";
            status_pub_->publish(status_msg);
            printMenu();
        });
    }

    void OpenArmHeadHandNode::printMenu()
    {
        std::cout << "\n\033[1;36m==================================================\033[0m\n";
        std::cout << "\033[1;33m      OpenArm Head + Hand Demo Node               \033[0m\n";
        std::cout << "\033[1;36m==================================================\033[0m\n";
        std::cout << "Select command to execute:\n";
        std::cout << "  \033[1;32mh\033[0m   : Head home\n";
        std::cout << "  \033[1;32mlo\033[0m  : Left hand open\n";
        std::cout << "  \033[1;32mlc\033[0m  : Left hand close\n";
        std::cout << "  \033[1;32mlh\033[0m  : Left hand home\n";
        std::cout << "  \033[1;32mro\033[0m  : Right hand open\n";
        std::cout << "  \033[1;32mrc\033[0m  : Right hand close\n";
        std::cout << "  \033[1;32mrh\033[0m  : Right hand home\n";
        std::cout << "  \033[1;32mbo\033[0m  : Both hands open\n";
        std::cout << "  \033[1;32mbc\033[0m  : Both hands close\n";
        std::cout << "  \033[1;32mbh\033[0m  : Both hands home\n";
        std::cout << "  \033[1;32ms\033[0m   : Stop/cancel currently running command\n";
        std::cout << "  \033[1;31mq\033[0m   : Quit\n";
        std::cout << "Enter command: ";
        std::cout.flush();
    }

    void OpenArmHeadHandNode::runConsole()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        printMenu();

        while (rclcpp::ok() && running_) {
            std::string cmd;
            if (!std::getline(std::cin, cmd)) {
                break;
            }
            handleCommand(cmd);
        }
    }
} // namespace openarm_head_hand

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<openarm_head_hand::OpenArmHeadHandNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
