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
    /// What the console/topic-driven demo state machine is currently doing.
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

    /// A standalone demo/exercise node predating sequence_executor: drives
    /// robot_skills' ExecuteSkill action directly from a console menu or the
    /// /openarm_demo/command topic, running hardcoded homing/waving/greeting/
    /// gripper sequences read out of a sequence.yaml.
    class OpenArmDemoNode : public rclcpp::Node
    {
    public:
        using ExecuteSkill = openarm_messages::action::ExecuteSkill;
        using GoalHandleExecuteSkill = rclcpp_action::ClientGoalHandle<ExecuteSkill>;

        /// Loads sequence.yaml, opens the ExecuteSkill action client, sets up
        /// the command topic/status publisher, and spawns the console thread.
        explicit OpenArmDemoNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
        /// Stops the console thread and joins it.
        virtual ~OpenArmDemoNode();

        // Member function to retrieve a waypoint parameter
        /// Looks up a waypoint's values from the loaded sequence config by key.
        std::vector<double> getWaypoint(const std::string & key) const;

    private:
        /// Loads sequence_yaml_path_ into config_ (and the global g_config
        /// used by getGlobalWaypoint()), logging a sketch of every section.
        void loadAndSketchSequence();
        /// Console thread body: prints the menu and reads commands from stdin
        /// in a loop until quit or shutdown.
        void runConsole();
        /// Prints the interactive command menu to stdout.
        void printMenu();
        /// Sends `goal` to robot_skills_server/execute_skill and blocks until
        /// it completes, is cancelled (cancel_flag_), or the node stops;
        /// returns the skill's reported success.
        bool sendSkillGoal(const ExecuteSkill::Goal& goal);

        // Core movement helpers
        /// Sends a "move_to_joint" goal for `arm` to the joint waypoint named `key`.
        bool moveArmToJoints(const std::string& arm, const std::string& key);
        /// Sends a "move_to_pose" goal for `arm` to the 7-value Cartesian waypoint named `key`.
        bool moveArmToPose(const std::string& arm, const std::string& key);
        /// Sends an open/close gripper goal for `arm`.
        bool controlGripper(const std::string& arm, bool open);

        // State Machine Action Handlers
        /// Sends both arms to their "<side>HomeAngle" waypoints.
        void runHomingSequence();
        /// Unused/reserved joint-space waving handler (no definition exists;
        /// only the Cartesian variant is wired to a command).
        void runWavingJointSequence();
        /// Builds a half-ellipse arc of Cartesian waypoints between
        /// laPreWaveJoint and laEndWaveJoint (slerping orientation along the
        /// way) and executes it as one "cartesian_move" skill goal.
        void runWavingCartesianSequence();
        /// Moves to the pre-greet joint pose, then (if defined) the greet
        /// pose and back.
        void runGreetingJointSequence();
        /// Same as runGreetingJointSequence() but using Cartesian pose waypoints.
        void runGreetingCartesianSequence();
        /// Opens or closes both grippers.
        void runGripperSequence(bool open);

        /// Parses and dispatches one command line (including "l"/"loop" and
        /// "s"/"stop"), running the matched sequence on a background thread
        /// so the console/topic callback never blocks.
        void handleCommand(const std::string& cmd);
        /// Runs exactly one command token's sequence handler (h/wc/gj/gc/o/c),
        /// used both directly and by handleCommand()'s loop mode.
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

    /// Looks up a waypoint by key from the most recently loaded sequence
    /// config (set by OpenArmDemoNode::loadAndSketchSequence()), for use
    /// from free functions that don't hold a node reference.
    std::vector<double> getGlobalWaypoint(const std::string & key);
} // namespace openarm_demo