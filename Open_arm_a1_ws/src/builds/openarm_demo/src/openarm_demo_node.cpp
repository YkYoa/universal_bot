#include <openarm_demo/openarm_demo_node.h>
#include <common/yaml_parser.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>

namespace openarm_demo
{
    auto DEMO_LOG = rclcpp::get_logger("demo_node");
    static YAML::Node g_config;

    OpenArmDemoNode::OpenArmDemoNode(const rclcpp::NodeOptions & options)
    : Node("openarm_demo_node", options), running_(true), current_state_(DemoState::IDLE)
    {
        // Declare the parameter for sequence.yaml file path
        this->declare_parameter<std::string>("sequence_yaml_path", 
            "/home/hans/universal_bot/Open_arm_a1_ws/src/builds/openarm_demo/config/sequence.yaml");

        this->get_parameter("sequence_yaml_path", sequence_yaml_path_);

        RCLCPP_INFO(DEMO_LOG, "Initializing OpenArm Demo Node");
        RCLCPP_INFO(DEMO_LOG, "Loading sequence config from: %s", sequence_yaml_path_.c_str());

        loadAndSketchSequence();

        // Setup ExecuteSkill action client
        action_client_ = rclcpp_action::create_client<ExecuteSkill>(this, "robot_skills_server/execute_skill");

        command_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/openarm_demo/command", 10,
            [this](const std_msgs::msg::String::SharedPtr msg) {
                RCLCPP_INFO(DEMO_LOG, "Received topic command: '%s'", msg->data.c_str());
                std::thread(&OpenArmDemoNode::handleCommand, this, msg->data).detach();
            });

        // Spawn Console thread
        console_thread_ = std::thread(&OpenArmDemoNode::runConsole, this);
    }

    OpenArmDemoNode::~OpenArmDemoNode()
    {
        running_ = false;
        if (console_thread_.joinable()) {
            console_thread_.join();
        }
        RCLCPP_INFO(DEMO_LOG, "OpenArm Demo Node stopped.");
    }

    void OpenArmDemoNode::loadAndSketchSequence()
    {
        config_ = common::loadAndSketchSequence(this->get_logger(), sequence_yaml_path_);
        g_config = config_;
    }

    std::vector<double> OpenArmDemoNode::getWaypoint(const std::string & key) const
    {
        return common::getWaypointFromConfig(config_, key);
    }

    std::vector<double> getGlobalWaypoint(const std::string & key)
    {
        return common::getWaypointFromConfig(g_config, key);
    }

    bool OpenArmDemoNode::sendSkillGoal(const ExecuteSkill::Goal& goal)
    {
        if (!action_client_->wait_for_action_server(std::chrono::seconds(5))) {
            RCLCPP_ERROR(DEMO_LOG, "Action server 'robot_skills_server/execute_skill' not available!");
            return false;
        }

        auto send_goal_options = rclcpp_action::Client<ExecuteSkill>::SendGoalOptions();
        auto goal_handle_future = action_client_->async_send_goal(goal, send_goal_options);

        // Wait for goal acceptance
        if (goal_handle_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            RCLCPP_ERROR(DEMO_LOG, "Action goal submission timed out!");
            return false;
        }

        auto goal_handle = goal_handle_future.get();
        if (!goal_handle) {
            RCLCPP_ERROR(DEMO_LOG, "Goal was rejected by robot_skills_server!");
            return false;
        }

        // Wait for result
        auto result_future = action_client_->async_get_result(goal_handle);
        while (rclcpp::ok() && running_ && !cancel_flag_) {
            if (result_future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
                break;
            }
        }

        if (cancel_flag_) {
            RCLCPP_INFO(DEMO_LOG, "Cancellation requested. Cancelling active goal on server...");
            action_client_->async_cancel_goal(goal_handle);
            return false;
        }

        if (!rclcpp::ok() || !running_) {
            return false;
        }

        auto wrapped_result = result_future.get();
        if (wrapped_result.code != rclcpp_action::ResultCode::SUCCEEDED) {
            RCLCPP_ERROR(DEMO_LOG, "Skill execution failed or was aborted! Code: %d", static_cast<int>(wrapped_result.code));
            return false;
        }

        return wrapped_result.result->success;
    }

    bool OpenArmDemoNode::moveArmToJoints(const std::string& arm, const std::string& key)
    {
        std::vector<double> joints = getWaypoint(key);
        if (joints.empty()) {
            RCLCPP_ERROR(DEMO_LOG, "Could not find waypoint joint angles for key: %s", key.c_str());
            return false;
        }

        ExecuteSkill::Goal goal;
        goal.skill_name = "move_to_joint";
        goal.arm = arm;
        goal.planner_profile = "safe_rrt";
        goal.planning_mode = "normal";
        goal.joint_targets = joints;
        goal.velocity_override = 0.0;
        goal.position_only = false;

        RCLCPP_INFO(DEMO_LOG, "Moving %s to joint waypoint '%s'", arm.c_str(), key.c_str());
        return sendSkillGoal(goal);
    }

    bool OpenArmDemoNode::moveArmToPose(const std::string& arm, const std::string& key)
    {
        std::vector<double> pose_vals = getWaypoint(key);
        if (pose_vals.size() != 7) {
            RCLCPP_ERROR(DEMO_LOG, "Could not find Cartesian pose (expected 7 elements) for key: %s", key.c_str());
            return false;
        }

        ExecuteSkill::Goal goal;
        goal.skill_name = "move_to_pose";
        goal.arm = arm;
        goal.planner_profile = "safe_rrt";
        goal.planning_mode = "normal";
        goal.velocity_override = 0.0;
        goal.position_only = false;

        goal.target_pose.header.frame_id = "world";
        goal.target_pose.header.stamp = this->now();
        goal.target_pose.pose.position.x = pose_vals[0];
        goal.target_pose.pose.position.y = pose_vals[1];
        goal.target_pose.pose.position.z = pose_vals[2];
        goal.target_pose.pose.orientation.x = pose_vals[3];
        goal.target_pose.pose.orientation.y = pose_vals[4];
        goal.target_pose.pose.orientation.z = pose_vals[5];
        goal.target_pose.pose.orientation.w = pose_vals[6];

        RCLCPP_INFO(DEMO_LOG, "Moving %s to Cartesian waypoint '%s'", arm.c_str(), key.c_str());
        return sendSkillGoal(goal);
    }

    bool OpenArmDemoNode::controlGripper(const std::string& arm, bool open)
    {
        ExecuteSkill::Goal goal;
        goal.skill_name = open ? "open_gripper" : "close_gripper";
        goal.arm = arm;
        goal.planner_profile = "";
        goal.planning_mode = "normal";

        RCLCPP_INFO(DEMO_LOG, "%s %s gripper", open ? "Opening" : "Closing", arm.c_str());
        return sendSkillGoal(goal);
    }

    void OpenArmDemoNode::runHomingSequence()
    {
        RCLCPP_INFO(DEMO_LOG, "=== Starting Homing Sequence ===");
        bool success = moveArmToJoints("left_arm", "laHomeAngle");
        success &= moveArmToJoints("right_arm", "raHomeAngle");

        if (success) {
            RCLCPP_INFO(DEMO_LOG, "=== Homing Sequence completed successfully! ===");
        } else {
            RCLCPP_ERROR(DEMO_LOG, "=== Homing Sequence failed! ===");
        }
    }

    void OpenArmDemoNode::runWavingJointSequence()
    {
        RCLCPP_INFO(DEMO_LOG, "=== Starting Joint Waving Sequence ===");
        bool success = moveArmToJoints("left_arm", "laPreWaveAngle");
        if (!success) return;

        // Wave back and forth only if wave1 and wave2 angles are defined
        std::vector<double> wave1 = getWaypoint("laWave1Angle");
        std::vector<double> wave2 = getWaypoint("laWave2Angle");
        if (!wave1.empty() && !wave2.empty()) {
            for (int i = 0; i < 2; ++i) {
                RCLCPP_INFO(DEMO_LOG, "Wave cycle %d/2...", i + 1);
                if (!moveArmToJoints("left_arm", "laWave1Angle")) return;
                if (!moveArmToJoints("left_arm", "laWave2Angle")) return;
            }
            moveArmToJoints("left_arm", "laPreWaveAngle");
            RCLCPP_INFO(DEMO_LOG, "=== Joint Waving Sequence completed! ===");
        } else {
            RCLCPP_INFO(DEMO_LOG, "=== Joint Waving Sequence completed! (Only pre-wave executed since wave1/wave2 angles are not defined) ===");
        }
    }

    void OpenArmDemoNode::runWavingCartesianSequence()
    {
        RCLCPP_INFO(DEMO_LOG, "=== Starting Cartesian Waving Sequence ===");

        std::vector<double> p0 = getWaypoint("laPreWaveJoint");
        std::vector<double> p_end = getWaypoint("laEndWaveJoint");

        if (p0.size() != 7 || p_end.size() != 7) {
            RCLCPP_ERROR(DEMO_LOG, "laPreWaveJoint and laEndWaveJoint must each have 7 elements [x,y,z,qx,qy,qz,qw]");
            return;
        }

        // Move to p0 position first 
        if(!moveArmToPose("left_arm", "laPreWaveJoint"))
            return;

        // --- Ellipse parameters ---
        const int    num_points   = 20;     // number of waypoints along the arc
        const double height_ratio = 0.5;    // semi-minor / semi-major  (controls arc height)

        // Start / end positions
        double sx = p0[0], sy = p0[1], sz = p0[2];
        double ex = p_end[0], ey = p_end[1], ez = p_end[2];

        // Midpoint (ellipse centre) and direction vector
        double mx = (sx + ex) / 2.0, my = (sy + ey) / 2.0, mz = (sz + ez) / 2.0;
        double dx = ex - sx, dy = ey - sy, dz = ez - sz;
        double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (dist < 1e-9) {
            RCLCPP_ERROR(DEMO_LOG, "Start and end poses are identical; cannot draw ellipse.");
            return;
        }

        // Unit vector along the major axis  (u)
        double ux = dx / dist, uy = dy / dist, uz = dz / dist;

        // Semi-major axis length
        double a = dist / 2.0;
        // Semi-minor axis length  (arc height)
        double b = a * height_ratio;

        // Build the arc-normal direction (v) — perpendicular to u, lying in the
        // plane formed by u and world-Z.  If u is nearly parallel to Z, fall back
        // to world-X.
        double ref_x = 0.0, ref_y = 0.0, ref_z = 1.0;  // world-Z as reference
        double dot_uz = ux * ref_x + uy * ref_y + uz * ref_z;
        if (std::abs(dot_uz) > 0.95) {
            // u ≈ Z → use world-X as reference instead
            ref_x = 1.0; ref_y = 0.0; ref_z = 0.0;
        }

        // v = cross(u, ref)  then normalise
        double vx = uy * ref_z - uz * ref_y;
        double vy = uz * ref_x - ux * ref_z;
        double vz = ux * ref_y - uy * ref_x;
        double vlen = std::sqrt(vx*vx + vy*vy + vz*vz);
        vx /= vlen; vy /= vlen; vz /= vlen;

        // --- Quaternion slerp helper (lambda) ---
        auto slerp = [](const double q1[4], const double q2[4], double t, double out[4])
        {
            double dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3];
            double q2c[4] = {q2[0], q2[1], q2[2], q2[3]};
            if (dot < 0.0) {                       // take the shorter arc
                dot = -dot;
                for (int i = 0; i < 4; ++i) q2c[i] = -q2c[i];
            }
            if (dot > 0.9995) {                    // nearly identical → lerp
                for (int i = 0; i < 4; ++i) out[i] = q1[i] + t * (q2c[i] - q1[i]);
            } else {
                double theta = std::acos(dot);
                double sin_t = std::sin(theta);
                double w1 = std::sin((1.0 - t) * theta) / sin_t;
                double w2 = std::sin(t * theta) / sin_t;
                for (int i = 0; i < 4; ++i) out[i] = w1 * q1[i] + w2 * q2c[i];
            }
            // normalise
            double n = std::sqrt(out[0]*out[0]+out[1]*out[1]+out[2]*out[2]+out[3]*out[3]);
            for (int i = 0; i < 4; ++i) out[i] /= n;
        };

        // Start / end quaternions  [qx, qy, qz, qw]
        double q_start[4] = { p0[3], p0[4], p0[5], p0[6] };
        double q_end[4]   = { p_end[3], p_end[4], p_end[5], p_end[6] };

        // --- Build waypoints along the half-ellipse (θ goes 0 → π) ---
        std::vector<geometry_msgs::msg::PoseStamped> waypoints;
        waypoints.reserve(num_points);

        for (int i = 0; i <= num_points; ++i) {
            double t     = static_cast<double>(i) / num_points;   // 0 … 1
            double theta = t * M_PI;                               // 0 … π

            // Parametric position on ellipse centred at midpoint:
            //   P(θ) = centre  +  a·cos(θ) · u  +  b·sin(θ) · v
            // At θ=0  → centre + a·u  = p_end
            // At θ=π  → centre - a·u  = p0
            // So we sweep from p_end to p0.  If the caller wants p0→p_end order
            // we reverse at the end, or just parameterise the other way:
            //   P(θ) = centre  -  a·cos(θ) · u  +  b·sin(θ) · v
            // At θ=0  → centre - a·u = p0  ✓
            // At θ=π  → centre + a·u = p_end  ✓
            double px = mx - a * std::cos(theta) * ux + b * std::sin(theta) * vx;
            double py = my - a * std::cos(theta) * uy + b * std::sin(theta) * vy;
            double pz = mz - a * std::cos(theta) * uz + b * std::sin(theta) * vz;

            // Slerp orientation
            double q[4];
            slerp(q_start, q_end, t, q);

            geometry_msgs::msg::PoseStamped ps;
            ps.header.frame_id = "world";
            ps.header.stamp    = this->now();
            ps.pose.position.x = px;
            ps.pose.position.y = py;
            ps.pose.position.z = pz;
            ps.pose.orientation.x = q[0];
            ps.pose.orientation.y = q[1];
            ps.pose.orientation.z = q[2];
            ps.pose.orientation.w = q[3];

            waypoints.push_back(ps);
        }

        RCLCPP_INFO(DEMO_LOG, "Generated %zu half-ellipse waypoints (height_ratio=%.2f)",
                     waypoints.size(), height_ratio);

        // --- Send as a single cartesian_move skill ---
        ExecuteSkill::Goal goal;
        goal.skill_name       = "cartesian_move";
        goal.arm              = "left_arm";
        goal.planner_profile  = "safe_rrt";
        goal.planning_mode    = "normal";
        goal.waypoints        = waypoints;
        goal.velocity_override = 0.0;
        goal.position_only    = false;

        bool success = sendSkillGoal(goal);
        if (success) {
            RCLCPP_INFO(DEMO_LOG, "=== Cartesian Waving Sequence completed successfully! ===");
        } else {
            RCLCPP_ERROR(DEMO_LOG, "=== Cartesian Waving Sequence FAILED! ===");
        }
    }

    void OpenArmDemoNode::runGreetingJointSequence()
    {
        RCLCPP_INFO(DEMO_LOG, "=== Starting Joint Greeting Sequence ===");
        bool success = moveArmToJoints("left_arm", "laPreGreetAngle");
        if (!success) return;

        // Move to greet angle only if defined
        std::vector<double> greet = getWaypoint("laGreetAngle");
        if (!greet.empty()) {
            if (!moveArmToJoints("left_arm", "laGreetAngle")) return;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            moveArmToJoints("left_arm", "laPreGreetAngle");
            RCLCPP_INFO(DEMO_LOG, "=== Joint Greeting Sequence completed! ===");
        } else {
            RCLCPP_INFO(DEMO_LOG, "=== Joint Greeting Sequence completed! (Only pre-greet executed since greet angle is not defined) ===");
        }
    }

    void OpenArmDemoNode::runGreetingCartesianSequence()
    {
        RCLCPP_INFO(DEMO_LOG, "=== Starting Cartesian Greeting Sequence ===");
        bool success = moveArmToPose("left_arm", "laPreGreetJoint");
        if (!success) return;

        // Move to greet pose only if defined
        std::vector<double> greet = getWaypoint("laGreetJoint");
        if (!greet.empty()) {
            if (!moveArmToPose("left_arm", "laGreetJoint")) return;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            moveArmToPose("left_arm", "laPreGreetJoint");
            RCLCPP_INFO(DEMO_LOG, "=== Cartesian Greeting Sequence completed! ===");
        } else {
            RCLCPP_INFO(DEMO_LOG, "=== Cartesian Greeting Sequence completed! (Only pre-greet executed since greet pose is not defined) ===");
        }
    }

    void OpenArmDemoNode::runGripperSequence(bool open)
    {
        RCLCPP_INFO(DEMO_LOG, "=== Triggering Grippers ===");
        controlGripper("left_arm", open);
        controlGripper("right_arm", open);
        RCLCPP_INFO(DEMO_LOG, "=== Gripper Sequence complete ===");
    }

    void OpenArmDemoNode::executeSingleCommand(const std::string& cmd)
    {
        if (cmd == "h") {
            current_state_ = DemoState::HOMING;
            runHomingSequence();
        } else if (cmd == "wj") {
            current_state_ = DemoState::WAVING_JOINT;
            runWavingJointSequence();
        } else if (cmd == "wc") {
            current_state_ = DemoState::WAVING_CARTESIAN;
            runWavingCartesianSequence();
        } else if (cmd == "gj") {
            current_state_ = DemoState::GREETING_JOINT;
            runGreetingJointSequence();
        } else if (cmd == "gc") {
            current_state_ = DemoState::GREETING_CARTESIAN;
            runGreetingCartesianSequence();
        } else if (cmd == "o") {
            current_state_ = DemoState::OPENING_GRIPPER;
            runGripperSequence(true);
        } else if (cmd == "c") {
            current_state_ = DemoState::CLOSING_GRIPPER;
            runGripperSequence(false);
        } else {
            std::cout << "\033[1;31mInvalid command: '" << cmd << "'\033[0m\n";
        }
    }

    void OpenArmDemoNode::handleCommand(const std::string& raw_cmd)
    {
        std::string cmd = raw_cmd;
        // Remove trailing carriage returns/spaces
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

        std::vector<std::string> args;
        std::stringstream ss(cmd);
        std::string token;
        while (ss >> token) {
            args.push_back(token);
        }
        if (args.empty()) return;

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (args[0] == "s" || args[0] == "stop") {
                if (executing_) {
                    RCLCPP_INFO(DEMO_LOG, "Stop command received! Cancelling current execution...");
                    cancel_flag_ = true;
                    if (execution_thread_.joinable()) {
                        execution_thread_.join();
                    }
                    executing_ = false;
                    cancel_flag_ = false;
                    RCLCPP_INFO(DEMO_LOG, "Execution successfully stopped.");
                } else {
                    std::cout << "\033[1;33mNo active execution to stop.\033[0m\n";
                }
                return;
            }

            if (executing_) {
                std::cout << "\033[1;33mExecution is currently running. Please type 's' or 'stop' to stop it first.\033[0m\n";
                return;
            }

            // Join any previously finished thread
            if (execution_thread_.joinable()) {
                execution_thread_.join();
            }

            cancel_flag_ = false;
            executing_ = true;

            execution_thread_ = std::thread([this, args]() {
                if ((args[0] == "l" || args[0] == "loop")) {
                    if (args.size() < 3) {
                        std::cout << "\033[1;33mUsage: l <command> <count/forever> (e.g., l wc 3, l wc forever)\033[0m\n";
                    } else {
                        std::string sub_cmd = args[1];
                        bool forever = (args[2] == "forever" || args[2] == "-1" || args[2] == "0");
                        int count = 1;
                        if (!forever) {
                            try {
                                count = std::stoi(args[2]);
                            } catch (...) {
                                count = 1;
                            }
                        }

                        if (forever) {
                            RCLCPP_INFO(DEMO_LOG, "Looping command '%s' forever. Type 's' to stop.", sub_cmd.c_str());
                            int i = 0;
                            while (rclcpp::ok() && running_ && !cancel_flag_) {
                                RCLCPP_INFO(DEMO_LOG, "[Loop %d] Executing '%s'", ++i, sub_cmd.c_str());
                                executeSingleCommand(sub_cmd);
                            }
                        } else {
                            RCLCPP_INFO(DEMO_LOG, "Looping command '%s' for %d times. Type 's' to stop.", sub_cmd.c_str(), count);
                            for (int i = 0; i < count && rclcpp::ok() && running_ && !cancel_flag_; ++i) {
                                RCLCPP_INFO(DEMO_LOG, "[Loop %d/%d] Executing '%s'", i + 1, count, sub_cmd.c_str());
                                executeSingleCommand(sub_cmd);
                            }
                        }
                    }
                } else {
                    executeSingleCommand(args[0]);
                }
                executing_ = false;
                current_state_ = DemoState::IDLE;
            });
        }
    }

    void OpenArmDemoNode::runConsole()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(2)); // wait for ROS init to print clean logs

        while (rclcpp::ok() && running_) {
            std::cout << "\n\033[1;36m==================================================\033[0m\n";
            std::cout << "\033[1;33m       OpenArm Demo Sequence State Machine       \033[0m\n";
            std::cout << "\033[1;36m==================================================\033[0m\n";
            std::cout << "Select state/sequence to execute:\n";
            std::cout << "  \033[1;32mh\033[0m   : Homing (Send left/right arms to pre-grasp angles)\n";
            std::cout << "  \033[1;32mwj\033[0m  : Run Waving sequence (Joint-space)\n";
            std::cout << "  \033[1;32mwc\033[0m  : Run Waving sequence (Cartesian-space)\n";
            std::cout << "  \033[1;32mgj\033[0m  : Run Greeting sequence (Joint-space)\n";
            std::cout << "  \033[1;32mgc\033[0m  : Run Greeting sequence (Cartesian-space)\n";
            std::cout << "  \033[1;32mo\033[0m   : Open Left & Right Grippers\n";
            std::cout << "  \033[1;32mc\033[0m   : Close Left & Right Grippers\n";
            std::cout << "  \033[1;32ml\033[0m   : Loop command N times (e.g. l wc 5) or forever (l wc forever)\n";
            std::cout << "  \033[1;32ms\033[0m   : Stop/cancel currently running command or loop\n";
            std::cout << "  \033[1;31mq\033[0m   : Quit Demo Node\n";
            std::cout << "Enter command: ";
            std::cout.flush();

            std::string cmd;
            if (!std::getline(std::cin, cmd)) {
                break;
            }

            handleCommand(cmd);
        }
    }
} // namespace openarm_demo

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<openarm_demo::OpenArmDemoNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}