// ─────────────────────────────────────────────────────────────────────────────
// bt_executor_node.cpp
//
// Main entry point for the BT executor.
//
// Responsibilities:
//   1. Initialize ROS 2 and spin a MultiThreadedExecutor (BT action nodes
//      need their own callback threads to remain non-blocking).
//   2. Register every BT node type with the factory.
//   3. Load the tree XML from a file (hot-reloadable via a ROS parameter).
//   4. Tick the tree at a fixed rate (default 50 Hz).
//   5. Publish the current BT status on /bt_executor/status for UI telemetry.
//
// Key design decisions:
//   - The node shares its rclcpp::Node with every BT action node via
//     RosNodeParams so they can all use the same executor/callback group.
//   - A shared blackboard is created here and passed into the tree; it
//     persists across ticks so condition nodes always see the latest data.
//   - The StateMonitor runs in a separate subscription thread and writes
//     arm / gripper state to the blackboard asynchronously.
// ─────────────────────────────────────────────────────────────────────────────

#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/executors/multi_threaded_executor.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

// BehaviorTree.CPP v4
#include "behaviortree_cpp/bt_factory.h"
#include "behaviortree_cpp/loggers/bt_cout_logger.h"
#include "behaviortree_cpp/loggers/bt_file_logger_v2.h"
#include "behaviortree_cpp/loggers/groot2_publisher.h"
// behaviortree_ros2 supplies RosNodeParams
#include "behaviortree_ros2/ros_node_params.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <sstream>
#include "behaviortree_cpp/loggers/abstract_logger.h"

// Our blackboard key constants
#include "bt_executor/blackboard_keys.hpp"

// ── Condition node headers ────────────────────────────────────────────────────
#include "bt_executor/nodes/conditions/is_object_visible.hpp"
#include "bt_executor/nodes/conditions/is_gripper_holding.hpp"
#include "bt_executor/nodes/conditions/is_arm_at_pose.hpp"
#include "bt_executor/nodes/conditions/is_replan_needed.hpp"
#include "bt_executor/nodes/conditions/is_goal_changed.hpp"

// ── Action node headers ───────────────────────────────────────────────────────
#include "bt_executor/nodes/actions/plan_to_named_pose.hpp"
#include "bt_executor/nodes/actions/plan_to_pose.hpp"
#include "bt_executor/nodes/actions/execute_trajectory.hpp"
#include "bt_executor/nodes/actions/control_gripper.hpp"
#include "bt_executor/nodes/actions/query_vla.hpp"
#include "bt_executor/nodes/actions/safe_abort.hpp"
#include "bt_executor/nodes/actions/set_planner_config.hpp"

using namespace std::chrono_literals;
using namespace bt_executor;

// ─────────────────────────────────────────────────────────────────────────────
// BTTransitionPublisher — forwards BehaviorTree node status transitions to ROS 2
// ─────────────────────────────────────────────────────────────────────────────
class BTTransitionPublisher : public BT::StatusChangeLogger
{
public:
  BTTransitionPublisher(const BT::Tree& tree, rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub)
  : StatusChangeLogger(tree.rootNode()), pub_(pub)
  {}

  void callback(BT::Duration timestamp, const BT::TreeNode& node, BT::NodeStatus prev_status, BT::NodeStatus status) override
  {
    double seconds = std::chrono::duration<double>(timestamp).count();
    std::stringstream ss;
    ss << "{"
       << "\"timestamp\":" << seconds << ","
       << "\"node_name\":\"" << node.name() << "\","
       << "\"registration_name\":\"" << node.registrationName() << "\","
       << "\"prev_status\":\"" << BT::toStr(prev_status) << "\","
       << "\"status\":\"" << BT::toStr(status) << "\""
       << "}";
    
    std_msgs::msg::String msg;
    msg.data = ss.str();
    pub_->publish(msg);
  }

  void flush() override {}

private:
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
};

// ─────────────────────────────────────────────────────────────────────────────
// StateMonitor — thin ROS node that writes joint / gripper state to the BB
// ─────────────────────────────────────────────────────────────────────────────
class StateMonitor : public rclcpp::Node
{
public:
  explicit StateMonitor(BT::Blackboard::Ptr blackboard)
  : Node("bt_state_monitor"), bb_(blackboard)
  {
    if (!bb_) {
      RCLCPP_ERROR(get_logger(), "StateMonitor: blackboard is null!");
      return;
    }

    // Joint states → blackboard
    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::JointState::SharedPtr msg) {
        bb_->set(BB_JOINT_STATE, *msg);
      });

    // TODO: subscribe to gripper force feedback topics when available.
    // Example placeholder — replace with real hardware topic:
    //   /left_gripper_controller/state  →  BB_LEFT_GRIP_FORCE
    //   /right_gripper_controller/state →  BB_RIGHT_GRIP_FORCE
    // Then derive BB_LEFT_GRIP_HOLDING / BB_RIGHT_GRIP_HOLDING using
    // GRIP_FORCE_THRESHOLD.

    RCLCPP_INFO(get_logger(), "StateMonitor ready");
  }

private:
  BT::Blackboard::Ptr bb_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
};

// ─────────────────────────────────────────────────────────────────────────────
// BTExecutorNode
// ─────────────────────────────────────────────────────────────────────────────
class BTExecutorNode : public rclcpp::Node
{
public:
  BTExecutorNode()
  : Node("bt_executor")
  {
    // ── Parameters ──────────────────────────────────────────────────────────
    declare_parameter("bt_xml_path", "");          // path to tree XML file
    declare_parameter("tick_rate_hz", 50.0);       // BT tick frequency
    declare_parameter("log_to_file", false);       // enable FileLogger
    declare_parameter("log_path", "/tmp/bt_log");  // FileLogger output path

    bt_xml_path_ = get_parameter("bt_xml_path").as_string();
    tick_rate_hz_ = get_parameter("tick_rate_hz").as_double();

    // ── Status publisher (UI telemetry) ──────────────────────────────────────
    status_pub_ = create_publisher<std_msgs::msg::String>(
      "~/status", rclcpp::SystemDefaultsQoS());

    blackboard_pub_ = create_publisher<std_msgs::msg::String>(
      "~/blackboard", rclcpp::SystemDefaultsQoS());

    transition_pub_ = create_publisher<std_msgs::msg::String>(
      "~/transitions", rclcpp::SystemDefaultsQoS());

    // ── Replan service (VLA bridge calls this to signal a new goal) ──────────
    replan_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/replan",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> resp) {
        blackboard_->set(BB_REPLAN_NEEDED, true);
        resp->success = true;
        resp->message = "Replan flag set";
      });

    // ── Goal update service (external systems signal a goal change) ──────────
    goal_update_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/goal_updated",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> resp) {
        blackboard_->set(BB_GOAL_CHANGED, true);
        int stamp = 0;
        blackboard_->get(BB_GOAL_STAMP, stamp);
        blackboard_->set(BB_GOAL_STAMP, stamp + 1);
        resp->success = true;
        resp->message = "Goal change signaled";
      });
    
    // ── Target pose subscription (updates blackboard key and signals goal changed) ──
    target_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "~/vla_grasp_pose", rclcpp::SystemDefaultsQoS(),
      [this](geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        blackboard_->set(BB_GRASP_POSE, *msg);
        blackboard_->set(BB_GOAL_CHANGED, true);
        int stamp = 0;
        blackboard_->get(BB_GOAL_STAMP, stamp);
        blackboard_->set(BB_GOAL_STAMP, stamp + 1);
        RCLCPP_INFO(get_logger(), "Received target pose update: (%.3f, %.3f, %.3f). Goal changed flagged.",
                    msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
      });

    RCLCPP_INFO(get_logger(), "BTExecutorNode constructed");
  }

  /// Called after the node is added to the executor so shared_from_this() works.
  void initialize()
  {
    // ── Shared blackboard ────────────────────────────────────────────────────
    blackboard_ = BT::Blackboard::create();

    // Seed default values so nodes never read an unset key
    blackboard_->set(BB_ACTIVE_ARM,        std::string("left_arm"));
    blackboard_->set(BB_VEL_SCALE,         0.3);
    blackboard_->set(BB_ACC_SCALE,         0.3);
    blackboard_->set(BB_REPLAN_NEEDED,     false);
    blackboard_->set(BB_RECOVERY_COUNT,    0);
    blackboard_->set(BB_STATUS_MSG,        std::string("idle"));
    blackboard_->set(BB_VLA_CONFIDENCE,    1.0);
    blackboard_->set(BB_NEW_GOAL_READY,    false);
    blackboard_->set(BB_LEFT_GRIP_HOLDING, false);
    blackboard_->set(BB_RIGHT_GRIP_HOLDING,false);
    blackboard_->set(BB_OBJECT_VISIBLE,    false);
    blackboard_->set(BB_PLAN_FEASIBLE,     false);
    blackboard_->set(BB_TASK_NAME,         std::string("Push the apple to the block"));

    // Planner defaults (safe_rrt equivalent)
    blackboard_->set(BB_PIPELINE_ID,       std::string("ompl"));
    blackboard_->set(BB_PLANNER_ID,        std::string("RRTConnectkConfigDefault"));
    blackboard_->set(BB_PLAN_TIME,         5.0);
    blackboard_->set(BB_NUM_ATTEMPTS,      10);
    blackboard_->set(BB_PLANNER_PROFILE,   std::string("safe_rrt"));

    // Realtime replanning
    blackboard_->set(BB_GOAL_CHANGED,      false);
    blackboard_->set(BB_GOAL_STAMP,        0);
    blackboard_->set(BB_PLANNING_MODE,     std::string("normal"));

    // Provide the shared node pointer so SyncActionNodes (e.g. SafeAbort) can
    // publish ROS messages without creating their own rclcpp::Node instances,
    // which would cause node-name collisions when the BT factory registers them.
    blackboard_->set("ros_node", std::static_pointer_cast<rclcpp::Node>(shared_from_this()));

    // ── Build the factory and register all node types ────────────────────────
    BT::RosNodeParams ros_params;
    ros_params.nh = shared_from_this();
    ros_params.default_port_value = "robot_skills_server/execute_skill";
    // All BT action nodes share one callback group on the main executor.
    ros_params.server_timeout = std::chrono::milliseconds(5000);

    BT::RosNodeParams gripper_params;
    gripper_params.nh = shared_from_this();
    gripper_params.default_port_value = ""; // Avoid default port conflict
    gripper_params.server_timeout = std::chrono::milliseconds(5000);

    BT::RosNodeParams vla_params;
    vla_params.nh = shared_from_this();
    vla_params.default_port_value = "/vla_bridge/query";
    vla_params.server_timeout = std::chrono::milliseconds(5000);

    // Condition nodes (synchronous — never block)
    factory_.registerNodeType<IsObjectVisible>("IsObjectVisible");
    factory_.registerNodeType<IsGripperHolding>("IsGripperHolding");
    factory_.registerNodeType<IsArmAtPose>("IsArmAtPose");
    factory_.registerNodeType<IsReplanNeeded>("IsReplanNeeded");
    factory_.registerNodeType<IsGoalChanged>("IsGoalChanged");

    // Synchronous action nodes
    factory_.registerNodeType<SafeAbort>("SafeAbort");
    factory_.registerNodeType<SetPlannerConfig>("SetPlannerConfig");
    factory_.registerNodeType<ExecuteTrajectory>("ExecuteTrajectory");

    // Stateful action nodes (async but not ROS actions)

    // ROS action nodes (async — return RUNNING until done)
    factory_.registerNodeType<QueryVLA>       ("QueryVLA",        vla_params);
    factory_.registerNodeType<PlanToPose>     ("PlanToPose",      ros_params);
    factory_.registerNodeType<PlanToNamedPose>("PlanToNamedPose", ros_params);
    factory_.registerNodeType<CloseGripper>   ("CloseGripper",    gripper_params);
    factory_.registerNodeType<OpenGripper>    ("OpenGripper",     gripper_params);

    // ── Load tree from XML ───────────────────────────────────────────────────
    load_tree();

    // ── Transition Publisher for Web UI ──────────────────────────────────────
    if (tree_) {
      transition_publisher_ = std::make_unique<BTTransitionPublisher>(*tree_, transition_pub_);
      RCLCPP_INFO(get_logger(), "BTTransitionPublisher initialized");
    }

    // ── Optional loggers ─────────────────────────────────────────────────────
    // cout_logger_ = std::make_unique<BT::StdCoutLogger>(*tree_); // Disabled to prevent console spam

    if (get_parameter("log_to_file").as_bool()) {
      const auto log_path = get_parameter("log_path").as_string();
      file_logger_ = std::make_unique<BT::FileLogger2>(*tree_, log_path);
      RCLCPP_INFO(get_logger(), "BT FileLogger → %s", log_path.c_str());
    }

    // ── Groot2 Publisher ─────────────────────────────────────────────────────
    // This allows Groot2 to connect via ZMQ to monitor the tree in real-time
    try {
      zmq_publisher_ = std::make_unique<BT::Groot2Publisher>(*tree_, 1666);
      RCLCPP_INFO(get_logger(), "Groot2 ZMQ Publisher started on port 1666");
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "Failed to start Groot2 ZMQ Publisher: %s", e.what());
    }

    // ── Tick timer ───────────────────────────────────────────────────────────
    const auto period = std::chrono::duration<double>(1.0 / tick_rate_hz_);
    tick_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&BTExecutorNode::tick_once, this));

    RCLCPP_INFO(get_logger(), "BT executor initialized at %.0f Hz", tick_rate_hz_);
  }

  /// Returns the shared blackboard pointer (for StateMonitor)
  BT::Blackboard::Ptr get_blackboard() const { return blackboard_; }

private:
  void load_tree()
  {
    std::string path_to_load = bt_xml_path_;
    if (path_to_load.empty()) {
      try {
        path_to_load = ament_index_cpp::get_package_share_directory("bt_executor") + "/bt_trees/pick_and_place.xml";
        RCLCPP_INFO(get_logger(), "bt_xml_path not set — loading default tree from share: %s", path_to_load.c_str());
      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to get share directory of bt_executor: %s", e.what());
        return;
      }
    } else {
      RCLCPP_INFO(get_logger(), "Loading BT from: %s", path_to_load.c_str());
    }

    try {
      tree_ = std::make_unique<BT::Tree>(
        factory_.createTreeFromFile(path_to_load, blackboard_));
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Failed to load behavior tree: %s", e.what());
    }
  }

  void publish_blackboard()
  {
    std::string active_arm = "left_arm";
    double vel_scale = 0.3, acc_scale = 0.3, confidence = 1.0;
    bool replan_needed = false, new_goal_ready = false, left_holding = false, right_holding = false;
    bool object_visible = false, plan_feasible = false, goal_changed = false;
    int recovery_count = 0, goal_stamp = 0;
    std::string status_msg = "idle", pipeline_id = "ompl", planner_id = "RRTConnectkConfigDefault", planner_profile = "safe_rrt", planning_mode = "normal";

    blackboard_->get(BB_ACTIVE_ARM, active_arm);
    blackboard_->get(BB_VEL_SCALE, vel_scale);
    blackboard_->get(BB_ACC_SCALE, acc_scale);
    blackboard_->get(BB_REPLAN_NEEDED, replan_needed);
    blackboard_->get(BB_RECOVERY_COUNT, recovery_count);
    blackboard_->get(BB_STATUS_MSG, status_msg);
    blackboard_->get(BB_VLA_CONFIDENCE, confidence);
    blackboard_->get(BB_NEW_GOAL_READY, new_goal_ready);
    blackboard_->get(BB_LEFT_GRIP_HOLDING, left_holding);
    blackboard_->get(BB_RIGHT_GRIP_HOLDING, right_holding);
    blackboard_->get(BB_OBJECT_VISIBLE, object_visible);
    blackboard_->get(BB_PLAN_FEASIBLE, plan_feasible);
    blackboard_->get(BB_PIPELINE_ID, pipeline_id);
    blackboard_->get(BB_PLANNER_ID, planner_id);
    blackboard_->get(BB_PLANNER_PROFILE, planner_profile);
    blackboard_->get(BB_GOAL_CHANGED, goal_changed);
    blackboard_->get(BB_GOAL_STAMP, goal_stamp);
    blackboard_->get(BB_PLANNING_MODE, planning_mode);

    std::stringstream ss;
    ss << "{"
       << "\"plan_active_arm\":\"" << active_arm << "\","
       << "\"plan_velocity_scaling\":" << vel_scale << ","
       << "\"plan_accel_scaling\":" << acc_scale << ","
       << "\"task_replan_needed\":" << (replan_needed ? "true" : "false") << ","
       << "\"task_recovery_count\":" << recovery_count << ","
       << "\"task_status_message\":\"" << status_msg << "\","
       << "\"vla_confidence\":" << confidence << ","
       << "\"vla_new_goal_ready\":" << (new_goal_ready ? "true" : "false") << ","
       << "\"gripper_left_holding\":" << (left_holding ? "true" : "false") << ","
       << "\"gripper_right_holding\":" << (right_holding ? "true" : "false") << ","
       << "\"task_object_visible\":" << (object_visible ? "true" : "false") << ","
       << "\"task_plan_feasible\":" << (plan_feasible ? "true" : "false") << ","
       << "\"plan_pipeline_id\":\"" << pipeline_id << "\","
       << "\"plan_planner_id\":\"" << planner_id << "\","
       << "\"plan_planner_profile\":\"" << planner_profile << "\","
       << "\"task_goal_changed\":" << (goal_changed ? "true" : "false") << ","
       << "\"task_goal_stamp\":" << goal_stamp << ","
       << "\"plan_planning_mode\":\"" << planning_mode << "\""
       << "}";

    std_msgs::msg::String msg;
    msg.data = ss.str();
    blackboard_pub_->publish(msg);
  }

  void tick_once()
  {
    if (!tree_) return;

    const BT::NodeStatus status = tree_->tickOnce();

    // Publish status string for UI / debugging
    std_msgs::msg::String msg;
    switch (status) {
      case BT::NodeStatus::RUNNING: msg.data = "RUNNING"; break;
      case BT::NodeStatus::SUCCESS: msg.data = "SUCCESS"; break;
      case BT::NodeStatus::FAILURE: msg.data = "FAILURE"; break;
      default:                      msg.data = "IDLE";    break;
    }
    status_pub_->publish(msg);

    // Throttle blackboard publishing to every 10 ticks (at 50 Hz, this is 5 Hz)
    static int tick_count = 0;
    if (++tick_count >= 10) {
      publish_blackboard();
      tick_count = 0;
    }

    // On tree completion (SUCCESS or FAILURE) halt so it can restart cleanly
    if (status != BT::NodeStatus::RUNNING) {
      tree_->haltTree();
    }
  }

  // ── Members ─────────────────────────────────────────────────────────────────
  BT::BehaviorTreeFactory factory_;
  BT::Blackboard::Ptr     blackboard_;
  std::unique_ptr<BT::Tree> tree_;

  std::unique_ptr<BT::StdCoutLogger> cout_logger_;
  std::unique_ptr<BT::FileLogger2>   file_logger_;
  std::unique_ptr<BT::Groot2Publisher>  zmq_publisher_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr blackboard_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr transition_pub_;
  std::unique_ptr<BTTransitionPublisher> transition_publisher_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr  replan_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr  goal_update_srv_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
  rclcpp::TimerBase::SharedPtr tick_timer_;

  std::string bt_xml_path_;
  double tick_rate_hz_{50.0};
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  // MultiThreadedExecutor: BT async action nodes need their own threads
  auto executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>(
    rclcpp::ExecutorOptions(), 4 /*threads*/);

  auto bt_node = std::make_shared<BTExecutorNode>();
  executor->add_node(bt_node);

  // Initialize creates the blackboard
  bt_node->initialize();

  // Now create StateMonitor with the real blackboard pointer
  auto monitor = std::make_shared<StateMonitor>(bt_node->get_blackboard());
  executor->add_node(monitor);

  RCLCPP_INFO(bt_node->get_logger(), "Spinning BT executor...");
  executor->spin();

  rclcpp::shutdown();
  return 0;
}
