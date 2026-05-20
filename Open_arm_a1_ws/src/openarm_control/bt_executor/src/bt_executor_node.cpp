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

// BehaviorTree.CPP v4
#include "behaviortree_cpp/bt_factory.h"
#include "behaviortree_cpp/loggers/bt_cout_logger.h"
#include "behaviortree_cpp/loggers/bt_file_logger_v2.h"
#include "behaviortree_cpp/loggers/groot2_publisher.h"
// behaviortree_ros2 supplies RosNodeParams
#include "behaviortree_ros2/ros_node_params.hpp"

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

    // ── Build the factory and register all node types ────────────────────────
    BT::RosNodeParams ros_params;
    ros_params.nh = shared_from_this();
    // All BT action nodes share one callback group on the main executor.
    ros_params.server_timeout = std::chrono::milliseconds(5000);

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
    factory_.registerNodeType<QueryVLA>("QueryVLA");

    // ROS action nodes (async — return RUNNING until done)
    factory_.registerNodeType<PlanToPose>     ("PlanToPose",      ros_params);
    factory_.registerNodeType<PlanToNamedPose>("PlanToNamedPose", ros_params);
    factory_.registerNodeType<CloseGripper>   ("CloseGripper",    ros_params);
    factory_.registerNodeType<OpenGripper>    ("OpenGripper",     ros_params);

    // ── Load tree from XML ───────────────────────────────────────────────────
    load_tree();

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
    if (bt_xml_path_.empty()) {
      // Fall back to the inline default tree (useful for first-run / testing)
      RCLCPP_WARN(get_logger(),
        "bt_xml_path not set — loading built-in pick_and_place tree");
      tree_ = std::make_unique<BT::Tree>(
        factory_.createTreeFromText(default_tree_xml_, blackboard_));
    } else {
      RCLCPP_INFO(get_logger(), "Loading BT from: %s", bt_xml_path_.c_str());
      tree_ = std::make_unique<BT::Tree>(
        factory_.createTreeFromFile(bt_xml_path_, blackboard_));
    }
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
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr  replan_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr  goal_update_srv_;
  rclcpp::TimerBase::SharedPtr tick_timer_;

  std::string bt_xml_path_;
  double tick_rate_hz_{50.0};

  // ── Built-in fallback tree XML ────────────────────────────────────────────
  // This is a minimal test tree.  In production load from file.
  static constexpr const char* default_tree_xml_ = R"(
<root BTCPP_format="4">
  <BehaviorTree ID="PickAndPlace">
    <Fallback name="root">

      <!-- Happy path -->
      <Sequence name="pick_and_place">
        <IsObjectVisible object_label="{vla_target_object}" />
        <SetPlannerConfig profile="safe_rrt" />
        <MoveToNamedPose arm="{plan_active_arm}" pose_name="ready" />

        <!-- Reactive approach: re-plans if goal changes mid-motion -->
        <ReactiveSequence name="reactive_approach">
          <Inverter>
            <IsGoalChanged />
          </Inverter>
          <SetPlannerConfig profile="linear_approach" />
          <MoveToPose
            arm="{plan_active_arm}"
            target_pose="{vla_grasp_pose}"
            velocity_scaling="{plan_velocity_scaling}"
            position_only="false" />
        </ReactiveSequence>

        <GraspObject arm="{plan_active_arm}" />
        <IsGripperHolding arm="{plan_active_arm}" />

        <SetPlannerConfig profile="fast_ptp" />
        <MoveToPose
          arm="{plan_active_arm}"
          target_pose="{vla_place_pose}"
          velocity_scaling="{plan_velocity_scaling}"
          position_only="true" />
        <ReleaseObject arm="{plan_active_arm}" />
        <MoveToNamedPose arm="{plan_active_arm}" pose_name="home" />
      </Sequence>

      <!-- Recovery path -->
      <Sequence name="recovery">
        <IsReplanNeeded confidence_threshold="0.65" />
        <QueryVLA task="{vla_task_name}" />
        <RetryUntilSuccessful num_attempts="2">
          <Sequence name="retry_pick">
            <IsObjectVisible object_label="{vla_target_object}" />
            <SetPlannerConfig profile="realtime_rrt" />
            <MoveToPose
              arm="{plan_active_arm}"
              target_pose="{vla_grasp_pose}"
              velocity_scaling="0.2"
              position_only="false" />
            <GraspObject arm="{plan_active_arm}" />
            <IsGripperHolding arm="{plan_active_arm}" />
          </Sequence>
        </RetryUntilSuccessful>
      </Sequence>

      <!-- Last resort -->
      <SafeAbort reason="all_recovery_failed" />

    </Fallback>
  </BehaviorTree>
</root>
)";
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
