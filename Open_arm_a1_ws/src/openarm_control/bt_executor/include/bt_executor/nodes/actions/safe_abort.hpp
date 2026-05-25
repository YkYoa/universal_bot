#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// safe_abort.hpp
//
// Synchronous action: called at the root Fallback's last child when all
// recovery attempts have failed.
//
// Responsibilities:
//   1. Halt the JointTrajectory controllers (stop motion immediately).
//   2. Open both grippers (drop object safely).
//   3. Write an error status to the blackboard.
//   4. Publish a /bt_executor/fault alert for the UI.
//   5. Returns FAILURE so the root tree is marked failed — the tick loop
//      will then halt the tree and wait for a new goal.
//
// This node is synchronous (SyncActionNode) because once you decide to abort
// there is no reason to yield — just execute the halt sequence blocking.
//
// The shared rclcpp::Node is provided via the blackboard key "ros_node"
// (an rclcpp::Node::SharedPtr written by BTExecutorNode::initialize()).
// This avoids creating a new Node per BT instance, which caused node-name
// collisions when the BT factory registers multiple instances.
//
// XML usage:
//   <SafeAbort reason="all_recovery_failed" />
// ─────────────────────────────────────────────────────────────────────────────
#include "behaviortree_cpp/behavior_tree.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "bt_executor/blackboard_keys.hpp"

namespace bt_executor {

class SafeAbort : public BT::SyncActionNode
{
public:
  // Blackboard key under which BTExecutorNode writes its shared node pointer.
  static constexpr const char* BB_ROS_NODE = "ros_node";

  SafeAbort(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config), last_seen_stamp_(-1), logged_(false)
  {}

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<std::string>("reason", "human-readable abort reason")
    };
  }

  BT::NodeStatus tick() override
  {
    std::string reason = "unknown";
    getInput("reason", reason);

    // ── Check if a new goal stamp has been set ───────────────────────────
    int current_stamp = 0;
    if (config().blackboard->get(BB_GOAL_STAMP, current_stamp)) {
      if (current_stamp != last_seen_stamp_) {
        last_seen_stamp_ = current_stamp;
        logged_ = false;
      }
    }

    if (!logged_) {
      // ── Resolve the shared ROS node ───────────────────────────────────────
      rclcpp::Node::SharedPtr node;
      if (!config().blackboard->get(BB_ROS_NODE, node) || !node) {
        // Fallback: log via the global rclcpp logger so at least something is visible.
        RCLCPP_ERROR(rclcpp::get_logger("SafeAbort"),
          "═══ SAFE ABORT: %s ═══ (no ros_node on blackboard — fault topic not published)",
          reason.c_str());
      } else {
        RCLCPP_ERROR(node->get_logger(),
          "═══ SAFE ABORT: %s ═══", reason.c_str());

        // Lazily create publisher (only once per node lifecycle).
        if (!fault_pub_) {
          fault_pub_ = node->create_publisher<std_msgs::msg::String>(
            "/bt_executor/fault", rclcpp::SystemDefaultsQoS());
        }

        // ── Publish fault alert ─────────────────────────────────────────────
        std_msgs::msg::String msg;
        msg.data = reason;
        fault_pub_->publish(msg);
      }
      logged_ = true;
    }

    config().blackboard->set(BB_STATUS_MSG,
      std::string("ABORTED: ") + reason);

    // ── TODO: Halt controllers ────────────────────────────────────────────
    // Call /controller_manager/switch_controller to deactivate arm controllers,
    // or publish an empty cancellation goal to the FollowJointTrajectory
    // action server.
    //
    // Example (requires a node handle):
    //   auto cancel_client = node->create_client<action_msgs::srv::CancelGoal>(
    //     "/left_arm_controller/follow_joint_trajectory/_action/cancel_goal");
    //   cancel_client->async_send_request(...);
    // 
    // ── TODO: Open grippers ───────────────────────────────────────────────
    // Send open position (0.044) to both gripper controllers directly.

    // Reset recovery counter for next task
    config().blackboard->set(BB_RECOVERY_COUNT, 0);
    config().blackboard->set(BB_REPLAN_NEEDED,  false);

    // Return FAILURE so root Fallback is marked failed and tree halts
    return BT::NodeStatus::FAILURE;
  }

private:
  // Publisher is created lazily so we don't need the node at construction time.
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr fault_pub_;
  int last_seen_stamp_;
  bool logged_;
};

}  // namespace bt_executor
