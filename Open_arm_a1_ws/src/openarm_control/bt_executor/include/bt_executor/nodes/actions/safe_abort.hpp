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
  SafeAbort(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config)
  {
    // Create a one-shot node handle for publishing the fault alert.
    // In a full implementation, pass in the shared rclcpp::Node instead.
    node_ = rclcpp::Node::make_shared("safe_abort_helper");
    fault_pub_ = node_->create_publisher<std_msgs::msg::String>(
      "/bt_executor/fault", rclcpp::SystemDefaultsQoS());
  }

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

    RCLCPP_ERROR(rclcpp::get_logger("SafeAbort"),
      "═══ SAFE ABORT: %s ═══", reason.c_str());

    config().blackboard->set(BB_STATUS_MSG,
      std::string("ABORTED: ") + reason);

    // ── TODO: Halt controllers ────────────────────────────────────────────
    // Call /controller_manager/switch_controller to deactivate arm controllers,
    // or publish an empty cancellation goal to the FollowJointTrajectory
    // action server.
    //
    // Example (requires a node handle):
    //   auto cancel_client = node_->create_client<action_msgs::srv::CancelGoal>(
    //     "/left_arm_controller/follow_joint_trajectory/_action/cancel_goal");
    //   cancel_client->async_send_request(...);

    // ── TODO: Open grippers ───────────────────────────────────────────────
    // Send open position (0.044) to both gripper controllers directly.

    // ── Publish fault alert ───────────────────────────────────────────────
    std_msgs::msg::String msg;
    msg.data = reason;
    fault_pub_->publish(msg);
    rclcpp::spin_some(node_);

    // Reset recovery counter for next task
    config().blackboard->set(BB_RECOVERY_COUNT, 0);
    config().blackboard->set(BB_REPLAN_NEEDED,  false);

    // Return FAILURE so root Fallback is marked failed and tree halts
    return BT::NodeStatus::FAILURE;
  }

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr fault_pub_;
};

}  // namespace bt_executor
