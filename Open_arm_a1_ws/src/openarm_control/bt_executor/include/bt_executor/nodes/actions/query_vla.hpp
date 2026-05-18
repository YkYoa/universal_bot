#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// query_vla.hpp
//
// Action node (synchronous-ish): called during the recovery path when the BT
// needs a fresh plan from the VLA layer.
//
// In the current skeleton this is a StatefulActionNode (not a ROS action)
// because the VLA bridge is a separate process that writes directly to the
// blackboard.  When you implement the real VLA bridge, two options:
//
//   Option A (recommended): Promote to RosActionNode wrapping a custom
//     openarm_msgs::action::QueryVLA action server running inside your
//     VLA process (Python, probably).  The action server runs the
//     VLM inference, updates the blackboard over the response, and
//     returns success/failure.
//
//   Option B: Keep this as a service call
//     (openarm_msgs::srv::QueryVLA) if replanning is fast enough that
//     blocking for ~1-2 s is acceptable.
//
// What this skeleton does:
//   - Sets BB_REPLAN_NEEDED = false (consumed)
//   - Increments BB_RECOVERY_COUNT
//   - If recovery count > MAX_RECOVERY_ATTEMPTS → returns FAILURE (triggers
//     SafeAbort)
//   - Otherwise returns SUCCESS so the retry sequence can proceed.
//     (The actual new poses must arrive on the blackboard from the VLA bridge
//     before or during this node's execution.)
//
// XML usage:
//   <QueryVLA task="{vla_task_name}" />
// ─────────────────────────────────────────────────────────────────────────────
#include "behaviortree_cpp/behavior_tree.h"
#include "rclcpp/rclcpp.hpp"
#include "bt_executor/blackboard_keys.hpp"

namespace bt_executor {

class QueryVLA : public BT::StatefulActionNode
{
public:
  QueryVLA(const std::string & name, const BT::NodeConfig & config)
  : BT::StatefulActionNode(name, config) {}

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<std::string>("task",
        "Task description forwarded to the VLA bridge for context"),
    };
  }

  BT::NodeStatus onStart() override
  {
    // Consume the replan flag
    config().blackboard->set(BB_REPLAN_NEEDED, false);

    // Guard: don't retry forever
    int count = 0;
    config().blackboard->get(BB_RECOVERY_COUNT, count);
    ++count;
    config().blackboard->set(BB_RECOVERY_COUNT, count);

    if (count > MAX_RECOVERY_ATTEMPTS) {
      RCLCPP_ERROR(rclcpp::get_logger("QueryVLA"),
        "Recovery count %d exceeds max (%d) — aborting",
        count, MAX_RECOVERY_ATTEMPTS);
      return BT::NodeStatus::FAILURE;
    }

    std::string task;
    getInput("task", task);

    // ── TODO: Trigger the VLA bridge to replan ──────────────────────────
    //
    // Option A (ROS action): send a goal to /vla_bridge/query with the
    //   task string + current joint state + camera frame.  The bridge runs
    //   VLM inference, writes new BB_GRASP_POSE / BB_PLACE_POSE, and
    //   returns success.  Promote this class to RosActionNode<QueryVLAAction>
    //   and move the logic into setGoal / onResultReceived.
    //
    // Option B (ROS service):
    //   auto client = node->create_client<openarm_msgs::srv::QueryVLA>("/vla_bridge/query");
    //   auto req = std::make_shared<openarm_msgs::srv::QueryVLA::Request>();
    //   req->task = task;
    //   auto future = client->async_send_request(req);
    //   // handle in onRunning() while future is pending
    //
    // For now: log and immediately succeed (assumes bridge will update BB)
    RCLCPP_INFO(rclcpp::get_logger("QueryVLA"),
      "Requesting replan for task='%s' (attempt %d/%d)",
      task.c_str(), count, MAX_RECOVERY_ATTEMPTS);

    config().blackboard->set(BB_STATUS_MSG,
      std::string("replanning: attempt ") + std::to_string(count));

    return BT::NodeStatus::RUNNING;
  }

  BT::NodeStatus onRunning() override
  {
    // TODO: Poll the async VLA response here.
    // Check if BB_GRASP_POSE has been updated with a fresh timestamp.
    // For now: succeed immediately (stub).

    bool new_goal = false;
    config().blackboard->get(BB_NEW_GOAL_READY, new_goal);
    if (new_goal) {
      config().blackboard->set(BB_NEW_GOAL_READY, false);
      config().blackboard->set(BB_RECOVERY_COUNT, 0);  // reset on fresh plan
      return BT::NodeStatus::SUCCESS;
    }

    // TODO: remove this immediate return once you wire the real bridge.
    return BT::NodeStatus::SUCCESS;
  }

  void onHalted() override
  {
    RCLCPP_WARN(rclcpp::get_logger("QueryVLA"), "QueryVLA halted");
  }
};

}  // namespace bt_executor
