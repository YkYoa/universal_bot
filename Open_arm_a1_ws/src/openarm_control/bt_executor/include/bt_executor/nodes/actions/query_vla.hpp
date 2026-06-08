#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// query_vla.hpp
//
// Action node (asynchronous ROS 2 Action): called during the recovery path when
// the BT needs a fresh plan from the VLA layer.
//
// This is now promoted to a RosActionNode wrapping the custom action 
// openarm_messages::action::QueryVLA action client. It communicates with the
// VLA bridge Action Server, which runs image acquisition and VLM inference, 
// and returns the target PoseStamped.
//
// XML usage:
//   <QueryVLA task="{vla_task_name}" />
// ─────────────────────────────────────────────────────────────────────────────
#include "behaviortree_ros2/bt_action_node.hpp"
#include "openarm_messages/action/query_vla.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "bt_executor/blackboard_keys.hpp"

namespace bt_executor {

class QueryVLA : public BT::RosActionNode<openarm_messages::action::QueryVLA>
{
public:
  using QueryVLAAction = openarm_messages::action::QueryVLA;

  QueryVLA(const std::string & name,
           const BT::NodeConfig & config,
           const BT::RosNodeParams & params)
  : BT::RosActionNode<QueryVLAAction>(name, config, params) {}

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<std::string>("task",
        "Task description forwarded to the VLA bridge for context"),
    });
  }

  bool setGoal(Goal & goal) override
  {
    // Consume the replan flag
    config().blackboard->set(BB_REPLAN_NEEDED, false);

    // Guard: don't retry forever
    int count = 0;
    config().blackboard->get(BB_RECOVERY_COUNT, count);
    ++count;
    config().blackboard->set(BB_RECOVERY_COUNT, count);

    if (count > MAX_RECOVERY_ATTEMPTS) {
      RCLCPP_ERROR(logger(),
        "Recovery count %d exceeds max (%d) — aborting VLA query",
        count, MAX_RECOVERY_ATTEMPTS);
      return false;
    }

    std::string task;
    if (!getInput("task", task)) {
      RCLCPP_ERROR(logger(), "QueryVLA: task port not set!");
      return false;
    }
    goal.task = task;

    RCLCPP_INFO(logger(),
      "Requesting VLA replan for task='%s' (attempt %d/%d)",
      task.c_str(), count, MAX_RECOVERY_ATTEMPTS);

    config().blackboard->set(BB_STATUS_MSG,
      std::string("replanning: attempt ") + std::to_string(count));

    return true;
  }

  BT::NodeStatus onResultReceived(const WrappedResult & result) override
  {
    if (result.result->success) {
      RCLCPP_INFO(logger(), "VLA query action succeeded!");

      // Update the blackboard with target pose from VLA bridge
      config().blackboard->set(BB_GRASP_POSE, result.result->target_pose);
      config().blackboard->set(BB_NEW_GOAL_READY, false); // Consumed
      config().blackboard->set(BB_RECOVERY_COUNT, 0);    // Reset count on success
      config().blackboard->set(BB_GOAL_CHANGED, false);
      
      int stamp = 0;
      config().blackboard->get(BB_GOAL_STAMP, stamp);
      config().blackboard->set(BB_GOAL_STAMP, stamp + 1);

      config().blackboard->set(BB_STATUS_MSG, std::string("replanning: success"));
      return BT::NodeStatus::SUCCESS;
    }

    RCLCPP_ERROR(logger(), "VLA query action server returned failure: %s", 
                 result.result->error_message.c_str());
    config().blackboard->set(BB_STATUS_MSG, std::string("replanning: failed: ") + result.result->error_message);
    config().blackboard->set(BB_REPLAN_NEEDED, true);
    return BT::NodeStatus::FAILURE;
  }

  BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override
  {
    RCLCPP_ERROR(logger(), "QueryVLA Action Node failure: %s", BT::toStr(error));
    config().blackboard->set(BB_STATUS_MSG, std::string("replanning: action failure"));
    config().blackboard->set(BB_REPLAN_NEEDED, true);
    return BT::NodeStatus::FAILURE;
  }

  BT::NodeStatus onFeedback(const std::shared_ptr<const Feedback> feedback) override
  {
    RCLCPP_INFO(logger(), "QueryVLA feedback: %s", feedback->status.c_str());
    return BT::NodeStatus::RUNNING;
  }
};

}  // namespace bt_executor
