#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// plan_to_pose.hpp
//
// Async action node: sends a Cartesian EE goal to MoveIt2's move_group action
// server with plan_only=true and returns SUCCESS if a valid plan is found.
// The planned trajectory is saved to the output_trajectory port.
//
// Inherits from BT::RosActionNode<moveit_msgs::action::MoveGroup>
//
// Blackboard reads:
//   BB_VEL_SCALE   double   velocity scaling forwarded to MoveIt
//   BB_ACC_SCALE   double   acceleration scaling
//
// Input ports:
//   arm              "left_arm" | "right_arm"
//   target_pose      geometry_msgs::msg::PoseStamped (from VLA BB key)
//   velocity_scaling double (optional, overrides BB_VEL_SCALE)
//   position_only    bool   (if true, orientation constraint is ignored)
//
// Output ports:
//   output_trajectory moveit_msgs::msg::RobotTrajectory
// ─────────────────────────────────────────────────────────────────────────────
#include "behaviortree_ros2/bt_action_node.hpp"
#include "moveit_msgs/action/move_group.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "shape_msgs/msg/solid_primitive.hpp"
#include "moveit_msgs/msg/constraints.hpp"
#include "moveit_msgs/msg/position_constraint.hpp"
#include "moveit_msgs/msg/orientation_constraint.hpp"
#include "moveit_msgs/msg/bounding_volume.hpp"
#include "moveit_msgs/msg/robot_trajectory.hpp"
#include "bt_executor/blackboard_keys.hpp"

namespace bt_executor {

class PlanToPose
  : public BT::RosActionNode<moveit_msgs::action::MoveGroup>
{
public:
  using MoveGroupAction = moveit_msgs::action::MoveGroup;

  PlanToPose(const std::string & name,
             const BT::NodeConfig & config,
             const BT::RosNodeParams & params)
  : BT::RosActionNode<MoveGroupAction>(name, config, params) {}

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<std::string>("arm", "left_arm | right_arm"),
      BT::InputPort<geometry_msgs::msg::PoseStamped>("target_pose"),
      BT::InputPort<double>("velocity_scaling", 0.3, "Velocity scaling"),
      BT::InputPort<double>("acceleration_scaling", 0.3, "Acceleration scaling"),
      BT::InputPort<bool>("position_only", false,
        "If true, skip orientation constraint"),
      BT::InputPort<double>("pos_tolerance", 0.01, "Position tolerance [m]"),
      BT::InputPort<double>("orient_tolerance", 0.5, "Orientation tolerance [rad]"),
      BT::OutputPort<moveit_msgs::msg::RobotTrajectory>("output_trajectory", "The planned trajectory"),
    });
  }

  // Called once when the node first enters RUNNING to build the goal message.
  bool setGoal(Goal & goal) override;

  // Called when the action server returns a result.
  BT::NodeStatus onResultReceived(const WrappedResult & result) override;

  // Called on action failure / abort.
  BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override;

  // Optional: called each tick while RUNNING to inspect feedback.
  BT::NodeStatus onFeedback(
    const std::shared_ptr<const Feedback> /*feedback*/) override
  {
    return BT::NodeStatus::RUNNING;  // keep going
  }
};

}  // namespace bt_executor
