#ifndef ARM_CONTROL__GET_ROBOT_STATE_HPP_
#define ARM_CONTROL__GET_ROBOT_STATE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <vector>
#include <string>
#include <memory>

/// Backing node for the get_robot_state CLI tool: caches the latest
/// /joint_states message and exposes a TF buffer for EE pose lookups.
class GetRobotStateNode : public rclcpp::Node
{
public:
  /// Subscribes to /joint_states and sets up the TF buffer/listener.
  GetRobotStateNode();
  virtual ~GetRobotStateNode() = default;

  /// Subscription callback: caches the latest joint names/positions.
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);

  /// True once at least one /joint_states message has been received.
  bool hasJointState() const;
  /// Joint names from the latest /joint_states message.
  const std::vector<std::string>& getJointNames() const;
  /// Joint positions (parallel to getJointNames()) from the latest message.
  const std::vector<double>& getJointPositions() const;

  /// TF buffer for looking up end-effector transforms.
  std::unique_ptr<tf2_ros::Buffer>& getTfBuffer();

private:
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;

  bool joint_state_received_ = false;
  std::vector<std::string> joint_names_;
  std::vector<double> joint_positions_;
};

#endif  // ARM_CONTROL__GET_ROBOT_STATE_HPP_
