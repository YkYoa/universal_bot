#ifndef ARM_CONTROL__GET_ROBOT_STATE_HPP_
#define ARM_CONTROL__GET_ROBOT_STATE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <vector>
#include <string>
#include <memory>

class GetRobotStateNode : public rclcpp::Node
{
public:
  GetRobotStateNode();
  virtual ~GetRobotStateNode() = default;

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);

  bool hasJointState() const;
  const std::vector<std::string>& getJointNames() const;
  const std::vector<double>& getJointPositions() const;

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
