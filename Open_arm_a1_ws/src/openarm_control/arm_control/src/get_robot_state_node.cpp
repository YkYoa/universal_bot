#include "arm_control/get_robot_state.hpp"

GetRobotStateNode::GetRobotStateNode()
: Node("get_robot_state_node")
{
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", 10,
    std::bind(&GetRobotStateNode::jointStateCallback, this, std::placeholders::_1));
}

void GetRobotStateNode::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  joint_names_ = msg->name;
  joint_positions_ = msg->position;
  joint_state_received_ = true;
}

bool GetRobotStateNode::hasJointState() const
{
  return joint_state_received_;
}

const std::vector<std::string>& GetRobotStateNode::getJointNames() const
{
  return joint_names_;
}

const std::vector<double>& GetRobotStateNode::getJointPositions() const
{
  return joint_positions_;
}

std::unique_ptr<tf2_ros::Buffer>& GetRobotStateNode::getTfBuffer()
{
  return tf_buffer_;
}
