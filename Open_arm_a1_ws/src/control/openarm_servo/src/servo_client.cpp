#include "openarm_servo/servo_client.hpp"

namespace openarm_servo {

ServoClient::ServoClient(const rclcpp::Node::SharedPtr& node, const std::string& side)
  : node_(node), side_(side)
{
  const std::string ns = "/" + side_ + "_arm_servo";

  twist_pub_ = node_->create_publisher<geometry_msgs::msg::TwistStamped>(
    ns + "/delta_twist_cmds", rclcpp::SystemDefaultsQoS());
  joint_jog_pub_ = node_->create_publisher<control_msgs::msg::JointJog>(
    ns + "/delta_joint_cmds", rclcpp::SystemDefaultsQoS());
  status_sub_ = node_->create_subscription<moveit_msgs::msg::ServoStatus>(
    ns + "/status", rclcpp::SystemDefaultsQoS(),
    [this](const moveit_msgs::msg::ServoStatus::ConstSharedPtr& msg) { onStatus(msg); });
}

void ServoClient::jogCartesian(double dx, double dy, double dz, double wx, double wy, double wz,
                               const std::string& frame_id)
{
  geometry_msgs::msg::TwistStamped msg;
  msg.header.stamp = node_->now();
  msg.header.frame_id = frame_id;
  msg.twist.linear.x = dx;
  msg.twist.linear.y = dy;
  msg.twist.linear.z = dz;
  msg.twist.angular.x = wx;
  msg.twist.angular.y = wy;
  msg.twist.angular.z = wz;
  twist_pub_->publish(msg);
}

void ServoClient::jogJoint(const std::vector<std::string>& joint_names,
                           const std::vector<double>& velocities)
{
  control_msgs::msg::JointJog msg;
  msg.header.stamp = node_->now();
  msg.joint_names = joint_names;
  msg.velocities = velocities;
  joint_jog_pub_->publish(msg);
}

void ServoClient::stop()
{
  geometry_msgs::msg::TwistStamped msg;
  msg.header.stamp = node_->now();
  twist_pub_->publish(msg);
}

int8_t ServoClient::lastStatusCode() const
{
  std::lock_guard<std::mutex> lock(status_mutex_);
  return last_status_code_;
}

std::string ServoClient::lastStatusMessage() const
{
  std::lock_guard<std::mutex> lock(status_mutex_);
  return last_status_message_;
}

void ServoClient::setStatusCallback(StatusCallback callback)
{
  std::lock_guard<std::mutex> lock(status_mutex_);
  status_callback_ = std::move(callback);
}

void ServoClient::onStatus(const moveit_msgs::msg::ServoStatus::ConstSharedPtr& msg)
{
  StatusCallback callback_copy;
  {
    std::lock_guard<std::mutex> lock(status_mutex_);
    last_status_code_ = msg->code;
    last_status_message_ = msg->message;
    callback_copy = status_callback_;
  }
  if (callback_copy) {
    callback_copy(msg->code, msg->message);
  }
}

}  // namespace openarm_servo
