#pragma once
// -----------------------------------------------------------------------------
// servo_client.hpp
//
// Clean C++ API over one arm's moveit_servo instance (see servo.launch.py:
// each arm gets its own servo_node, namespaced "<side>_arm_servo") - callers
// send jogCartesian()/jogJoint() without needing to know moveit_servo's raw
// topic names or message types (geometry_msgs::TwistStamped,
// control_msgs::JointJog, moveit_msgs::ServoStatus). Same shape as this
// package's sibling client classes in sequence_executor (SkillClient,
// HandGripperClient, MotorEnableClient) - wrap the raw ROS interface once,
// let callers work in plain values.
//
// Does NOT talk to RobotSupervisor/motors_enabled_ or any FSM state - this
// is a thin transport wrapper only. A caller that needs "don't jog while
// FAULT/E-stopped" has to check that itself (e.g. via the ~/state topic)
// before calling jogCartesian()/jogJoint() - moveit_servo's own node has no
// idea this robot has an FSM at all.
// -----------------------------------------------------------------------------
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <control_msgs/msg/joint_jog.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <moveit_msgs/msg/servo_status.hpp>

namespace openarm_servo {

/// See file header comment.
class ServoClient
{
public:
  using StatusCallback = std::function<void(int8_t code, const std::string& message)>;

  /// `side` must be "left" or "right" - resolves to the matching
  /// "/<side>_arm_servo/..." topics servo.launch.py's servo_node_for()
  /// namespaces each instance under.
  ServoClient(const rclcpp::Node::SharedPtr& node, const std::string& side);

  /// Cartesian velocity jog: dx/dy/dz in m/s, wx/wy/wz in rad/s, expressed
  /// in `frame_id` (must be a frame moveit_servo can transform into its
  /// planning/EE frame - see servo_params.yaml's move_group_name). Values
  /// are NOT clamped here - servo_params.yaml's scale.linear/rotational is
  /// what actually limits them once received.
  void jogCartesian(double dx, double dy, double dz, double wx, double wy, double wz,
                    const std::string& frame_id);

  /// Joint-space velocity jog. `joint_names`/`velocities` must be the same
  /// length, one entry per joint being jogged (moveit_servo accepts a
  /// subset of the group's joints, not necessarily all of them).
  void jogJoint(const std::vector<std::string>& joint_names, const std::vector<double>& velocities);

  /// Publishes a single zero-velocity twist. moveit_servo's own
  /// incoming_command_timeout (servo_params.yaml) already halts on stale
  /// input, so this isn't the only way motion stops - it's for an explicit,
  /// immediate "stop jogging now" from the caller.
  void stop();

  /// Last status moveit_servo published on "/<side>_arm_servo/status"
  /// (moveit_msgs::msg::ServoStatus - NO_WARNING/DECELERATE_FOR_.../
  /// HALT_FOR_SINGULARITY/HALT_FOR_COLLISION/JOINT_BOUND). NO_WARNING with
  /// an empty message until the first one actually arrives.
  int8_t lastStatusCode() const;
  std::string lastStatusMessage() const;

  /// Invoked on every new ServoStatus, in addition to updating
  /// lastStatusCode()/lastStatusMessage() - e.g. for a caller that wants to
  /// react to HALT_FOR_SINGULARITY/HALT_FOR_COLLISION as they happen rather
  /// than by polling.
  void setStatusCallback(StatusCallback callback);

private:
  void onStatus(const moveit_msgs::msg::ServoStatus::ConstSharedPtr& msg);

  rclcpp::Node::SharedPtr node_;
  std::string side_;

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
  rclcpp::Publisher<control_msgs::msg::JointJog>::SharedPtr joint_jog_pub_;
  rclcpp::Subscription<moveit_msgs::msg::ServoStatus>::SharedPtr status_sub_;

  mutable std::mutex status_mutex_;
  int8_t last_status_code_ = moveit_msgs::msg::ServoStatus::NO_WARNING;
  std::string last_status_message_;
  StatusCallback status_callback_;
};

}  // namespace openarm_servo
