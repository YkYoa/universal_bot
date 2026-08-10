#include "sequence_executor/hand_gripper_client.hpp"

namespace sequence_executor {

namespace {

std::string sideOf(const std::string& arm)
{
  return (arm == "right_arm") ? "right" : "left";
}

}  // namespace

HandGripperClient::HandGripperClient(const rclcpp::Node::SharedPtr& node)
: node_(node), logger_(rclcpp::get_logger("HandGripperClient"))
{
}

rclcpp_action::Client<HandGripperClient::FJT>::SharedPtr HandGripperClient::clientFor(const std::string& action_name)
{
  auto it = clients_.find(action_name);
  if (it != clients_.end()) {
    return it->second;
  }
  auto client = rclcpp_action::create_client<FJT>(node_, action_name);
  clients_[action_name] = client;
  return client;
}

void HandGripperClient::sendFourJointGoal(
  const std::string& action_name, const std::vector<std::string>& joint_names, const std::vector<double>& positions,
  double duration, ResultCallback callback)
{
  if (positions.size() != 4) {
    RCLCPP_ERROR(logger_, "%s: positions must have exactly 4 values (got %zu)", action_name.c_str(), positions.size());
    callback(false, "positions must have exactly 4 values");
    return;
  }

  auto client = clientFor(action_name);
  if (!client->wait_for_action_server(std::chrono::seconds(15))) {
    RCLCPP_ERROR(logger_, "%s not available", action_name.c_str());
    callback(false, action_name + " not available");
    return;
  }

  FJT::Goal goal;
  goal.trajectory.joint_names = joint_names;
  trajectory_msgs::msg::JointTrajectoryPoint pt;
  pt.positions = positions;
  pt.time_from_start.sec = static_cast<int32_t>(duration);
  pt.time_from_start.nanosec = static_cast<uint32_t>((duration - std::floor(duration)) * 1e9);
  goal.trajectory.points.push_back(pt);

  RCLCPP_INFO(
    logger_, "%s -> [%.4f, %.4f, %.4f, %.4f]", action_name.c_str(), positions[0], positions[1], positions[2],
    positions[3]);

  rclcpp_action::Client<FJT>::SendGoalOptions options;
  options.result_callback = [this, action_name, callback](const rclcpp_action::ClientGoalHandle<FJT>::WrappedResult& result) {
    if (result.code == rclcpp_action::ResultCode::SUCCEEDED &&
        result.result->error_code == FJT::Result::SUCCESSFUL) {
      callback(true, "");
      return;
    }
    RCLCPP_WARN(logger_, "%s failed error_code=%d", action_name.c_str(), result.result ? result.result->error_code : -1);
    callback(false, action_name + " failed");
  };
  client->async_send_goal(goal, options);
}

void HandGripperClient::sendSingleJointGoal(
  const std::string& action_name, const std::string& joint_name, double position, double duration,
  ResultCallback callback)
{
  auto client = clientFor(action_name);
  if (!client->wait_for_action_server(std::chrono::seconds(15))) {
    RCLCPP_ERROR(logger_, "%s not available", action_name.c_str());
    callback(false, action_name + " not available");
    return;
  }

  FJT::Goal goal;
  goal.trajectory.joint_names = {joint_name};
  trajectory_msgs::msg::JointTrajectoryPoint pt;
  pt.positions = {position};
  pt.time_from_start.sec = static_cast<int32_t>(duration);
  pt.time_from_start.nanosec = static_cast<uint32_t>((duration - std::floor(duration)) * 1e9);
  goal.trajectory.points.push_back(pt);

  RCLCPP_INFO(logger_, "%s -> %.4f", action_name.c_str(), position);

  rclcpp_action::Client<FJT>::SendGoalOptions options;
  options.result_callback = [this, action_name, callback](const rclcpp_action::ClientGoalHandle<FJT>::WrappedResult& result) {
    if (result.code == rclcpp_action::ResultCode::SUCCEEDED &&
        result.result->error_code == FJT::Result::SUCCESSFUL) {
      callback(true, "");
      return;
    }
    RCLCPP_WARN(logger_, "%s failed error_code=%d", action_name.c_str(), result.result ? result.result->error_code : -1);
    callback(false, action_name + " failed");
  };
  client->async_send_goal(goal, options);
}

void HandGripperClient::setHandYaw(
  const std::string& arm, const std::vector<double>& positions, double duration, ResultCallback callback)
{
  const std::string side = sideOf(arm);
  const std::string action_name = side + "_hand_j1_controller/follow_joint_trajectory";
  const std::vector<std::string> joints = {
    "openarm_" + side + "_j11", "openarm_" + side + "_j12", "openarm_" + side + "_j13", "openarm_" + side + "_j14"};
  sendFourJointGoal(action_name, joints, positions, duration, std::move(callback));
}

void HandGripperClient::setHandFlex(
  const std::string& arm, const std::vector<double>& positions, double duration, ResultCallback callback)
{
  const std::string side = sideOf(arm);
  const std::string action_name = side + "_hand_j2_controller/follow_joint_trajectory";
  const std::vector<std::string> joints = {
    "openarm_" + side + "_j21", "openarm_" + side + "_j22", "openarm_" + side + "_j23", "openarm_" + side + "_j24"};
  sendFourJointGoal(action_name, joints, positions, duration, std::move(callback));
}

void HandGripperClient::openGripper(
  const std::string& arm, double open_position, double duration, ResultCallback callback)
{
  const std::string side = sideOf(arm);
  const std::string action_name = side + "_gripper_controller/follow_joint_trajectory";
  const std::string joint = "openarm_" + side + "_finger_joint1";
  sendSingleJointGoal(action_name, joint, open_position, duration, std::move(callback));
}

void HandGripperClient::closeGripper(
  const std::string& arm, double close_position, double duration, ResultCallback callback)
{
  const std::string side = sideOf(arm);
  const std::string action_name = side + "_gripper_controller/follow_joint_trajectory";
  const std::string joint = "openarm_" + side + "_finger_joint1";
  sendSingleJointGoal(action_name, joint, close_position, duration, std::move(callback));
}

}  // namespace sequence_executor
