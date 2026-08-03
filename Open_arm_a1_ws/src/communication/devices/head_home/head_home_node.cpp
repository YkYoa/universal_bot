// Plain ROS2 service node exposing "go home" for the neck (pan/tilt),
// callable by anything that talks to ROS2 (rosbridge, an Android app, other
// nodes) without needing to construct a FollowJointTrajectory goal by hand.
//
// Uses std_srvs::srv::Trigger (empty request, bool success + string
// message) rather than a custom srv type - this is exactly Trigger's shape,
// no need to invent a new message for it.
//
// Internally sends a single-point FollowJointTrajectory goal to
// head_controller (position 0.0 rad on both joints = the calibrated
// "center"/home position - see head_calibration.yaml's center_deg and the
// STM32 firmware's own HOME command for the raw-PWM equivalent used when
// something talks to the board directly instead of through ROS2).
//
// Run: ros2 run communication_devices head_home_node

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;

namespace {
constexpr double kHomeMoveDurationS = 2.0;
const std::vector<std::string> kJointNames = {
  "openarm_body_neck_joint", "openarm_body_head_joint"};
}  // namespace

class HeadHomeNode : public rclcpp::Node
{
public:
  HeadHomeNode() : Node("head_home_node")
  {
    action_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
      this, "/head_controller/follow_joint_trajectory");

    service_ = create_service<std_srvs::srv::Trigger>(
      "head/go_home",
      std::bind(&HeadHomeNode::handleGoHome, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(get_logger(), "head_home_node ready - call head/go_home to center pan/tilt");
  }

private:
  void handleGoHome(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (!action_client_->wait_for_action_server(std::chrono::seconds(2))) {
      response->success = false;
      response->message = "head_controller action server not available";
      return;
    }

    FollowJointTrajectory::Goal goal;
    goal.trajectory.joint_names = kJointNames;
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = {0.0, 0.0};
    point.time_from_start = rclcpp::Duration::from_seconds(kHomeMoveDurationS);
    goal.trajectory.points.push_back(point);

    auto goal_future = action_client_->async_send_goal(goal);

    // Block this service call until the goal is accepted/rejected, then
    // until it finishes - a synchronous "go home and tell me when done" is
    // exactly what callers of a simple Trigger service expect.
    if (rclcpp::spin_until_future_complete(get_node_base_interface(), goal_future) !=
        rclcpp::FutureReturnCode::SUCCESS) {
      response->success = false;
      response->message = "Failed to send goal to head_controller";
      return;
    }

    auto goal_handle = goal_future.get();
    if (!goal_handle) {
      response->success = false;
      response->message = "head_controller rejected the goal";
      return;
    }

    auto result_future = action_client_->async_get_result(goal_handle);
    if (rclcpp::spin_until_future_complete(get_node_base_interface(), result_future) !=
        rclcpp::FutureReturnCode::SUCCESS) {
      response->success = false;
      response->message = "Failed waiting for head_controller result";
      return;
    }

    auto result = result_future.get();
    response->success = (result.code == rclcpp_action::ResultCode::SUCCEEDED);
    response->message = response->success ? "Head homed" : "head_controller did not report success";
  }

  rclcpp_action::Client<FollowJointTrajectory>::SharedPtr action_client_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr service_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HeadHomeNode>());
  rclcpp::shutdown();
  return 0;
}
