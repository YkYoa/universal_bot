#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

using namespace std::chrono_literals;

class TestRealtimeNode : public rclcpp::Node
{
public:
  TestRealtimeNode()
  : Node("test_realtime_node")
  {
    // Publisher to the target pose topic of the BT executor
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "/bt_executor/vla_grasp_pose", rclcpp::SystemDefaultsQoS());

    RCLCPP_INFO(get_logger(), "Test Realtime Node initialized.");

    // Start orchestrating the test after a delay
    timer_ = create_wall_timer(
      1s, std::bind(&TestRealtimeNode::orchestrate_test, this));
  }

private:
  void orchestrate_test()
  {
    timer_->cancel(); // Run once

    // Step 1: Wait 20 seconds to ensure MoveIt and BT are fully up and ready
    RCLCPP_INFO(get_logger(), "Waiting 20 seconds for systems to initialize...");
    std::this_thread::sleep_for(20s);

    // Step 2: Publish target Pose 1 (first goal position)
    geometry_msgs::msg::PoseStamped pose1;
    pose1.header.frame_id = "openarm_body_link0";
    pose1.header.stamp = now();
    pose1.pose.position.x = -0.52;
    pose1.pose.position.y = -0.20;
    pose1.pose.position.z = 0.55;
    pose1.pose.orientation.x = 0.0;
    pose1.pose.orientation.y = -0.7071;
    pose1.pose.orientation.z = 0.0;
    pose1.pose.orientation.w = 0.7071;

    RCLCPP_INFO(get_logger(), "Publishing first target goal: Pose 1 (x: -0.52, y: -0.20, z: 0.55)");
    pose_pub_->publish(pose1);

    // Step 3: Wait 2 seconds so the arm plans and starts moving towards Pose 1
    RCLCPP_INFO(get_logger(), "Waiting 2 seconds while arm moves towards Pose 1...");
    std::this_thread::sleep_for(2s);

    // Step 4: Publish target Pose 2 (interrupting goal position)
    geometry_msgs::msg::PoseStamped pose2;
    pose2.header.frame_id = "openarm_body_link0";
    pose2.header.stamp = now();
    pose2.pose.position.x = -0.45;
    pose2.pose.position.y = -0.22;
    pose2.pose.position.z = 0.58;
    pose2.pose.orientation.x = 0.0;
    pose2.pose.orientation.y = -0.7071;
    pose2.pose.orientation.z = 0.0;
    pose2.pose.orientation.w = 0.7071;

    RCLCPP_INFO(get_logger(), "Publishing SECOND target goal (INTERRUPTING): Pose 2 (x: -0.45, y: -0.22, z: 0.58)");
    pose_pub_->publish(pose2);

    RCLCPP_INFO(get_logger(), "Interruption command sent! Watch the BT executor logs to observe preemption.");
  }

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TestRealtimeNode>());
  rclcpp::shutdown();
  return 0;
}
