#include "arm_control/get_robot_state.hpp"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <algorithm>


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GetRobotStateNode>();

  std::cout << "Waiting for robot state from /joint_states and TF (/tf)..." << std::endl;

  auto start_time = node->now();
  double timeout_sec = 4.0;

  std::vector<double> left_arm_joints;
  std::vector<double> right_arm_joints;
  bool has_left_ee = false;
  bool has_right_ee = false;
  geometry_msgs::msg::TransformStamped left_ee_tf;
  geometry_msgs::msg::TransformStamped right_ee_tf;
  double left_gripper = 0.0;
  double right_gripper = 0.0;

  rclcpp::WallRate rate(10);
  while (rclcpp::ok()) {
    rclcpp::spin_some(node);

    double elapsed = (node->now() - start_time).seconds();

    if (node->hasJointState()) {
      const auto& names = node->getJointNames();
      const auto& positions = node->getJointPositions();

      // Extract left arm joints
      std::vector<double> la_vals;
      for (int i = 1; i <= 7; ++i) {
        std::string name = "openarm_left_joint" + std::to_string(i);
        auto it = std::find(names.begin(), names.end(), name);
        if (it != names.end()) {
          la_vals.push_back(positions[std::distance(names.begin(), it)]);
        }
      }
      if (la_vals.size() == 7) {
        left_arm_joints = la_vals;
      }

      // Extract right arm joints
      std::vector<double> ra_vals;
      for (int i = 1; i <= 7; ++i) {
        std::string name = "openarm_right_joint" + std::to_string(i);
        auto it = std::find(names.begin(), names.end(), name);
        if (it != names.end()) {
          ra_vals.push_back(positions[std::distance(names.begin(), it)]);
        }
      }
      if (ra_vals.size() == 7) {
        right_arm_joints = ra_vals;
      }

      // Grippers
      auto it_l = std::find(names.begin(), names.end(), "openarm_left_finger_joint1");
      if (it_l != names.end()) {
        left_gripper = positions[std::distance(names.begin(), it_l)];
      }
      auto it_r = std::find(names.begin(), names.end(), "openarm_right_finger_joint1");
      if (it_r != names.end()) {
        right_gripper = positions[std::distance(names.begin(), it_r)];
      }
    }

    // TF lookup
    if (!has_left_ee) {
      try {
        left_ee_tf = node->getTfBuffer()->lookupTransform(
          "world", "openarm_left_hand_tcp", tf2::TimePointZero);
        has_left_ee = true;
      } catch (const tf2::TransformException & ex) {
        // Wait
      }
    }

    if (!has_right_ee) {
      try {
        right_ee_tf = node->getTfBuffer()->lookupTransform(
          "world", "openarm_right_hand_tcp", tf2::TimePointZero);
        has_right_ee = true;
      } catch (const tf2::TransformException & ex) {
        // Wait
      }
    }

    if (!left_arm_joints.empty() && !right_arm_joints.empty() && has_left_ee && has_right_ee) {
      break;
    }

    if (elapsed > timeout_sec) {
      RCLCPP_WARN(node->get_logger(), "Timeout waiting for complete robot state. Printing whatever was retrieved.");
      break;
    }

    rate.sleep();
  }

  std::cout << "\n==================================================\n";
  std::cout << "      ROBOT CURRENT STATE                           \n";
  std::cout << "==================================================\n\n";

  std::cout << std::fixed << std::setprecision(2);

  // ── Left Arm ──
  std::cout << "# Left Arm State:\n";
  if (!left_arm_joints.empty()) {
    std::cout << "  laAngle: ";
    for (size_t i = 0; i < left_arm_joints.size(); ++i) {
      std::cout << (left_arm_joints[i] * 180.0 / M_PI);
      if (i < left_arm_joints.size() - 1) std::cout << ", ";
    }
    std::cout << "  # (Degrees)\n";

    std::cout << "  laAngleRad: " << std::setprecision(4);
    for (size_t i = 0; i < left_arm_joints.size(); ++i) {
      std::cout << left_arm_joints[i];
      if (i < left_arm_joints.size() - 1) std::cout << ", ";
    }
    std::cout << "  # (Radians)\n" << std::setprecision(2);
  } else {
    std::cout << "  laAngle: # Not found in /joint_states\n";
  }

  if (has_left_ee) {
    std::cout << "  laJoint: " << std::setprecision(5)
              << left_ee_tf.transform.translation.x << ", "
              << left_ee_tf.transform.translation.y << ", "
              << left_ee_tf.transform.translation.z << ", "
              << left_ee_tf.transform.rotation.x << ", "
              << left_ee_tf.transform.rotation.y << ", "
              << left_ee_tf.transform.rotation.z << ", "
              << left_ee_tf.transform.rotation.w << "  # (x, y, z, qx, qy, qz, qw)\n"
              << std::setprecision(2);
  } else {
    std::cout << "  laJoint: # Not found in TF transforms\n";
  }
  std::cout << "  lhGrasp: " << std::setprecision(4) << left_gripper << "\n\n";

  // ── Right Arm ──
  std::cout << "# Right Arm State:\n";
  if (!right_arm_joints.empty()) {
    std::cout << "  raAngle: ";
    for (size_t i = 0; i < right_arm_joints.size(); ++i) {
      std::cout << (right_arm_joints[i] * 180.0 / M_PI);
      if (i < right_arm_joints.size() - 1) std::cout << ", ";
    }
    std::cout << "  # (Degrees)\n";

    std::cout << "  raAngleRad: " << std::setprecision(4);
    for (size_t i = 0; i < right_arm_joints.size(); ++i) {
      std::cout << right_arm_joints[i];
      if (i < right_arm_joints.size() - 1) std::cout << ", ";
    }
    std::cout << "  # (Radians)\n" << std::setprecision(2);
  } else {
    std::cout << "  raAngle: # Not found in /joint_states\n";
  }

  if (has_right_ee) {
    std::cout << "  raJoint: " << std::setprecision(5)
              << right_ee_tf.transform.translation.x << ", "
              << right_ee_tf.transform.translation.y << ", "
              << right_ee_tf.transform.translation.z << ", "
              << right_ee_tf.transform.rotation.x << ", "
              << right_ee_tf.transform.rotation.y << ", "
              << right_ee_tf.transform.rotation.z << ", "
              << right_ee_tf.transform.rotation.w << "  # (x, y, z, qx, qy, qz, qw)\n"
              << std::setprecision(2);
  } else {
    std::cout << "  raJoint: # Not found in TF transforms\n";
  }
  std::cout << "  rhGrasp: " << std::setprecision(4) << right_gripper << "\n\n";

  std::cout << "==================================================\n" << std::endl;

  rclcpp::shutdown();
  return 0;
}
