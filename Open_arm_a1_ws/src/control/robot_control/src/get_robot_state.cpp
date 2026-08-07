#include "robot_control/get_robot_state.hpp"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <algorithm>


namespace {

// amazing_hand's alias command joints (see ahand.ros2_control.xacro):
// j11..j14 = yaw/abduction per finger 1-4 (4 = thumb), j21..j24 = proximal
// flexion per finger. These are the stable SRDF/ros2_control names -
// hand_kinematics_node's mapping to the real closed-loop linkage joints is
// per-build/non-derivable (see openarm_amazing_hand memory), so the alias
// joints are what's actually meaningful to print here.
const std::vector<std::string> AHAND_JOINT_SUFFIXES = {
  "j11", "j12", "j13", "j14", "j21", "j22", "j23", "j24"
};

// Detects which end effector is live from whichever joint names actually
// show up in /joint_states, instead of requiring this tool to be told
// ee_type separately - it already comes from the launch that's running.
struct EndEffectorState {
  bool is_gripper = false;
  bool is_ahand = false;
  double gripper_value = 0.0;
  std::vector<double> ahand_values;  // parallel to AHAND_JOINT_SUFFIXES, NaN if missing
};

EndEffectorState detectEndEffector(const std::string& side_prefix,
                                    const std::vector<std::string>& names,
                                    const std::vector<double>& positions) {
  EndEffectorState state;

  auto it_gripper = std::find(names.begin(), names.end(), side_prefix + "finger_joint1");
  if (it_gripper != names.end()) {
    state.is_gripper = true;
    state.gripper_value = positions[std::distance(names.begin(), it_gripper)];
  }

  for (const auto& suffix : AHAND_JOINT_SUFFIXES) {
    auto it = std::find(names.begin(), names.end(), side_prefix + suffix);
    if (it != names.end()) {
      state.is_ahand = true;
      state.ahand_values.push_back(positions[std::distance(names.begin(), it)]);
    } else {
      state.ahand_values.push_back(std::nan(""));
    }
  }

  return state;
}

void printHandState(const std::string& label, const EndEffectorState& state) {
  if (state.is_ahand) {
    std::cout << "  " << label << "Joints: ";
    for (size_t i = 0; i < AHAND_JOINT_SUFFIXES.size(); ++i) {
      std::cout << AHAND_JOINT_SUFFIXES[i] << "=";
      if (std::isnan(state.ahand_values[i])) {
        std::cout << "?";
      } else {
        std::cout << std::setprecision(4) << state.ahand_values[i];
      }
      if (i < AHAND_JOINT_SUFFIXES.size() - 1) std::cout << ", ";
    }
    std::cout << "  # (Radians; j1x=yaw, j2x=flex, finger 1-4/4=thumb)\n";
  } else if (state.is_gripper) {
    std::cout << "  " << label << "Grasp: " << std::setprecision(4) << state.gripper_value << "\n";
  } else {
    std::cout << "  " << label << "Grasp: # Not found in /joint_states (no gripper or amazing_hand joints seen)\n";
  }
}

}  // namespace

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
  EndEffectorState left_hand;
  EndEffectorState right_hand;

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

      // End effector (gripper or amazing_hand, auto-detected)
      left_hand = detectEndEffector("openarm_left_", names, positions);
      right_hand = detectEndEffector("openarm_right_", names, positions);
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
  printHandState("lh", left_hand);
  std::cout << "\n";

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
  printHandState("rh", right_hand);
  std::cout << "\n";

  std::cout << "==================================================\n" << std::endl;

  rclcpp::shutdown();
  return 0;
}
