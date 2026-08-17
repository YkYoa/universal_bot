// gravity_calibration_tool.cpp
//
// Interactive tool to calibrate the amazing_hand's lumped mass used by
// gravity_compensation_controller, against REAL measured joint torque
// (OpenArm_v10HW's tau_states_, sourced from the CAN motors' own reported
// torque - see hardware_interface.cpp's read()) instead of guessing.
//
// Run with the arm in `position` control_mode (bringup.launch.py's default -
// NOT torque/gravity-comp mode): a position controller actively holds each
// commanded pose against gravity, so the real torque it reports is a
// genuine measurement of what's needed to counteract gravity there, unlike
// torque mode with 0 command (nothing holding the arm, so nothing to
// measure) or gravity-comp-enabled mode (the very thing we're calibrating,
// which would bias the reading).
//
// Usage:
//   1. bringup.launch.py with control_mode: position (the default).
//   2. Move the arm to a pose (MoveIt Plan&Execute, or a direct
//      FollowJointTrajectory goal to left_arm_controller) and let it settle.
//   3. Press Enter here to capture (positions, measured effort) at that pose.
//   4. Repeat for 6+ diverse poses - vary shoulder/elbow AND wrist
//      (joint5/6/7) angles, since the amazing_hand connector's gravity
//      contribution depends on the whole chain's configuration.
//   5. Type 'done' + Enter to fit and print the calibrated hand mass, plus a
//      per-joint residual table.
//
// Method: for a lumped point mass at a fixed location, gravity torque is
// LINEAR in that mass (torque = mass * g * lever_arm_geometry, and the
// lever geometry is fixed by pose alone) - so instead of a black-box
// nonlinear fit, this computes gravity torque twice per captured pose (hand
// mass = 0 kg "baseline", hand mass = 1 kg "unit") using two chains cloned
// from the SAME live robot_description with only the connector segment's
// mass swapped. The unit-baseline difference is the per-joint sensitivity
// d(tau)/d(mass) for that pose. The correct mass is then a single closed-form
// weighted least-squares solve: minimize sum((measured - baseline - mass *
// sensitivity)^2) across every joint and every captured pose at once.
//
// The per-joint RESIDUAL after fitting (measured - predicted-with-fitted-
// mass) is reported too: a joint with a large, consistent residual the mass
// fit can't explain points at a DIFFERENT bug for that joint specifically
// (e.g. a URDF axis/sign mismatch), not a hand-mass calibration issue -
// don't chase mass value further if you see that.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <kdl/chain.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>

namespace
{

const std::vector<std::string> kArmJoints = {
  "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3", "openarm_left_joint4",
  "openarm_left_joint5", "openarm_left_joint6", "openarm_left_joint7",
  // Motor 8 (amazing_hand connector rotation) is a real revolute joint on
  // the base_link->tip_link chain now (see the ahand_connector comment in
  // openarm_robot.xacro) - must be listed here to match, same fix/reason as
  // left_gravity_comp_controller's `joints` param in bimanual_controllers
  // .yaml.
  "openarm_left_finger_joint1",
};
constexpr const char* kBaseLink = "openarm_body_link0";
constexpr const char* kTipLink = "openarm_left_hand_tcp";
// KDL::Segment::getName() returns the URDF LINK name (not the joint name -
// that's a separate accessor, seg.getJoint().getName()).
constexpr const char* kConnectorSegmentName = "openarm_left_ahand_connector";

std::string fetchRobotDescription(const rclcpp::Node::SharedPtr& node)
{
  std::promise<std::string> promise;
  auto future = promise.get_future();
  auto sub = node->create_subscription<std_msgs::msg::String>(
    "/robot_description", rclcpp::QoS(1).transient_local(),
    [&promise](const std_msgs::msg::String::SharedPtr msg) { promise.set_value(msg->data); });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
    if (std::chrono::steady_clock::now() > deadline) {
      throw std::runtime_error("Timed out waiting for /robot_description");
    }
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  executor.remove_node(node);
  return future.get();
}

// Clone `chain`, replacing the named segment's inertia mass (keeping its
// existing COG/rotational inertia shape) with `new_mass`.
KDL::Chain withSegmentMass(const KDL::Chain& chain, const std::string& segment_name, double new_mass)
{
  KDL::Chain out;
  for (unsigned int i = 0; i < chain.getNrOfSegments(); ++i) {
    const KDL::Segment& seg = chain.getSegment(i);
    if (seg.getName() == segment_name) {
      const KDL::RigidBodyInertia& inertia = seg.getInertia();
      KDL::RigidBodyInertia new_inertia(
        new_mass, inertia.getCOG(), KDL::RotationalInertia(
          inertia.getRotationalInertia().data[0], inertia.getRotationalInertia().data[4],
          inertia.getRotationalInertia().data[8], inertia.getRotationalInertia().data[1],
          inertia.getRotationalInertia().data[2], inertia.getRotationalInertia().data[5]));
      out.addSegment(KDL::Segment(seg.getName(), seg.getJoint(), seg.getFrameToTip(), new_inertia));
    } else {
      out.addSegment(seg);
    }
  }
  return out;
}

struct Sample
{
  std::vector<double> position;  // kArmJoints order
  std::vector<double> effort;    // kArmJoints order
};

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("gravity_calibration_tool");

  std::cout << "Fetching /robot_description..." << std::endl;
  std::string urdf_xml;
  try {
    urdf_xml = fetchRobotDescription(node);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }

  KDL::Tree tree;
  if (!kdl_parser::treeFromString(urdf_xml, tree)) {
    std::cerr << "ERROR: failed to parse robot_description into a KDL tree" << std::endl;
    return 1;
  }
  KDL::Chain base_chain;
  if (!tree.getChain(kBaseLink, kTipLink, base_chain)) {
    std::cerr << "ERROR: no KDL chain from " << kBaseLink << " to " << kTipLink << std::endl;
    return 1;
  }

  bool found_connector = false;
  for (unsigned int i = 0; i < base_chain.getNrOfSegments(); ++i) {
    if (base_chain.getSegment(i).getName() == kConnectorSegmentName) {
      found_connector = true;
      break;
    }
  }
  if (!found_connector) {
    std::cerr << "ERROR: segment '" << kConnectorSegmentName << "' not found in the chain - "
              << "did the connector joint/link name change? All segment names in the chain:" << std::endl;
    for (unsigned int i = 0; i < base_chain.getNrOfSegments(); ++i) {
      std::cerr << "  [" << i << "] " << base_chain.getSegment(i).getName()
                << "  (joint: " << base_chain.getSegment(i).getJoint().getName()
                << ", mass=" << base_chain.getSegment(i).getInertia().getMass() << ")" << std::endl;
    }
    return 1;
  }

  const KDL::Chain baseline_chain = withSegmentMass(base_chain, kConnectorSegmentName, 0.0);
  const KDL::Chain unit_chain = withSegmentMass(base_chain, kConnectorSegmentName, 1.0);
  KDL::ChainDynParam baseline_dyn(baseline_chain, KDL::Vector(0.0, 0.0, -9.81));
  KDL::ChainDynParam unit_dyn(unit_chain, KDL::Vector(0.0, 0.0, -9.81));
  const unsigned int n_joints = base_chain.getNrOfJoints();
  if (n_joints != kArmJoints.size()) {
    std::cerr << "ERROR: chain has " << n_joints << " joints, expected " << kArmJoints.size()
              << " - check kArmJoints order matches the chain" << std::endl;
    return 1;
  }

  // Latest /joint_states, updated in the background.
  auto latest_position = std::make_shared<std::vector<double>>(kArmJoints.size(), 0.0);
  auto latest_effort = std::make_shared<std::vector<double>>(kArmJoints.size(), 0.0);
  auto latest_stamp = std::make_shared<rclcpp::Time>(0, 0, RCL_ROS_TIME);
  auto js_sub = node->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", 50,
    [latest_position, latest_effort, latest_stamp, node](const sensor_msgs::msg::JointState::SharedPtr msg) {
      for (std::size_t i = 0; i < kArmJoints.size(); ++i) {
        for (std::size_t j = 0; j < msg->name.size(); ++j) {
          if (msg->name[j] == kArmJoints[i]) {
            if (j < msg->position.size()) (*latest_position)[i] = msg->position[j];
            if (j < msg->effort.size()) (*latest_effort)[i] = msg->effort[j];
            break;
          }
        }
      }
      *latest_stamp = node->now();
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() { executor.spin(); });

  std::cout << "\nReady. Move the arm (position mode) to a pose, let it settle, then press Enter\n"
            << "to capture. Type 'done' + Enter when you have 6+ diverse poses captured.\n"
            << std::endl;

  std::vector<Sample> samples;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "done") {
      break;
    }
    Sample s;
    s.position = *latest_position;
    s.effort = *latest_effort;
    samples.push_back(s);
    std::printf("Captured sample %zu: positions=[", samples.size());
    for (double p : s.position) std::printf("%.3f ", p);
    std::printf("] effort=[");
    for (double e : s.effort) std::printf("%.3f ", e);
    std::printf("]\n");
  }

  rclcpp::shutdown();
  spin_thread.join();

  if (samples.size() < 3) {
    std::cerr << "Need at least 3 samples (6+ recommended) to fit - got " << samples.size() << std::endl;
    return 1;
  }

  // Weighted least squares for a single scalar `mass`:
  //   measured_ij - baseline_ij = mass * sensitivity_ij + residual_ij
  // mass* = sum(diff * sensitivity) / sum(sensitivity^2), over every
  // (sample i, joint j) pair at once.
  std::vector<std::vector<double>> baseline_tau(samples.size(), std::vector<double>(n_joints));
  std::vector<std::vector<double>> sensitivity(samples.size(), std::vector<double>(n_joints));

  for (std::size_t i = 0; i < samples.size(); ++i) {
    KDL::JntArray q(n_joints);
    for (unsigned int j = 0; j < n_joints; ++j) q(j) = samples[i].position[j];

    KDL::JntArray tau_baseline(n_joints), tau_unit(n_joints);
    baseline_dyn.JntToGravity(q, tau_baseline);
    unit_dyn.JntToGravity(q, tau_unit);
    for (unsigned int j = 0; j < n_joints; ++j) {
      baseline_tau[i][j] = tau_baseline(j);
      sensitivity[i][j] = tau_unit(j) - tau_baseline(j);
    }
  }

  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < samples.size(); ++i) {
    for (unsigned int j = 0; j < n_joints; ++j) {
      const double diff = samples[i].effort[j] - baseline_tau[i][j];
      num += diff * sensitivity[i][j];
      den += sensitivity[i][j] * sensitivity[i][j];
    }
  }
  if (den < 1e-9) {
    std::cerr << "ERROR: sensitivity is ~0 across all samples/joints - captured poses don't vary "
              << "enough to identify the hand mass (try more diverse wrist angles)" << std::endl;
    return 1;
  }
  const double fitted_mass = num / den;

  std::printf("\n=== Fitted amazing_hand lumped mass: %.4f kg ===\n\n", fitted_mass);
  std::printf("Per-joint residuals (measured - predicted with fitted mass), by sample:\n");
  std::printf("%-24s", "joint");
  for (std::size_t i = 0; i < samples.size(); ++i) std::printf("  s%-6zu", i + 1);
  std::printf("   rms\n");
  for (unsigned int j = 0; j < n_joints; ++j) {
    std::printf("%-24s", kArmJoints[j].c_str());
    double sq_sum = 0.0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
      const double predicted = baseline_tau[i][j] + fitted_mass * sensitivity[i][j];
      const double residual = samples[i].effort[j] - predicted;
      sq_sum += residual * residual;
      std::printf("  %+6.3f", residual);
    }
    std::printf("  %6.3f\n", std::sqrt(sq_sum / samples.size()));
  }
  std::printf(
    "\nA joint with a large, consistently-signed residual across samples is NOT a hand-mass\n"
    "issue - that joint has a separate bug (e.g. a URDF axis/sign mismatch) worth auditing\n"
    "directly instead of adjusting mass further.\n");

  return 0;
}
