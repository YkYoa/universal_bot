#pragma once

// Structural C++ port of
// openarm_description/scripts/hand_kinematics_node.py - see that file's
// module docstring for the mechanism explanation (closed-loop rod/gimbal
// linkage, two command spaces). Function/variable names deliberately mirror
// the python implementation so the two can be diffed side by side.
//
// NOT launched by default anywhere yet (see plan Decision 6) - the python
// node remains the active implementation until this is validated against it
// by replaying the same joint_states_raw sequence and diffing published
// /joint_states (see plan Step 7).

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

namespace robot_control
{

struct JointInfo
{
  std::string name;
  std::string type;  // "revolute" | "continuous" | "prismatic" | "fixed" | "unknown"
  Eigen::Vector3d xyz{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d axis{Eigen::Vector3d::UnitZ()};
  std::string parent;
  std::string child;
  bool has_limit{false};
  double lower{0.0};
  double upper{0.0};
};

/// Generic URDF forward-kinematics helper (root-frame transforms) - port of
/// hand_kinematics_node.py's Kinematics class, built on top of the `urdf`
/// package's parser instead of hand-written XML parsing.
class Kinematics
{
public:
  explicit Kinematics(const std::string& urdf_xml);

  Eigen::Matrix4d jointLocalT(const std::string& jname, double value) const;
  Eigen::Matrix4d fk(const std::string& link, const std::map<std::string, double>& values = {}) const;
  /// jtype empty = any type, matching python's child_links(link, jtype=None).
  std::vector<std::pair<std::string, std::string>> childLinks(
    const std::string& link, const std::string& jtype = "") const;
  /// Ancestor LINK names (not joint names), immediate parent first.
  std::vector<std::string> ancestors(const std::string& link) const;

  std::map<std::string, JointInfo> joints;
  std::map<std::string, std::vector<std::string>> children;  // parent link -> joint names
  std::map<std::string, std::string> parent_of_link;          // child link -> joint name
  std::set<std::string> all_links;
};

struct ServoData
{
  std::string servo;
  std::string actuator_joint;
  std::string horn;
  std::string horn_ball;
  double lower{0.0};
  double upper{0.0};
};

struct RodChain
{
  std::array<std::string, 3> joints;
  std::string terminal;
  std::string target_ball;
  Eigen::Vector3d aim_local{Eigen::Vector3d::Zero()};
};

/// Port of hand_kinematics_node.py's per-finger dict. Also holds the
/// forward/inverse solve caches (python's f['_last_*'] entries) and warm
/// starts, since each finger solves independently.
struct FingerInfo
{
  std::string gimbal, q1_joint, bushing, host;
  std::string q2_joint, link_body;
  std::string prox_joint, proximal;
  double prox_lower{0.0}, prox_upper{0.0};
  double q1_lower{0.0}, q1_upper{0.0};
  std::string knuckle_joint, distal_dup;
  std::string prism_joint, twist_joint, distal_real;
  std::array<std::string, 2> rod_balls;
  std::array<ServoData, 2> servo_data;

  // filled by calibrate()
  std::array<double, 2> rod_lengths{0.0, 0.0};
  Eigen::Vector3d knuckle_offset_local{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d knuckle_rot_offset_local{Eigen::Matrix3d::Identity()};
  std::vector<RodChain> rod_chains;

  // solve_finger_servo/knuckle warm start + cache (python's x0 + f['_last_*'])
  Eigen::VectorXd warm_start{Eigen::VectorXd::Zero(6)};
  bool has_last_solve{false};
  std::array<double, 2> last_solve_args{0.0, 0.0};
  std::map<std::string, double> last_solved;
  double last_cost{0.0};
  // Bumped every time solve_finger_servo/knuckle actually recomputes (cache
  // miss) - replaces python's `f['_last_rod_solved_ref'] is solved` identity
  // check (same intent: "has solve_rod_chains already run for this exact
  // solved-result").
  int solve_generation{0};
  int rod_solved_generation{-1};

  // solve_rod_chains warm starts, one per rod_chains entry.
  std::vector<Eigen::Vector3d> rod_warm_start;
};

std::vector<FingerInfo> discoverFingers(const Kinematics& hand, const std::string& link_prefix = "");
std::vector<FingerInfo> calibrate(const Kinematics& hand, std::vector<FingerInfo> fingers);

/// Bounded (box-constrained) Levenberg-Marquardt with a numerically
/// estimated Jacobian - the C++ analog of scipy.optimize.least_squares'
/// default (numerical-Jacobian, bounds via 'trf') used throughout the
/// python node, none of whose residual functions supply an analytic
/// Jacobian either.
struct LeastSquaresResult
{
  Eigen::VectorXd x;
  double cost{0.0};
};

LeastSquaresResult boundedLeastSquares(
  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& residual_fn,
  const Eigen::VectorXd& x0,
  const Eigen::VectorXd& lower,
  const Eigen::VectorXd& upper,
  double xtol = 1e-10, double ftol = 1e-10, int max_nfev = 200);

class HandKinematicNode : public rclcpp::Node
{
public:
  explicit HandKinematicNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  using JointStateMsg = sensor_msgs::msg::JointState;

  Eigen::VectorXd residualServo(const FingerInfo& f, const Eigen::VectorXd& x, double theta_a, double theta_b) const;
  Eigen::VectorXd residualKnuckle(const FingerInfo& f, const Eigen::VectorXd& x, double q1, double q_prox) const;

  std::pair<LeastSquaresResult, std::map<std::string, double>> solveFingerServo(
    FingerInfo& f, const std::map<std::string, double>& raw);
  std::pair<LeastSquaresResult, std::map<std::string, double>> solveFingerKnuckle(
    FingerInfo& f, const std::map<std::string, double>& raw);
  void solveRodChains(
    FingerInfo& f, const std::map<std::string, double>& raw, std::map<std::string, double>& solved);

  void onJointStates(const JointStateMsg& msg);
  void republishStates();

  std::unique_ptr<Kinematics> hand_;
  std::vector<FingerInfo> fingers_;
  std::string command_space_{"knuckle"};

  std::map<std::string, std::string> alias_of_;    // alias -> real joint name
  std::map<std::string, double> alias_state_;      // alias -> last solved/clamped value
  std::map<std::string, double> alias_sign_;        // alias -> +1/-1

  std::vector<std::string> all_movable_;

  std::optional<std::pair<std::vector<std::string>, std::vector<double>>> last_full_state_;

  rclcpp::Publisher<JointStateMsg>::SharedPtr pub_;
  rclcpp::Publisher<JointStateMsg>::SharedPtr alias_pub_;
  rclcpp::Subscription<JointStateMsg>::SharedPtr sub_;
  rclcpp::Subscription<JointStateMsg>::SharedPtr cmd_sub_;
  rclcpp::TimerBase::SharedPtr state_timer_;
};

}  // namespace robot_control
