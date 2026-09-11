#pragma once

// Structural C++ port of
// openarm_description/scripts/hand_kinematics_node.py's math - see that
// file's module docstring for the mechanism explanation (closed-loop
// rod/gimbal linkage, two command spaces). Function/variable names
// deliberately mirror the python implementation so the two can be diffed
// side by side.
//
// Pure solver library - no rclcpp::Node, no pub/sub. Two consumers:
//   - control/robot_control's hand_kinematic node (topic-based, kept for
//     standalone dev/debug use)
//   - control/robot_hardware_interface's AmazingHandHW ros2_control plugin
//     (calls HandSolver::solve() directly from read(), no topics at all)

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace amazing_hand_kinematics
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

/// Generic URDF forward-kinematics helper (root-frame transforms).
class Kinematics
{
public:
  /// Parses joints/links out of a URDF string into the lookup tables used by
  /// fk()/childLinks()/ancestors().
  explicit Kinematics(const std::string& urdf_xml);

  /// Transform contributed by joint `jname` at `value`: its fixed origin
  /// composed with its motion (revolute/continuous rotate about `axis`,
  /// prismatic translates along it).
  Eigen::Matrix4d jointLocalT(const std::string& jname, double value) const;
  /// World-from-root transform of `link`, walking the parent chain (joints
  /// missing from `values` default to 0).
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
/// forward/inverse solve caches and warm starts, since each finger solves
/// independently.
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

  // solve_finger_servo/knuckle warm start + cache
  Eigen::VectorXd warm_start{Eigen::VectorXd::Zero(6)};
  bool has_last_solve{false};
  std::array<double, 2> last_solve_args{0.0, 0.0};
  std::map<std::string, double> last_solved;
  double last_cost{0.0};
  // Bumped every time solve_finger_servo/knuckle actually recomputes (cache
  // miss) - lets solveRodChains know whether it needs to re-run for this
  // exact solved-result without an object-identity check.
  int solve_generation{0};
  int rod_solved_generation{-1};

  // solve_rod_chains warm starts, one per rod_chains entry.
  std::vector<Eigen::Vector3d> rod_warm_start;
};

/// Finds each finger's gimbal/servo/rod-ball/knuckle joints and links by
/// structure (joint types + tree ancestry), scoped to one hand's links via
/// `link_prefix` when a combined robot_description holds both hands. Port of
/// hand_kinematics_node.py's discover_fingers().
std::vector<FingerInfo> discoverFingers(const Kinematics& hand, const std::string& link_prefix = "");
/// Computes rod lengths and knuckle offset from the assembled rest pose,
/// pairing each rod ball with its servo by rest-pose distance. Port of
/// hand_kinematics_node.py's calibrate().
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

/// Minimizes `residual_fn` starting from `x0` within [`lower`, `upper`] via
/// Levenberg-Marquardt with a numerically estimated Jacobian; see the
/// LeastSquaresResult doc above for why this mirrors scipy's least_squares.
LeastSquaresResult boundedLeastSquares(
  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& residual_fn,
  const Eigen::VectorXd& x0,
  const Eigen::VectorXd& lower,
  const Eigen::VectorXd& upper,
  double xtol = 1e-10, double ftol = 1e-10, int max_nfev = 200);

/// Owns the discovered/calibrated finger geometry and per-finger solve
/// state, and exposes the one method callers actually need: give it a flat
/// name->value map (alias names like "j23", real URDF joint names, or a mix
/// - matching sensor_msgs/JointState's name/position arrays), get back the
/// full solved joint state (same assembly onJointStates() does: alias->real
/// mapping, per-finger solve, rod-chain visual solve, all_movable_
/// zero-fallback for untouched passive joints).
class HandSolver
{
public:
  /// Discovers and calibrates the hand's fingers from `urdf_xml`, builds the
  /// alias<->real joint name maps, and prepares per-finger solve state ready
  /// for solve(). See hand_kinematics_node.py's HandKinematicsNode.__init__
  /// for the numbered-alias / thumb-reorder logic this mirrors.
  HandSolver(
    const std::string& urdf_xml, const std::string& link_prefix, const std::string& alias_prefix,
    const std::string& command_space);

  /// raw: joint name (alias or real) -> value. Returns: full joint name ->
  /// value for every joint this hand solves or reports (aliases excluded
  /// from the output, matching onJointStates()'s out_names/out_pos - use
  /// aliasState() for the alias-keyed view).
  std::map<std::string, double> solve(const std::map<std::string, double>& raw);

  /// Alias name -> last solved/clamped value (what republishStates()
  /// publishes today as the ros2_control-facing state feedback).
  const std::map<std::string, double>& aliasState() const { return alias_state_; }

  const std::map<std::string, std::string>& aliasOf() const { return alias_of_; }
  const std::vector<std::string>& allMovable() const { return all_movable_; }

  /// Gimbal link names of every finger whose most recent solve() call had
  /// residual cost > 1e-6 (scipy least_squares' .cost convention,
  /// 0.5*sum(residual**2)) - i.e. "command out of mechanism reach, showing
  /// nearest feasible pose". The library has no rclcpp dependency, so it
  /// doesn't log this itself - callers (HandKinematicNode, AmazingHandHW)
  /// do their own (possibly throttled) warning from this.
  const std::vector<std::string>& lastOutOfReachFingers() const { return last_out_of_reach_; }

private:
  /// Forward-solve residual: how far the 6 unknowns in `x` are from
  /// satisfying the two rod-length constraints and the distal_dup/distal_real
  /// match, given commanded servo angles `theta_a`/`theta_b`.
  Eigen::VectorXd residualServo(const FingerInfo& f, const Eigen::VectorXd& x, double theta_a, double theta_b) const;
  /// Inverse-solve residual: same constraints as residualServo(), but with
  /// `q1`/`q_prox` (knuckle command) fixed and the two servo angles unknown.
  Eigen::VectorXd residualKnuckle(const FingerInfo& f, const Eigen::VectorXd& x, double q1, double q_prox) const;

  /// Forward solve for one finger (servo angles -> passive joints), with
  /// warm-start/cache reuse when the commanded angles haven't changed.
  std::pair<LeastSquaresResult, std::map<std::string, double>> solveFingerServo(
    FingerInfo& f, const std::map<std::string, double>& raw);
  /// Inverse solve for one finger (knuckle command -> servo angles + passives),
  /// with the same warm-start/cache reuse as solveFingerServo().
  std::pair<LeastSquaresResult, std::map<std::string, double>> solveFingerKnuckle(
    FingerInfo& f, const std::map<std::string, double>& raw);
  /// Aims each rod's ball-joint chain at the opposite ball so the lever/rod
  /// meshes stay visually connected; skipped when `solved` is unchanged since
  /// the last call (see solve_generation/rod_solved_generation on FingerInfo).
  void solveRodChains(FingerInfo& f, const std::map<std::string, double>& raw, std::map<std::string, double>& solved);

  std::unique_ptr<Kinematics> hand_;
  std::vector<FingerInfo> fingers_;
  std::string command_space_{"knuckle"};

  std::map<std::string, std::string> alias_of_;  // alias -> real joint name
  std::map<std::string, double> alias_state_;     // alias -> last solved/clamped value
  std::map<std::string, double> alias_sign_;      // alias -> +1/-1

  std::vector<std::string> all_movable_;
  std::vector<std::string> last_out_of_reach_;
};

}  // namespace amazing_hand_kinematics
