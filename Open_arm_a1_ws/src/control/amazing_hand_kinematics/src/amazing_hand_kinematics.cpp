#include "amazing_hand_kinematics/amazing_hand_kinematics.hpp"

#include <algorithm>
#include <cmath>
#include <regex>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <urdf/model.h>

namespace amazing_hand_kinematics
{

// ---------------------------------------------------------------------------
// Kinematics
// ---------------------------------------------------------------------------

Kinematics::Kinematics(const std::string& urdf_xml)
{
  urdf::Model model;
  if (!model.initString(urdf_xml)) {
    throw std::runtime_error("amazing_hand_kinematics: failed to parse robot_description URDF");
  }

  for (const auto& [name, link] : model.links_) {
    all_links.insert(name);
  }

  for (const auto& [name, joint] : model.joints_) {
    JointInfo info;
    info.name = name;
    switch (joint->type) {
      case urdf::Joint::REVOLUTE:
        info.type = "revolute";
        break;
      case urdf::Joint::CONTINUOUS:
        info.type = "continuous";
        break;
      case urdf::Joint::PRISMATIC:
        info.type = "prismatic";
        break;
      case urdf::Joint::FIXED:
        info.type = "fixed";
        break;
      default:
        info.type = "unknown";
        break;
    }

    const auto& origin = joint->parent_to_joint_origin_transform;
    info.xyz = Eigen::Vector3d(origin.position.x, origin.position.y, origin.position.z);
    double qx, qy, qz, qw;
    origin.rotation.getQuaternion(qx, qy, qz, qw);
    info.R = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
    info.axis = Eigen::Vector3d(joint->axis.x, joint->axis.y, joint->axis.z);
    info.parent = joint->parent_link_name;
    info.child = joint->child_link_name;
    if (joint->limits) {
      info.has_limit = true;
      info.lower = joint->limits->lower;
      info.upper = joint->limits->upper;
    }

    joints[name] = info;
    children[info.parent].push_back(name);
    parent_of_link[info.child] = name;
  }
}

Eigen::Matrix4d Kinematics::jointLocalT(const std::string& jname, double value) const
{
  const JointInfo& j = joints.at(jname);
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = j.R;
  T.block<3, 1>(0, 3) = j.xyz;

  if (j.type == "revolute" || j.type == "continuous") {
    Eigen::Matrix4d Tm = Eigen::Matrix4d::Identity();
    Tm.block<3, 3>(0, 0) = Eigen::AngleAxisd(value, j.axis.normalized()).toRotationMatrix();
    return T * Tm;
  }
  if (j.type == "prismatic") {
    Eigen::Matrix4d Tm = Eigen::Matrix4d::Identity();
    Tm.block<3, 1>(0, 3) = j.axis.normalized() * value;
    return T * Tm;
  }
  return T;
}

Eigen::Matrix4d Kinematics::fk(const std::string& link, const std::map<std::string, double>& values) const
{
  std::vector<std::string> chain;
  std::string cur = link;
  while (parent_of_link.count(cur)) {
    const std::string& jn = parent_of_link.at(cur);
    chain.push_back(jn);
    cur = joints.at(jn).parent;
  }

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    double v = 0.0;
    auto vi = values.find(*it);
    if (vi != values.end()) {
      v = vi->second;
    }
    T = T * jointLocalT(*it, v);
  }
  return T;
}

std::vector<std::pair<std::string, std::string>> Kinematics::childLinks(
  const std::string& link, const std::string& jtype) const
{
  std::vector<std::pair<std::string, std::string>> out;
  auto it = children.find(link);
  if (it == children.end()) {
    return out;
  }
  for (const auto& jn : it->second) {
    const JointInfo& j = joints.at(jn);
    if (jtype.empty() || j.type == jtype) {
      out.emplace_back(jn, j.child);
    }
  }
  return out;
}

std::vector<std::string> Kinematics::ancestors(const std::string& link) const
{
  std::vector<std::string> path;
  std::string cur = link;
  while (parent_of_link.count(cur)) {
    const std::string& jn = parent_of_link.at(cur);
    cur = joints.at(jn).parent;
    path.push_back(cur);
  }
  return path;
}

namespace
{

std::pair<std::vector<std::string>, std::string> ballChain(const Kinematics& hand, const std::string& start_link)
{
  std::vector<std::string> joints;
  std::string cur = start_link;
  // No artificial cap in the python original; capped here at a generous 20 as
  // a defensive guard against a malformed URDF hanging the node, which would
  // otherwise be worse in a long-running C++ node than in the python
  // prototype it never actually hit.
  for (int guard = 0; guard < 20; ++guard) {
    auto kids = hand.childLinks(cur, "continuous");
    if (kids.empty()) {
      return {joints, cur};
    }
    joints.push_back(kids[0].first);
    cur = kids[0].second;
  }
  throw std::runtime_error("ballChain: exceeded 20 continuous joints from " + start_link + " (malformed URDF?)");
}

std::string regexEscape(const std::string& s)
{
  static const std::regex special(R"([.^$|()\[\]{}*+?\\])");
  return std::regex_replace(s, special, R"(\$&)");
}

}  // namespace

// ---------------------------------------------------------------------------
// discoverFingers / calibrate
// ---------------------------------------------------------------------------

std::vector<FingerInfo> discoverFingers(const Kinematics& hand, const std::string& link_prefix)
{
  const std::string p = regexEscape(link_prefix);
  const std::regex servo_re("^" + p + "scs0009(_[0-9]+)?$");
  const std::regex bushing_re("^" + p + "bushing_0608_04(_[0-9]+)?$");

  std::vector<std::string> servo_links, bushing_links;
  for (const auto& l : hand.all_links) {
    if (std::regex_match(l, servo_re)) servo_links.push_back(l);
    if (std::regex_match(l, bushing_re)) bushing_links.push_back(l);
  }
  std::sort(servo_links.begin(), servo_links.end());
  std::sort(bushing_links.begin(), bushing_links.end());
  const std::set<std::string> bushing_set(bushing_links.begin(), bushing_links.end());

  std::vector<std::tuple<std::string, std::string, std::string>> gimbals;
  for (const auto& [jn, j] : hand.joints) {
    if (j.type == "revolute" && bushing_set.count(j.parent)) {
      gimbals.emplace_back(jn, j.parent, j.child);
    }
  }

  std::vector<FingerInfo> fingers;

  for (const auto& [q1_joint, bushing, gimbal] : gimbals) {
    std::string host;
    std::vector<std::string> servos;
    for (const auto& cand : hand.ancestors(bushing)) {
      std::vector<std::string> servos_here;
      for (const auto& s : servo_links) {
        auto anc = hand.ancestors(s);
        if (std::find(anc.begin(), anc.end(), cand) != anc.end()) {
          servos_here.push_back(s);
        }
      }
      if (servos_here.size() == 2) {
        host = cand;
        servos = servos_here;
        break;
      }
    }
    if (host.empty()) {
      throw std::runtime_error("could not find a 2-servo host ancestor for gimbal " + gimbal);
    }

    std::string proximal_j, proximal_link, link_body_j, link_body;
    for (const auto& [kjn, kc] : hand.childLinks(gimbal)) {
      const JointInfo& kj = hand.joints.at(kjn);
      if (kj.type == "fixed") {
        continue;
      }
      if (kj.type == "revolute" && kj.has_limit) {
        proximal_j = kjn;
        proximal_link = kc;
      } else if (kj.type == "continuous") {
        link_body_j = kjn;
        link_body = kc;
      }
    }
    if (proximal_link.empty() || link_body.empty()) {
      throw std::runtime_error("gimbal " + gimbal + ": expected a proximal + link_body child pair");
    }

    std::vector<std::string> rb_children;
    for (const auto& [jn2, c2] : hand.childLinks(link_body, "fixed")) {
      if (c2.find("rotule_ball") != std::string::npos) {
        rb_children.push_back(c2);
      }
    }
    if (rb_children.size() != 2) {
      throw std::runtime_error("link_body " + link_body + ": expected 2 rotule_ball children");
    }

    auto prisms = hand.childLinks(link_body, "prismatic");
    if (prisms.size() != 1) {
      throw std::runtime_error("link_body " + link_body + ": expected 1 prismatic child");
    }
    const std::string prism_j = prisms[0].first;
    const std::string cyl = prisms[0].second;

    auto conts = hand.childLinks(cyl, "continuous");
    if (conts.size() != 1) {
      throw std::runtime_error(cyl + ": expected 1 continuous child (distal twist)");
    }
    const std::string twist_j = conts[0].first;
    const std::string distal_real = conts[0].second;

    auto pkids = hand.childLinks(proximal_link, "continuous");
    if (pkids.size() != 1) {
      throw std::runtime_error("proximal " + proximal_link + ": expected 1 continuous knuckle child");
    }
    const std::string knuckle_j = pkids[0].first;
    const std::string distal_dup = pkids[0].second;

    std::array<ServoData, 2> servo_data;
    for (int i = 0; i < 2; ++i) {
      const std::string& sv = servos[i];
      std::vector<std::pair<std::string, JointInfo>> acts;
      for (const auto& [jn3, j3] : hand.joints) {
        if (j3.parent == sv && j3.type == "revolute" && j3.has_limit) {
          acts.emplace_back(jn3, j3);
        }
      }
      if (acts.size() != 1) {
        throw std::runtime_error("servo " + sv + ": expected 1 actuator joint");
      }
      const std::string& act_jn = acts[0].first;
      const JointInfo& act_j = acts[0].second;
      const std::string& horn = act_j.child;

      std::vector<std::string> hb;
      for (const auto& [jn4, c4] : hand.childLinks(horn, "fixed")) {
        if (c4.find("rotule_ball") != std::string::npos) {
          hb.push_back(c4);
        }
      }
      if (hb.size() != 1) {
        throw std::runtime_error("horn " + horn + ": expected 1 rotule_ball child");
      }
      servo_data[i] = ServoData{sv, act_jn, horn, hb[0], act_j.lower, act_j.upper};
    }

    FingerInfo finger;
    finger.gimbal = gimbal;
    finger.q1_joint = q1_joint;
    finger.bushing = bushing;
    finger.host = host;
    finger.q2_joint = link_body_j;
    finger.link_body = link_body;
    finger.prox_joint = proximal_j;
    finger.proximal = proximal_link;
    finger.prox_lower = hand.joints.at(proximal_j).lower;
    finger.prox_upper = hand.joints.at(proximal_j).upper;
    finger.q1_lower = hand.joints.at(q1_joint).lower;
    finger.q1_upper = hand.joints.at(q1_joint).upper;
    finger.knuckle_joint = knuckle_j;
    finger.distal_dup = distal_dup;
    finger.prism_joint = prism_j;
    finger.twist_joint = twist_j;
    finger.distal_real = distal_real;
    finger.rod_balls = {rb_children[0], rb_children[1]};
    finger.servo_data = servo_data;
    fingers.push_back(std::move(finger));
  }

  return fingers;
}

std::vector<FingerInfo> calibrate(const Kinematics& hand, std::vector<FingerInfo> fingers)
{
  std::map<std::string, Eigen::Vector3d> rest_pos;
  for (const auto& link : hand.all_links) {
    rest_pos[link] = hand.fk(link).block<3, 1>(0, 3);
  }

  for (auto& f : fingers) {
    double d[2][2];
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        d[i][j] = (rest_pos.at(f.rod_balls[i]) - rest_pos.at(f.servo_data[j].horn_ball)).norm();
      }
    }
    if (d[0][0] + d[1][1] > d[0][1] + d[1][0]) {
      std::swap(f.servo_data[0], f.servo_data[1]);
      double d00 = d[0][1], d01 = d[0][0], d10 = d[1][1], d11 = d[1][0];
      d[0][0] = d00;
      d[0][1] = d01;
      d[1][0] = d10;
      d[1][1] = d11;
    }
    f.rod_lengths = {d[0][0], d[1][1]};

    // distal_dup and distal_real are the same physical piece exported as two
    // links with different frames - offset is rigid in the dup frame, not
    // constant in the world, so store it locally and rotate at solve time.
    // Also matching the rest-pose ROTATION offset (not just position) is
    // required, or the proximal/distal sides visibly twist apart under flex
    // (see hand_kinematics_node.py's calibrate() comment for the history).
    Eigen::Matrix4d T_dd = hand.fk(f.distal_dup);
    f.knuckle_offset_local =
      T_dd.block<3, 3>(0, 0).transpose() * (rest_pos.at(f.distal_real) - T_dd.block<3, 1>(0, 3));
    Eigen::Matrix4d T_dr = hand.fk(f.distal_real);
    f.knuckle_rot_offset_local = T_dd.block<3, 3>(0, 0).transpose() * T_dr.block<3, 3>(0, 0);

    f.rod_chains.clear();
    for (int i = 0; i < 2; ++i) {
      const std::string& hb = f.servo_data[i].horn_ball;
      const std::string& rb = f.rod_balls[i];
      const std::vector<std::pair<std::string, std::string>> ends = {{hb, rb}, {rb, hb}};
      for (const auto& [start, target] : ends) {
        auto [chain_joints, terminal] = ballChain(hand, start);
        if (chain_joints.size() != 3) {
          continue;
        }
        Eigen::Matrix4d T_t = hand.fk(terminal);
        Eigen::Vector3d aim_local = T_t.block<3, 3>(0, 0).transpose() * (rest_pos.at(target) - T_t.block<3, 1>(0, 3));
        RodChain rc;
        rc.joints = {chain_joints[0], chain_joints[1], chain_joints[2]};
        rc.terminal = terminal;
        rc.target_ball = target;
        rc.aim_local = aim_local;
        f.rod_chains.push_back(rc);
      }
    }
  }

  return fingers;
}

// ---------------------------------------------------------------------------
// boundedLeastSquares - box-constrained Levenberg-Marquardt, numeric Jacobian
// ---------------------------------------------------------------------------

namespace
{

Eigen::VectorXd clampToBounds(const Eigen::VectorXd& x, const Eigen::VectorXd& lower, const Eigen::VectorXd& upper)
{
  Eigen::VectorXd out = x;
  for (int i = 0; i < x.size(); ++i) {
    out[i] = std::min(std::max(x[i], lower[i]), upper[i]);
  }
  return out;
}

Eigen::MatrixXd numericJacobian(
  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& f, const Eigen::VectorXd& x,
  const Eigen::VectorXd& f0)
{
  const int n = static_cast<int>(x.size());
  const int m = static_cast<int>(f0.size());
  Eigen::MatrixXd J(m, n);
  for (int j = 0; j < n; ++j) {
    Eigen::VectorXd xp = x;
    const double h = 1e-7 * std::max(1.0, std::abs(x[j]));
    xp[j] += h;
    Eigen::VectorXd f1 = f(xp);
    J.col(j) = (f1 - f0) / h;
  }
  return J;
}

}  // namespace

namespace
{

// Runs boundedLeastSquares' actual LM iteration in a pre-scaled parameter
// space (see boundedLeastSquares below for why).
LeastSquaresResult boundedLeastSquaresScaled(
  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& residual_fn, const Eigen::VectorXd& x0,
  const Eigen::VectorXd& lower, const Eigen::VectorXd& upper, double xtol, double ftol, int max_nfev)
{
  Eigen::VectorXd x = clampToBounds(x0, lower, upper);
  Eigen::VectorXd r = residual_fn(x);
  double cost = 0.5 * r.squaredNorm();
  int nfev = 1;

  double lambda = 1e-3;
  constexpr double kLambdaUp = 10.0;
  constexpr double kLambdaDown = 0.1;

  while (nfev < max_nfev) {
    Eigen::MatrixXd J = numericJacobian(residual_fn, x, r);
    nfev += static_cast<int>(x.size());

    Eigen::MatrixXd JtJ = J.transpose() * J;
    Eigen::VectorXd Jtr = J.transpose() * r;

    bool improved = false;
    for (int attempt = 0; attempt < 10 && nfev < max_nfev; ++attempt) {
      Eigen::MatrixXd A = JtJ;
      for (int i = 0; i < A.rows(); ++i) {
        A(i, i) += lambda * std::max(JtJ(i, i), 1e-12);
      }
      Eigen::VectorXd step = A.ldlt().solve(-Jtr);

      Eigen::VectorXd x_new = clampToBounds(x + step, lower, upper);
      Eigen::VectorXd r_new = residual_fn(x_new);
      ++nfev;
      const double cost_new = 0.5 * r_new.squaredNorm();

      if (cost_new < cost) {
        const double step_norm = (x_new - x).norm();
        const double cost_decrease = cost - cost_new;
        x = x_new;
        r = r_new;
        cost = cost_new;
        lambda = std::max(lambda * kLambdaDown, 1e-12);
        improved = true;
        if (step_norm < xtol || cost_decrease < ftol * std::max(1.0, cost)) {
          return {x, cost};
        }
        break;
      }
      lambda *= kLambdaUp;
    }
    if (!improved) {
      break;
    }
  }
  return {x, cost};
}

}  // namespace

LeastSquaresResult boundedLeastSquares(
  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& residual_fn, const Eigen::VectorXd& x0,
  const Eigen::VectorXd& lower, const Eigen::VectorXd& upper, double xtol, double ftol, int max_nfev)
{
  // This finger solve's unknowns span wildly different natural scales in
  // the same problem (knuckle/servo angles ~O(1) rad vs. e.g. a rod-chain
  // prism DOF bounded to +-0.02) - the LM step above already scales its
  // damping by each variable's own JtJ diagonal (Marquardt scaling), but
  // bound-clamping and step-length convergence (xtol) are NOT scale-aware,
  // so a naive clamp-to-bounds step can behave poorly (get stuck well short
  // of a feasible solution scipy's bounds-aware TRF would find) exactly
  // when a small-range variable needs to move a large fraction of its own
  // range while a large-range variable barely needs to move at all.
  // Fix: normalize every variable by its own bound half-width before
  // running the iteration (so every dimension's range is ~[-1,1]), then map
  // the result back. Residual/cost semantics are unchanged - only the
  // search-space geometry (steps, clamping, xtol) is what benefits.
  const int n = static_cast<int>(x0.size());
  Eigen::VectorXd scale(n);
  for (int i = 0; i < n; ++i) {
    const double width = upper[i] - lower[i];
    scale[i] = (width > 1e-9) ? (0.5 * width) : 1.0;
  }

  auto scaled_residual_fn = [&](const Eigen::VectorXd& xs) { return residual_fn(xs.cwiseProduct(scale)); };
  const Eigen::VectorXd lower_s = lower.cwiseQuotient(scale);
  const Eigen::VectorXd upper_s = upper.cwiseQuotient(scale);
  const Eigen::VectorXd x0_s = x0.cwiseQuotient(scale);

  const auto sol_s = boundedLeastSquaresScaled(scaled_residual_fn, x0_s, lower_s, upper_s, xtol, ftol, max_nfev);
  return {sol_s.x.cwiseProduct(scale), sol_s.cost};
}

namespace
{

bool anyNonzero(const Eigen::VectorXd& v)
{
  return (v.array() != 0.0).any();
}

// boundedLeastSquares chases whichever basin its damped-Newton steps happen
// to fall into from the given start - for a genuinely multi-modal residual
// (this finger's rod/gimbal loop-closure has more than one geometrically
// valid configuration, e.g. the twist DOF can wind either "short way" or
// "long way" around), a single start from zero can land in a much worse
// basin than scipy's TRF finds from the identical start. Chase the current
// best with a few damping-reset restarts, then re-seed the search from a
// small grid of starting points spread across the last variable's own
// range (empirically the one most prone to a bad basin - it's the
// least-constrained/most range-tolerant DOF in both command spaces) to give
// the search a chance to find a fundamentally different basin instead of
// just refining the one it's already stuck in.
LeastSquaresResult solveMultistart(
  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& residual_fn, const Eigen::VectorXd& x0,
  const Eigen::VectorXd& lower, const Eigen::VectorXd& upper)
{
  auto refine = [&](const Eigen::VectorXd& start) {
    auto s = boundedLeastSquares(residual_fn, start, lower, upper, 1e-10, 1e-10, 200);
    for (int i = 0; i < 4 && s.cost > 1e-9; ++i) {
      auto s_next = boundedLeastSquares(residual_fn, s.x, lower, upper, 1e-10, 1e-10, 200);
      if (s_next.cost >= s.cost - 1e-12) {
        break;
      }
      s = s_next;
    }
    return s;
  };

  auto sol = refine(x0);
  if (sol.cost > 1e-9 && anyNonzero(x0)) {
    auto sol_rest = refine(Eigen::VectorXd::Zero(x0.size()));
    if (sol_rest.cost < sol.cost) {
      sol = sol_rest;
    }
  }

  if (sol.cost > 1e-9) {
    const int n = static_cast<int>(x0.size());
    const int last = n - 1;
    for (double frac : {-0.75, -0.35, 0.35, 0.75}) {
      if (sol.cost <= 1e-9) {
        break;
      }
      Eigen::VectorXd start = Eigen::VectorXd::Zero(n);
      start[last] = lower[last] + (0.5 + 0.5 * frac) * (upper[last] - lower[last]);
      auto sol_grid = refine(start);
      if (sol_grid.cost < sol.cost) {
        sol = sol_grid;
      }
    }
  }

  return sol;
}

bool argsUnchanged(const std::array<double, 2>& prev, const std::array<double, 2>& cur, double tol = 1e-6)
{
  return std::abs(prev[0] - cur[0]) < tol && std::abs(prev[1] - cur[1]) < tol;
}

Eigen::Vector3d orientationResidual(const Eigen::Matrix3d& Ra, const Eigen::Matrix3d& Rb)
{
  Eigen::Matrix3d Rdiff = Ra.transpose() * Rb;
  return 0.5 * Eigen::Vector3d(Rdiff(2, 1) - Rdiff(1, 2), Rdiff(0, 2) - Rdiff(2, 0), Rdiff(1, 0) - Rdiff(0, 1));
}

Eigen::VectorXd fingerLoopResidual(
  const Kinematics& hand, const FingerInfo& f, double q1, double q2, double q_prox, double q_knu, double q_prism,
  double q_twist, double theta_a, double theta_b)
{
  Eigen::Vector3d p_rb0 = hand.fk(f.rod_balls[0], {{f.q1_joint, q1}, {f.q2_joint, q2}}).block<3, 1>(0, 3);
  Eigen::Vector3d p_rb1 = hand.fk(f.rod_balls[1], {{f.q1_joint, q1}, {f.q2_joint, q2}}).block<3, 1>(0, 3);
  Eigen::Vector3d p_hb0 =
    hand.fk(f.servo_data[0].horn_ball, {{f.servo_data[0].actuator_joint, theta_a}}).block<3, 1>(0, 3);
  Eigen::Vector3d p_hb1 =
    hand.fk(f.servo_data[1].horn_ball, {{f.servo_data[1].actuator_joint, theta_b}}).block<3, 1>(0, 3);
  Eigen::Matrix4d T_dr =
    hand.fk(f.distal_real, {{f.q1_joint, q1}, {f.q2_joint, q2}, {f.prism_joint, q_prism}, {f.twist_joint, q_twist}});
  Eigen::Matrix4d T_dd = hand.fk(f.distal_dup, {{f.q1_joint, q1}, {f.prox_joint, q_prox}, {f.knuckle_joint, q_knu}});
  Eigen::Vector3d p_target = T_dd.block<3, 3>(0, 0) * f.knuckle_offset_local + T_dd.block<3, 1>(0, 3);
  Eigen::Matrix3d R_target = T_dd.block<3, 3>(0, 0) * f.knuckle_rot_offset_local;

  Eigen::VectorXd res(8);
  res[0] = (p_rb0 - p_hb0).norm() - f.rod_lengths[0];
  res[1] = (p_rb1 - p_hb1).norm() - f.rod_lengths[1];
  res.segment<3>(2) = T_dr.block<3, 1>(0, 3) - p_target;
  res.segment<3>(5) = orientationResidual(T_dr.block<3, 3>(0, 0), R_target);
  return res;
}

}  // namespace

// ---------------------------------------------------------------------------
// HandSolver
// ---------------------------------------------------------------------------

Eigen::VectorXd HandSolver::residualServo(
  const FingerInfo& f, const Eigen::VectorXd& x, double theta_a, double theta_b) const
{
  return fingerLoopResidual(*hand_, f, x[0], x[1], x[2], x[3], x[4], x[5], theta_a, theta_b);
}

Eigen::VectorXd HandSolver::residualKnuckle(const FingerInfo& f, const Eigen::VectorXd& x, double q1, double q_prox) const
{
  // x = (theta_A, theta_B, q2, q_knu, q_prism, q_twist)
  return fingerLoopResidual(*hand_, f, q1, x[2], q_prox, x[3], x[4], x[5], x[0], x[1]);
}

HandSolver::HandSolver(
  const std::string& urdf_xml, const std::string& link_prefix, const std::string& alias_prefix,
  const std::string& command_space)
: command_space_(command_space)
{
  if (urdf_xml.empty()) {
    throw std::runtime_error("amazing_hand_kinematics: robot_description is empty; can't build kinematics");
  }
  if (command_space_ != "servo" && command_space_ != "knuckle") {
    throw std::runtime_error("amazing_hand_kinematics: command_space must be 'servo' or 'knuckle', got '" + command_space_ + "'");
  }

  hand_ = std::make_unique<Kinematics>(urdf_xml);
  fingers_ = calibrate(*hand_, discoverFingers(*hand_, link_prefix));

  // Deterministic finger numbering, identical for both hands: sort by rest
  // height of the gimbal in the hand's OWN frame (see hand_kinematics_node.py).
  const std::string root_link = link_prefix + "root";
  Eigen::Matrix4d T_root_inv = hand_->fk(root_link).inverse();
  std::stable_sort(fingers_.begin(), fingers_.end(), [&](const FingerInfo& a, const FingerInfo& b) {
    double za = (T_root_inv * hand_->fk(a.gimbal))(2, 3);
    double zb = (T_root_inv * hand_->fk(b.gimbal))(2, 3);
    return za > zb;
  });
  // Height order alone doesn't match anatomy for this hand (thumb sits
  // third-highest) - reorder so alias index 4 = thumb, empirically confirmed
  // (see hand_kinematics_node.py).
  if (fingers_.size() == 4) {
    fingers_ = {fingers_[0], fingers_[1], fingers_[3], fingers_[2]};
  }

  for (auto& f : fingers_) {
    f.warm_start = Eigen::VectorXd::Zero(6);
    f.rod_warm_start.assign(f.rod_chains.size(), Eigen::Vector3d::Zero());
  }

  for (std::size_t i = 0; i < fingers_.size(); ++i) {
    const FingerInfo& f = fingers_[i];
    const std::string idx = std::to_string(i + 1);
    const std::string alias_j1 = alias_prefix + "j1" + idx;
    const std::string alias_j2 = alias_prefix + "j2" + idx;
    alias_of_[alias_j1] = f.q1_joint;
    alias_of_[alias_j2] = f.prox_joint;
    alias_state_[alias_j1] = 0.0;
    alias_state_[alias_j2] = 0.0;
    alias_sign_[alias_j1] = 1.0;
    // l_ahand/r_ahand mirror the prox_joint rotation axis between hands (+X
    // vs -X) but keep the same sign convention ("positive = close") expected
    // from the alias regardless of side - canonicalize here instead of in
    // the URDF (see hand_kinematics_node.py).
    alias_sign_[alias_j2] = hand_->joints.at(f.prox_joint).axis.x() > 0.0 ? -1.0 : 1.0;
  }

  // Every movable joint of THIS hand must appear in the published joint
  // state - robot_state_publisher only emits TF for joints it has seen.
  // Scoped to link_prefix so, when both hands share one robot_description,
  // each solver instance only reports its own side.
  for (const auto& [jn, j] : hand_->joints) {
    if (j.type != "fixed" && !alias_of_.count(jn) && j.parent.rfind(link_prefix, 0) == 0) {
      all_movable_.push_back(jn);
    }
  }
}

std::pair<LeastSquaresResult, std::map<std::string, double>> HandSolver::solveFingerServo(
  FingerInfo& f, const std::map<std::string, double>& raw)
{
  auto get_or_0 = [&](const std::string& k) {
    auto it = raw.find(k);
    return it != raw.end() ? it->second : 0.0;
  };
  const double theta_a = get_or_0(f.servo_data[0].actuator_joint);
  const double theta_b = get_or_0(f.servo_data[1].actuator_joint);
  const std::array<double, 2> args{theta_a, theta_b};

  if (f.has_last_solve && argsUnchanged(f.last_solve_args, args)) {
    return {LeastSquaresResult{f.warm_start, f.last_cost}, f.last_solved};
  }

  // q2/q_knu bounds (+-1.6, not a real URDF limit): a generic safety guard
  // on this optimizer's intermediate DOF, not a physical joint limit. Was
  // +-1.2, which silently capped max achievable flex around 1.15 rad even
  // after the actual servo range was corrected - confirmed against real
  // hardware reaching ~1.31 rad, needing q2 up to ~1.42 there.
  Eigen::VectorXd lower(6), upper(6);
  lower << f.q1_lower, -1.6, f.prox_lower, -1.6, -0.02, -3.2;
  upper << f.q1_upper, 1.6, f.prox_upper, 1.6, 0.02, 3.2;

  auto residual_fn = [this, &f, theta_a, theta_b](const Eigen::VectorXd& x) {
    return residualServo(f, x, theta_a, theta_b);
  };
  auto sol = solveMultistart(residual_fn, f.warm_start, lower, upper);
  f.warm_start = sol.x;

  const double q1 = sol.x[0], q2 = sol.x[1], q_prox = sol.x[2];
  const double q_knu = sol.x[3], q_prism = sol.x[4], q_twist = sol.x[5];
  std::map<std::string, double> solved{
    {f.q1_joint, q1},         {f.q2_joint, q2},         {f.prox_joint, q_prox},
    {f.knuckle_joint, q_knu}, {f.prism_joint, q_prism}, {f.twist_joint, q_twist},
  };

  f.last_solve_args = args;
  f.last_solved = solved;
  f.last_cost = sol.cost;
  f.has_last_solve = true;
  ++f.solve_generation;

  return {sol, solved};
}

std::pair<LeastSquaresResult, std::map<std::string, double>> HandSolver::solveFingerKnuckle(
  FingerInfo& f, const std::map<std::string, double>& raw)
{
  auto get_or_0 = [&](const std::string& k) {
    auto it = raw.find(k);
    return it != raw.end() ? it->second : 0.0;
  };
  const double q1 = std::clamp(get_or_0(f.q1_joint), f.q1_lower, f.q1_upper);
  const double q_prox = std::clamp(get_or_0(f.prox_joint), f.prox_lower, f.prox_upper);
  const std::array<double, 2> args{q1, q_prox};

  if (f.has_last_solve && argsUnchanged(f.last_solve_args, args)) {
    return {LeastSquaresResult{f.warm_start, f.last_cost}, f.last_solved};
  }

  // q2/q_knu bounds (+-1.6, not a real URDF limit) - see solveFingerServo.
  Eigen::VectorXd lower(6), upper(6);
  lower << f.servo_data[0].lower, f.servo_data[1].lower, -1.6, -1.6, -0.02, -3.2;
  upper << f.servo_data[0].upper, f.servo_data[1].upper, 1.6, 1.6, 0.02, 3.2;

  auto residual_fn = [this, &f, q1, q_prox](const Eigen::VectorXd& x) { return residualKnuckle(f, x, q1, q_prox); };
  auto sol = solveMultistart(residual_fn, f.warm_start, lower, upper);
  f.warm_start = sol.x;

  const double theta_a = sol.x[0], theta_b = sol.x[1], q2 = sol.x[2];
  const double q_knu = sol.x[3], q_prism = sol.x[4], q_twist = sol.x[5];
  std::map<std::string, double> solved{
    {f.servo_data[0].actuator_joint, theta_a}, {f.servo_data[1].actuator_joint, theta_b},
    {f.q1_joint, q1},                          {f.q2_joint, q2},
    {f.prox_joint, q_prox},                    {f.knuckle_joint, q_knu},
    {f.prism_joint, q_prism},                  {f.twist_joint, q_twist},
  };

  f.last_solve_args = args;
  f.last_solved = solved;
  f.last_cost = sol.cost;
  f.has_last_solve = true;
  ++f.solve_generation;

  return {sol, solved};
}

void HandSolver::solveRodChains(
  FingerInfo& f, const std::map<std::string, double>& raw, std::map<std::string, double>& solved)
{
  // See FingerInfo::rod_solved_generation - replaces the python original's
  // `is` identity check on the cached solved-result dict.
  if (f.rod_solved_generation == f.solve_generation) {
    return;
  }
  f.rod_solved_generation = f.solve_generation;

  std::map<std::string, double> vals = raw;
  for (const auto& [k, v] : solved) {
    vals[k] = v;
  }

  std::map<std::string, Eigen::Vector3d> ball_pos;
  for (int i = 0; i < 2; ++i) {
    ball_pos[f.servo_data[i].horn_ball] = hand_->fk(f.servo_data[i].horn_ball, vals).block<3, 1>(0, 3);
    ball_pos[f.rod_balls[i]] = hand_->fk(f.rod_balls[i], vals).block<3, 1>(0, 3);
  }

  const Eigen::VectorXd lower = Eigen::VectorXd::Constant(3, -1e9);
  const Eigen::VectorXd upper = Eigen::VectorXd::Constant(3, 1e9);

  for (std::size_t ci = 0; ci < f.rod_chains.size(); ++ci) {
    const RodChain& chain = f.rod_chains[ci];
    Eigen::Vector3d& x0 = f.rod_warm_start[ci];
    const Eigen::Vector3d target = ball_pos.at(chain.target_ball);
    const auto& cjoints = chain.joints;
    const std::string& terminal = chain.terminal;
    const Eigen::Vector3d& aim_local = chain.aim_local;

    auto residual_fn = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
      std::map<std::string, double> cv = vals;
      cv[cjoints[0]] = x[0];
      cv[cjoints[1]] = x[1];
      cv[cjoints[2]] = x[2];
      Eigen::Matrix4d T = hand_->fk(terminal, cv);
      Eigen::Vector3d err = T.block<3, 3>(0, 0) * aim_local + T.block<3, 1>(0, 3) - target;
      // The point constraint leaves roll about the rod axis free; softly
      // prefer minimal rotation from rest (weight small enough not to
      // disturb the point match) instead of drifting.
      Eigen::VectorXd out(6);
      out.segment<3>(0) = err;
      out.segment<3>(3) = 2e-3 * x;
      return out;
    };

    auto sol = boundedLeastSquares(residual_fn, x0, lower, upper, 1e-8, 1e-8, 60);
    if (sol.x.norm() > 0.3) {
      auto sol_rest = boundedLeastSquares(residual_fn, Eigen::Vector3d::Zero(), lower, upper, 1e-8, 1e-8, 60);
      if (sol_rest.cost <= sol.cost) {
        sol = sol_rest;
      }
    }
    x0 = sol.x;
    solved[cjoints[0]] = sol.x[0];
    solved[cjoints[1]] = sol.x[1];
    solved[cjoints[2]] = sol.x[2];
  }
}

std::map<std::string, double> HandSolver::solve(const std::map<std::string, double>& raw_in)
{
  std::map<std::string, double> raw = raw_in;
  for (const auto& [alias, real] : alias_of_) {
    auto it = raw.find(alias);
    if (it != raw.end()) {
      raw[real] = alias_sign_.at(alias) * it->second;
    }
  }

  std::map<std::string, double> out;
  for (const auto& [name, value] : raw_in) {
    if (!alias_of_.count(name)) {
      out[name] = value;
    }
  }

  last_out_of_reach_.clear();
  for (auto& f : fingers_) {
    LeastSquaresResult sol;
    std::map<std::string, double> solved;
    if (command_space_ == "servo") {
      std::tie(sol, solved) = solveFingerServo(f, raw);
    } else {
      std::tie(sol, solved) = solveFingerKnuckle(f, raw);
    }
    solveRodChains(f, raw, solved);

    for (const auto& [alias, real] : alias_of_) {
      auto it = solved.find(real);
      if (it != solved.end()) {
        alias_state_[alias] = alias_sign_.at(alias) * it->second;
      }
    }

    if (sol.cost > 1e-6) {
      last_out_of_reach_.push_back(f.gimbal);
    }

    for (const auto& [jn, v] : solved) {
      out[jn] = v;
    }
  }

  for (const auto& jn : all_movable_) {
    if (!out.count(jn)) {
      out[jn] = 0.0;
    }
  }

  return out;
}

}  // namespace amazing_hand_kinematics
