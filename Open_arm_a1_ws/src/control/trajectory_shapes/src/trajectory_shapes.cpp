#include "trajectory_shapes/trajectory_shapes.hpp"

#include <array>
#include <cmath>

namespace trajectory_shapes
{

namespace
{

// Returns a unit vector perpendicular to axis_unit (which must already be
// unit length). Picks world-Z as the reference to cross with, falling back
// to world-X when axis_unit is nearly parallel to Z - same trick used
// throughout this codebase's Cartesian-path code (e.g. the original
// runWavingCartesianSequence) to avoid a near-zero cross product.
Eigen::Vector3d pickPerpendicular(const Eigen::Vector3d& axis_unit)
{
  Eigen::Vector3d ref(0.0, 0.0, 1.0);
  if (std::abs(axis_unit.dot(ref)) > 0.95) {
    ref = Eigen::Vector3d(1.0, 0.0, 0.0);
  }
  return axis_unit.cross(ref).normalized();
}

}  // namespace

std::vector<Eigen::Vector3d> line(const Eigen::Vector3d& start, const Eigen::Vector3d& end, int num_points)
{
  std::vector<Eigen::Vector3d> points;
  points.reserve(static_cast<size_t>(num_points) + 1);
  for (int i = 0; i <= num_points; ++i) {
    const double t = static_cast<double>(i) / num_points;
    points.push_back(start + t * (end - start));
  }
  return points;
}

std::vector<Eigen::Vector3d> ellipseArc(
  const Eigen::Vector3d& start, const Eigen::Vector3d& end, double height_ratio, int num_points)
{
  const Eigen::Vector3d midpoint = (start + end) / 2.0;
  const Eigen::Vector3d delta = end - start;
  const double dist = delta.norm();
  if (dist < 1e-9) {
    return {};  // degenerate - start and end coincide, no arc to draw
  }

  const Eigen::Vector3d u = delta / dist;  // major axis direction
  const Eigen::Vector3d v = pickPerpendicular(u);  // arc-height direction
  const double a = dist / 2.0;
  const double b = a * height_ratio;

  std::vector<Eigen::Vector3d> points;
  points.reserve(static_cast<size_t>(num_points) + 1);
  for (int i = 0; i <= num_points; ++i) {
    const double t = static_cast<double>(i) / num_points;
    const double theta = t * M_PI;
    // At theta=0 -> midpoint - a*u = start; at theta=pi -> midpoint + a*u = end.
    points.push_back(midpoint - a * std::cos(theta) * u + b * std::sin(theta) * v);
  }
  return points;
}

std::vector<Eigen::Vector3d> circle(
  const Eigen::Vector3d& center, double radius, const Eigen::Vector3d& normal, int num_points,
  double start_angle_rad, double end_angle_rad)
{
  const Eigen::Vector3d n = normal.normalized();
  const Eigen::Vector3d u = pickPerpendicular(n);
  const Eigen::Vector3d v = n.cross(u).normalized();

  std::vector<Eigen::Vector3d> points;
  points.reserve(static_cast<size_t>(num_points) + 1);
  for (int i = 0; i <= num_points; ++i) {
    const double t = static_cast<double>(i) / num_points;
    const double theta = start_angle_rad + t * (end_angle_rad - start_angle_rad);
    points.push_back(center + radius * std::cos(theta) * u + radius * std::sin(theta) * v);
  }
  return points;
}

std::vector<Eigen::Vector3d> square(
  const Eigen::Vector3d& center, double side_length, const Eigen::Vector3d& normal, int num_points_per_side)
{
  const Eigen::Vector3d n = normal.normalized();
  const Eigen::Vector3d u = pickPerpendicular(n);
  const Eigen::Vector3d v = n.cross(u).normalized();
  const double h = side_length / 2.0;

  const std::array<Eigen::Vector3d, 4> corners = {
    center - h * u - h * v,
    center + h * u - h * v,
    center + h * u + h * v,
    center - h * u + h * v,
  };

  std::vector<Eigen::Vector3d> points;
  points.reserve(static_cast<size_t>(4 * num_points_per_side) + 1);
  for (int side = 0; side < 4; ++side) {
    const Eigen::Vector3d& edge_start = corners[static_cast<size_t>(side)];
    const Eigen::Vector3d& edge_end = corners[static_cast<size_t>((side + 1) % 4)];
    // Skip the duplicate corner at every join except the very first sample.
    const int begin_k = (side == 0) ? 0 : 1;
    for (int k = begin_k; k <= num_points_per_side; ++k) {
      const double t = static_cast<double>(k) / num_points_per_side;
      points.push_back(edge_start + t * (edge_end - edge_start));
    }
  }
  return points;
}

Eigen::Quaterniond slerp(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2, double t)
{
  double dot = q1.x() * q2.x() + q1.y() * q2.y() + q1.z() * q2.z() + q1.w() * q2.w();
  Eigen::Quaterniond q2c = q2;
  if (dot < 0.0) {
    dot = -dot;
    q2c = Eigen::Quaterniond(-q2.w(), -q2.x(), -q2.y(), -q2.z());
  }

  Eigen::Quaterniond result;
  if (dot > 0.9995) {
    result = Eigen::Quaterniond(
      q1.w() + t * (q2c.w() - q1.w()), q1.x() + t * (q2c.x() - q1.x()), q1.y() + t * (q2c.y() - q1.y()),
      q1.z() + t * (q2c.z() - q1.z()));
  } else {
    const double theta = std::acos(dot);
    const double sin_t = std::sin(theta);
    const double w1 = std::sin((1.0 - t) * theta) / sin_t;
    const double w2 = std::sin(t * theta) / sin_t;
    result = Eigen::Quaterniond(
      w1 * q1.w() + w2 * q2c.w(), w1 * q1.x() + w2 * q2c.x(), w1 * q1.y() + w2 * q2c.y(),
      w1 * q1.z() + w2 * q2c.z());
  }
  result.normalize();
  return result;
}

}  // namespace trajectory_shapes
