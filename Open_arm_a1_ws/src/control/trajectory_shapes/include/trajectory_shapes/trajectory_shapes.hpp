#pragma once

#include <Eigen/Geometry>
#include <vector>

namespace trajectory_shapes
{

// Straight line from start to end, inclusive - num_points+1 total samples
// (t = 0, 1/num_points, ..., 1).
std::vector<Eigen::Vector3d> line(const Eigen::Vector3d& start, const Eigen::Vector3d& end, int num_points);

// Half-ellipse arc from start to end, inclusive - ported from
// runWavingCartesianSequence() (builds/openarm_demo/src/openarm_demo_node.cpp).
// height_ratio = semi-minor/semi-major axis ratio (controls arc height);
// 0 degenerates to a straight line. num_points+1 total samples. Returns an
// empty vector if start and end (nearly) coincide - degenerate, no arc.
std::vector<Eigen::Vector3d> ellipseArc(
  const Eigen::Vector3d& start, const Eigen::Vector3d& end, double height_ratio, int num_points);

// Circle (or an arc of one), in the plane through `center` perpendicular to
// `normal`, starting at start_angle_rad and sweeping to end_angle_rad
// (default a full 0..2*pi sweep - first and last sample coincide).
// num_points+1 total samples. `normal` need not be unit length.
std::vector<Eigen::Vector3d> circle(
  const Eigen::Vector3d& center, double radius, const Eigen::Vector3d& normal, int num_points,
  double start_angle_rad = 0.0, double end_angle_rad = 2.0 * M_PI);

// Square perimeter centered at `center`, in the plane through `normal`,
// side length `side_length`. Closed loop: 4*num_points_per_side+1 total
// samples (last sample coincides with the first). NOTE: unlike the other
// shapes, num_points_per_side is a PER-SIDE count, not a total.
std::vector<Eigen::Vector3d> square(
  const Eigen::Vector3d& center, double side_length, const Eigen::Vector3d& normal, int num_points_per_side);

// Shared orientation helper - every shape above only produces positions;
// callers pair them with an orientation sweep via this, using the same t
// (0..1) used for position sampling. Handles the shorter-arc / near-
// identical-quaternion cases; result is always unit-normalized.
Eigen::Quaterniond slerp(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2, double t);

}  // namespace trajectory_shapes
