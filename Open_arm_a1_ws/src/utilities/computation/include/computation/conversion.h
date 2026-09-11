#pragma once

#include <vector>
#include <string>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <Eigen/Dense>

namespace utilities {
namespace computation {

/**
 * @brief Convert degrees to radians.
 * @param degrees Angle in degrees
 * @return Angle in radians
 */
double degreesToRadians(double degrees);

/**
 * @brief Convert a vector of degrees to radians.
 * @param degrees Vector of angles in degrees
 * @return Vector of angles in radians
 */
std::vector<double> degreesToRadians(const std::vector<double>& degrees);

/**
 * @brief Convert radians to degrees.
 * @param radians Angle in radians
 * @return Angle in degrees
 */
double radiansToDegrees(double radians);

/**
 * @brief Convert a vector of radians to degrees.
 * @param radians Vector of angles in radians
 * @return Vector of angles in degrees
 */
std::vector<double> radiansToDegrees(const std::vector<double>& radians);

// --- User-added conversions ---
/// [x,y,z,qx,qy,qz,qw] (must have >= 7 elements, else logs an error and
/// returns a default Pose) -> geometry_msgs/Pose.
geometry_msgs::msg::Pose vecToPoseMsgs(const std::vector<double>& pose);
/// Comma-joined values, e.g. "1.000000,2.000000".
std::string vecToString(const std::vector<double>& vec);
/// "(x, y, z)".
std::string tfvec3ToString(const tf2::Vector3& vec);
/// "(x, y, z, w)".
std::string tfQuatToString(const tf2::Quaternion& quat);

/// tf2::Vector3 -> [x, y, z].
std::vector<double> poseToVec(const tf2::Vector3& pos);
/// tf2::Quaternion -> [x, y, z, w].
std::vector<double> quatToVec(const tf2::Quaternion& quat);
/// Concatenation of poseToVec(pos) and quatToVec(quat): [x,y,z,qx,qy,qz,qw].
std::vector<double> poseQuatToVec(const tf2::Vector3& pos, const tf2::Quaternion& quat);
/// Builds a geometry_msgs/Pose directly from a position + orientation pair.
geometry_msgs::msg::Pose poseQuatToPoseMsgs(const tf2::Vector3& pos, const tf2::Quaternion& quat);
/// Inverse of poseQuatToVec(): unpacks a 7-value [x,y,z,qx,qy,qz,qw] vector
/// into `pos`/`quat` (logs an error and leaves them untouched if too short).
void vecToPoseQuat(const std::vector<double>& input, tf2::Vector3& pos, tf2::Quaternion& quat);
/// [x,y,z,qx,qy,qz,qw] -> tf2::Transform (identity, with an error log, if too short).
tf2::Transform vecToTfTransform(const std::vector<double>& input);
/// [x,y,z,qx,qy,qz,qw] -> a 4x4 homogeneous Eigen transform matrix (identity,
/// with an error log, if too short).
Eigen::Matrix4d vecToEigenMatrix(const std::vector<double>& input);

/// Wraps `vec`'s data as an Eigen::VectorXd (no copy of values, but `vec` is
/// taken by value so the wrapped buffer is this call's own copy).
Eigen::VectorXd vecToEigenVec(std::vector<double> vec);
/// Eigen::VectorXd -> std::vector<double>.
std::vector<double> eigenVecToVec(Eigen::VectorXd eigenVec);

/// Quaternion -> [roll, pitch, yaw] (radians).
std::vector<double> quatToEuler(const tf2::Quaternion& quat);
/// Same as the tf2::Quaternion overload, from raw x/y/z/w components.
std::vector<double> quatToEuler(const double x, const double y, const double z, const double w);
/// [roll, pitch, yaw] (radians, must have exactly 3 elements) -> rotation matrix.
tf2::Matrix3x3 eulerToTfMatrix(const std::vector<double>& euler_angles);
/// Rotation matrix -> [roll, pitch, yaw] (radians).
std::vector<double> tfMatrixToEuler(const tf2::Matrix3x3& rotation_matrix);

/// Quaternion -> rotation matrix (normalizes first; logs an error and
/// returns identity if `quat` is ~zero-length).
tf2::Matrix3x3 quaternionToTfMatrix(const tf2::Quaternion& quat);
/// Rotation matrix -> quaternion.
tf2::Quaternion tfMatrixToQuaternion(const tf2::Matrix3x3& rotation_matrix);

/// TransformStamped's translation+rotation -> geometry_msgs/Pose.
geometry_msgs::msg::Pose tfTransStampedToPoseMsgs(const geometry_msgs::msg::TransformStamped& transformStamped);
/// Builds a tf2::Transform from a separate rotation matrix and translation.
tf2::Transform posRotToTfTransform(const tf2::Matrix3x3& rotation, const tf2::Vector3& translation);
/// 4x4 homogeneous Eigen matrix -> tf2::Transform.
tf2::Transform eigenToTfTransform(const Eigen::Matrix4d& eigen_mat);
/// TransformStamped -> a 4x4 homogeneous Eigen transform matrix.
Eigen::Matrix4d transformStampedToMatrix(const geometry_msgs::msg::TransformStamped& transformStamped);

/// Parses a "x,y,z"-style string into a tf2::Vector3.
tf2::Vector3 StringToTfvec3(const std::string& input_str);
/// Parses a "x,y,z,w"-style string into a tf2::Quaternion.
tf2::Quaternion StringToTfQuat(const std::string& input_str);
/// Parses a comma-separated string into a vector of doubles.
std::vector<double> StringToDoubleVec(const std::string& input_str);

} // namespace computation
} // namespace utilities
