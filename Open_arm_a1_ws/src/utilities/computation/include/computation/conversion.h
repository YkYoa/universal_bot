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
geometry_msgs::msg::Pose vecToPoseMsgs(const std::vector<double>& pose);
std::string vecToString(const std::vector<double>& vec);
std::string tfvec3ToString(const tf2::Vector3& vec);
std::string tfQuatToString(const tf2::Quaternion& quat);

std::vector<double> poseToVec(const tf2::Vector3& pos);
std::vector<double> quatToVec(const tf2::Quaternion& quat);
std::vector<double> poseQuatToVec(const tf2::Vector3& pos, const tf2::Quaternion& quat);
geometry_msgs::msg::Pose poseQuatToPoseMsgs(const tf2::Vector3& pos, const tf2::Quaternion& quat);
void vecToPoseQuat(const std::vector<double>& input, tf2::Vector3& pos, tf2::Quaternion& quat);
tf2::Transform vecToTfTransform(const std::vector<double>& input);
Eigen::Matrix4d vecToEigenMatrix(const std::vector<double>& input);

Eigen::VectorXd vecToEigenVec(std::vector<double> vec);
std::vector<double> eigenVecToVec(Eigen::VectorXd eigenVec);

std::vector<double> quatToEuler(const tf2::Quaternion& quat);
std::vector<double> quatToEuler(const double x, const double y, const double z, const double w);
tf2::Matrix3x3 eulerToTfMatrix(const std::vector<double>& euler_angles);
std::vector<double> tfMatrixToEuler(const tf2::Matrix3x3& rotation_matrix);

tf2::Matrix3x3 quaternionToTfMatrix(const tf2::Quaternion& quat);
tf2::Quaternion tfMatrixToQuaternion(const tf2::Matrix3x3& rotation_matrix);

geometry_msgs::msg::Pose tfTransStampedToPoseMsgs(const geometry_msgs::msg::TransformStamped& transformStamped);
tf2::Transform posRotToTfTransform(const tf2::Matrix3x3& rotation, const tf2::Vector3& translation);
tf2::Transform eigenToTfTransform(const Eigen::Matrix4d& eigen_mat);
Eigen::Matrix4d transformStampedToMatrix(const geometry_msgs::msg::TransformStamped& transformStamped);

tf2::Vector3 StringToTfvec3(const std::string& input_str);
tf2::Quaternion StringToTfQuat(const std::string& input_str);
std::vector<double> StringToDoubleVec(const std::string& input_str);

} // namespace computation
} // namespace utilities
