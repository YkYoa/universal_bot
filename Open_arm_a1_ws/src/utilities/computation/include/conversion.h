#pragma once

#include <vector>

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

} // namespace computation
} // namespace utilities
