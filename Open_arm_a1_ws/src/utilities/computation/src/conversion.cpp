#include "conversion.h"
#include <cmath>

namespace utilities {
namespace computation {

double degreesToRadians(double degrees) {
    return degrees * (M_PI / 180.0);
}

std::vector<double> degreesToRadians(const std::vector<double>& degrees) {
    std::vector<double> radians;
    radians.reserve(degrees.size());
    for (double d : degrees) {
        radians.push_back(degreesToRadians(d));
    }
    return radians;
}

double radiansToDegrees(double radians) {
    return radians * (180.0 / M_PI);
}

std::vector<double> radiansToDegrees(const std::vector<double>& radians) {
    std::vector<double> degrees;
    degrees.reserve(radians.size());
    for (double r : radians) {
        degrees.push_back(radiansToDegrees(r));
    }
    return degrees;
}

} // namespace computation
} // namespace utilities
