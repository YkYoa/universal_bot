#pragma once

#include <vector>
#include <string>
#include <map>
#include <tf2/LinearMath/Vector3.h>

namespace utilities 
{
namespace computation
{
    typedef enum
    {
        PLANE_XY,
        PLANE_YZ,
        PLANE_ZX
    } Plane;

    typedef enum
    {
        AXIS_X,
        AXIS_Y,
        AXIS_Z
    } Axis;

    typedef struct
    {
        tf2::Vector3 origin;
        tf2::Vector3 normal_vector;
    } PlaneParameter;

} // namespace computation
} // namespace utilities