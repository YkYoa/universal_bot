#pragma once

// Visibility macros for the openarm_hardware shared library.
// Pattern adapted from ROS 2 control plugin conventions.

#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef __GNUC__
#define OPENARM_HARDWARE_EXPORT __attribute__((dllexport))
#define OPENARM_HARDWARE_IMPORT __attribute__((dllimport))
#else
#define OPENARM_HARDWARE_EXPORT __declspec(dllexport)
#define OPENARM_HARDWARE_IMPORT __declspec(dllimport)
#endif
#ifdef OPENARM_HARDWARE_BUILDING_DLL
#define OPENARM_HARDWARE_PUBLIC OPENARM_HARDWARE_EXPORT
#else
#define OPENARM_HARDWARE_PUBLIC OPENARM_HARDWARE_IMPORT
#endif
#else
#define OPENARM_HARDWARE_EXPORT __attribute__((visibility("default")))
#define OPENARM_HARDWARE_IMPORT
#if __GNUC__ >= 4
#define OPENARM_HARDWARE_PUBLIC __attribute__((visibility("default")))
#define OPENARM_HARDWARE_LOCAL __attribute__((visibility("hidden")))
#else
#define OPENARM_HARDWARE_PUBLIC
#define OPENARM_HARDWARE_LOCAL
#endif
#endif
