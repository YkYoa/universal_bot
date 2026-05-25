# `devices` — OpenArm Low-Level Communication Drivers

This package is a placeholder/stub intended for housing custom low-level device communication drivers (e.g. serial interfaces, CAN-open wrappers, MODBUS interfaces, or socket clients) for sensors, cameras, and grippers integrated into the OpenArm system.

---

## Intended Purpose

When integrating new physical actuators or hardware:
* Place low-level drivers (such as serial protocols to command custom grippers or force/torque sensor listeners) in this package.
* Expose their states/actions via ROS 2 nodes or interfaces so they can be read by `openarm_hardware` or other control layers.
