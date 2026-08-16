# CAN Motor Zero-Position Calibration

Real-hardware firmware-stage calibration for the Damiao motors on OpenArm
(IQ9075). Use this when a joint's reported position no longer matches its
real physical position (e.g. after the arm "lost home", or after adjusting
where "home" should physically be for motor 8's connector).

## Before you start

- **Stop bringup first** - no `ros2_control_node` should be running, or it
  will fight the calibration tool for the CAN bus:
  ```bash
  ps aux | grep ros2_control_node   # should show nothing but the grep itself
  ```
  If something's running, `Ctrl+C` the bringup terminal (or `pkill -f
  ros2_control_node`) before continuing.

- **CAN interface per side**: `left_can: can1`, `right_can: can0` (see
  `config/hardware_config.yaml`). Motor 8 (the gripper/"motor 8" CAN
  channel) shares the same bus/interface as that side's 7 arm joints - it's
  motor ID `8` on the same `canN`.

## DO NOT use `openarm-can-zero-position-calibration` for motor 8

That tool is the vendor's compact calibration script, designed for
`openarm_hand`'s stock 2-finger gripper - its zero-finding routine seeks a
**mechanical hard stop** (the gripper jaw fully closing). If ee_type is
`amazing_hand`, motor 8 instead drives the connector rotation, which
generally has no hard stop in that same direction - the tool will spin the
motor indefinitely searching for a stop that isn't there. This has actually
happened once already; don't run it against motor 8 again.

It's still fine for joints 1-7 (`--arm-side left_arm/right_arm`), which do
have real hard-stop-seeking logic appropriate for the arm itself. But the
lower-level procedure below (`set_zero`) works for everything, doesn't seek
anything, and is what's used for motor 8 either way.

## Procedure: `openarm-can-cli set_zero` (safe for any motor ID)

`set_zero` does **not** move or seek anything - it just tells the motor
"whatever position you're at right now is your new zero". You position the
joint by hand first, then the tool records it.

```bash
# 1. Disable torque so you can move the joint(s) by hand.
#    Left arm joints 1-7:      openarm-can-cli -i can1 disable --id 1,2,3,4,5,6,7
#    Right arm joints 1-7:     openarm-can-cli -i can0 disable --id 1,2,3,4,5,6,7
#    Left motor 8 only:        openarm-can-cli -i can1 disable --id 8
#    Right motor 8 only:       openarm-can-cli -i can0 disable --id 8
openarm-can-cli -i <can_interface> disable --id <motor_id(s)>

# 2. By hand, move the joint(s) to the position you want to become the new zero.
#    - Arm joints 1-7: the arm's true mechanical home pose (fully straight,
#      as originally designed).
#    - Motor 8 (amazing_hand connector): wherever you want "0 rad" to mean
#      for openarm_left_finger_joint1/openarm_right_finger_joint1 - e.g. if
#      the connector currently reads a small offset from where it should be
#      centered, nudge it to the corrected position.

# 3. Record that position as the new zero.
openarm-can-cli -i <can_interface> set_zero --id <motor_id(s)>

# 4. Verify - Motor Position / Output Shaft Position should now read ~0.
openarm-can-cli -i <can_interface> show_param --id <motor_id(s)>

# 5. Re-enable torque.
openarm-can-cli -i <can_interface> enable --id <motor_id(s)>
```

### Examples

**Left arm joints 1-7** (full re-home after the arm lost its zero reference):
```bash
openarm-can-cli -i can1 disable --id 1,2,3,4,5,6,7
# -- move arm by hand to straight/home pose --
openarm-can-cli -i can1 set_zero --id 1,2,3,4,5,6,7
openarm-can-cli -i can1 show_param --id 1,2,3,4,5,6,7
openarm-can-cli -i can1 enable --id 1,2,3,4,5,6,7
```

**Left motor 8 only** (small home-position adjustment, doesn't touch
joints 1-7 at all):
```bash
openarm-can-cli -i can1 disable --id 8
# -- move the connector by hand a little to the corrected zero position --
openarm-can-cli -i can1 set_zero --id 8
openarm-can-cli -i can1 show_param --id 8
openarm-can-cli -i can1 enable --id 8
```
Substitute `can0` for the right arm's motor 8.

## After recalibrating motor 8

The joint limits (`hand_rotate_lower`/`hand_rotate_upper` in
`openarm_description/urdf/ee/amazing_hand_arguments.xacro`, default
`-1.570796`/`1.570796` = +-90deg) are relative to wherever "0" is defined by
this calibration. If you moved zero by a non-trivial amount, re-check that
the mechanism can still physically reach both limits without hitting
anything, and adjust `hand_rotate_lower`/`hand_rotate_upper` (or
`hand_rotate_axis` if the rotation direction itself seems backwards) via
launch args if needed - see the joint's comment in
`openarm_description/urdf/robot/openarm_robot.xacro`.
