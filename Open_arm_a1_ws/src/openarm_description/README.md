# openarm_description

URDF/Xacro robot description for the OpenArm A1 (hardware generation **v1.0**):
7-DOF arm, mobile base, body/head, and two interchangeable end-effectors
(`openarm_hand` gripper and the custom `amazing_hand`).

## Layout

Mirrors the asset-based layout upstream ([enactic/openarm_description](https://github.com/enactic/openarm_description))
uses for its `openarm_v1.0` legacy-compatibility tree, adapted to keep this
package's existing xacro/CMake (`ament_cmake`) build — this repo does not use
upstream's newer Python/`pyproject.toml` pipeline (that pipeline targets
OpenArm 2.0 hardware and has no `amazing_hand` support).

```
assets/
  robot/openarm_v1.0/       # arm + body + base + ros2_control + top-level robot xacro
    urdf/                   #   arm/, body/, base/, ros2_control/, openarm_robot.xacro,
                             #   base.urdf.xacro, v10.urdf.xacro (+ v10.urdf snapshot), v10/*.usd
    meshes/                 #   arm/, body/, base/ (visual + collision STL/DAE)
    config/                 #   arm/, body/, base/ (kinematics, inertials, joint limits, gains)
  end_effector/
    openarm_hand/            # stock 1-DOF parallel gripper
      urdf/ meshes/ config/
    amazing_hand/             # custom tendon-driven hand (this project's addition, not upstream)
      urdf/ meshes/ config/
    ee_with_one_link.xacro   # generic single-mesh EE macro, not currently wired into
                              # openarm_robot.xacro -- kept for a future simple EE variant
launch/                      # ros2 launch entry points (see terminal_command.md)
rviz/                        # RViz configs referenced by the launch files
scripts/                     # hand_kinematics_node.py (amazing_hand linkage solver)
```

Every xacro/launch file resolves paths through `$(find openarm_description)`
or `FindPackageShare("openarm_description")` plus the path shown above --
nothing is hardcoded outside this package's own tree. `ee_type` selects
`assets/end_effector/<ee_type>/` and its `<ee_type>_arguments.xacro`
dynamically, so both hands build from the same `openarm_robot.xacro`.

## Adding a new arm/body/base version

Add a new subfolder under `assets/robot/` (e.g. `openarm_v1.1/`) with the
same `urdf/`, `meshes/`, `config/` shape, then point a new top-level xacro
(like `v10.urdf.xacro`) at it. Don't add version-conditional branches inside
the existing `openarm_v1.0` files -- each hardware generation gets its own
folder, same as `openarm_hand` and `amazing_hand` each get their own.

## Adding a new end-effector

Add `assets/end_effector/<name>/{urdf,meshes,config}/` plus a
`<name>_arguments.xacro` (mount offset defaults) and a `<name>_macro.xacro`
(the xacro macro `openarm_robot.xacro` calls). Wire the new macro's
`xacro:include` into `openarm_robot.xacro` next to the existing
`openarm_hand`/`amazing_hand` ones -- `ee_type:=<name>` then selects it.

## Commands

See [terminal_command.md](terminal_command.md).
