# OpenArm A1 Workspace — Getting Started

ROS 2 (Jazzy) workspace for the OpenArm A1 bimanual robot. This page is the
on-ramp: follow it top to bottom once, and by the end you can build the
workspace, read the generated code docs, run the arm in simulation, and send
real motion commands to the physical robot (IQ9075).

For day-to-day operation once you're set up (bringup, calibration, control
commands, health checks), use **[`terminal_command.md`](../terminal_command.md)**
instead — this page won't repeat it.

---

## 0. What you're setting up

- **Target OS for real use:** Ubuntu 24.04 (Noble) + ROS 2 **Jazzy**. That's
  what the real robot (IQ9075) runs, and what every launch file/hardware
  interface in this workspace is built and tested against.
- **The robot itself** runs headless on its own onboard computer
  (`ssh ubuntu@192.168.1.226` — ask a teammate for access). Most people don't
  develop *on* the robot: you build this workspace on your own machine,
  simulate there, then either `ssh` in to run things directly on the robot or
  use `./deploy_to_robot.sh` to push your code to it (see step 6).
- **Windows:** there's no supported native-Windows path for this repo — parts
  of it are Linux-only (Unix domain sockets in `head_motor_driver_node`,
  SocketCAN for the arms, realtime scheduling for `ros2_control`). Use
  **WSL2 + Ubuntu 24.04** (step 1b) and follow the Ubuntu instructions inside
  it. You'll still be able to build, simulate, and remotely control the real
  arm over the network from there — you just can't plug a CAN adapter
  directly into a WSL2 instance.

---

## 1a. Install ROS 2 Jazzy — Ubuntu 24.04

Official install, condensed. Full reference: https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html

```bash
# UTF-8 locale (skip if `locale` already shows en_US.UTF-8)
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# ROS 2 apt repo
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install
sudo apt update && sudo apt upgrade -y
sudo apt install -y ros-jazzy-desktop ros-dev-tools python3-colcon-common-extensions python3-rosdep

# One-time rosdep setup
sudo rosdep init
rosdep update

# Source it every new shell (or add this line to ~/.bashrc)
source /opt/ros/jazzy/setup.bash
```

## 1b. Install ROS 2 Jazzy — Windows (via WSL2)

From an elevated PowerShell:

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot if prompted, open the "Ubuntu-24.04" app from the Start menu, finish
the Linux user setup, then follow **1a above inside that WSL2 terminal**.
Everything else in this guide (build, docs, simulation, REST control) works
the same from there — you're just in Ubuntu.

---

## 2. Get the workspace and install its package dependencies

```bash
git clone <this repo>          # or: git pull, if you already have it
cd Open_arm_a1_ws
```

**`ros-jazzy-desktop` (step 1) does not include MoveIt or `ros2_control`** —
those are separate metapackages, and several packages here (`motion_planner`,
`robot_hardware_interface`, `robot_skills`, ...) need them to even build.
Install those explicitly first:

```bash
sudo apt install -y \
  ros-jazzy-moveit \
  ros-jazzy-moveit-planners-ompl \
  ros-jazzy-moveit-simple-controller-manager \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-xacro \
  ros-jazzy-joint-state-publisher-gui
```

Then let `rosdep` resolve everything else this workspace's packages declare —
the finer-grained ROS packages (`joint_trajectory_controller`, `cv_bridge`,
`kdl_parser`, ...) plus system libs (`libeigen3-dev`, `libyaml-cpp-dev`,
`libsqlite3-dev`, numpy/scipy, ...) — in one shot, instead of hand-listing
every `ros-jazzy-<package>` yourself:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

Two things `rosdep` won't cover:

- **Flask stack for the REST API** (`moveit_api`, `camera_bridge`) — if
  `rosdep` doesn't resolve `python3-flask`/`python3-flask-socketio` on your
  system, install via **apt, not pip**:
  ```bash
  sudo apt install -y python3-flask python3-flask-socketio python3-flask-compress
  ```
  (`pip install --user` hits `externally-managed-environment` on Ubuntu
  24.04, and a venv would also need `--system-site-packages` to still see the
  apt-installed `rclpy`.)
- **OpenArm's own CAN driver/tools** (`openarm-can-cli`,
  `openarm-can-configure-socketcan-4-arms`) — these aren't ROS packages and
  aren't in this repo. You only need them to talk to real CAN hardware
  (bringup, calibration); ask a teammate for the vendor install steps if
  you're setting up a new machine that will touch the real arm's CAN bus.
  Skip this if you're only simulating.

---

## 3. Build

```bash
cd Open_arm_a1_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

`--symlink-install` means editing a Python file under `src/` takes effect
without rebuilding; C++ changes still need a rebuild. Re-run `colcon build`
(scoped with `--packages-select <pkg>` if you only touched one package) after
pulling new code or editing C++.

---

## 4. Read the code docs (Doxygen)

Every package, class, and function in `src/` is documented — this is the
fastest way to actually understand what a node/class does before reading its
source line by line.

```bash
sudo apt install -y doxygen graphviz   # one-time
cd Open_arm_a1_ws
doxygen Doxyfile
xdg-open docs/doxygen/html/index.html  # or open the file in a browser manually
```

Where to look:

- **Topics** tab — the workspace grouped the way it's actually organized:
  Control (further split into Planning & Trajectories / MoveIt Integration /
  Hardware & Execution), Communication, Utilities, Description, Builds.
- **Classes** tab — every class with a one-line (or longer) description of
  what it does and why.
- **Files** tab — browse by directory if you already know roughly where
  something lives.

Re-run `doxygen Doxyfile` any time after pulling new code to refresh it —
output goes to `docs/doxygen/` (gitignored, generated locally, not committed).

---

## 5. Workspace layout, in one table

| Folder | What's in it |
|---|---|
| `openarm_description/` | URDF/Xacro robot model, meshes, launch files to just *view* the robot (RViz, no motors) |
| `communication/` | Message/action/service definitions, camera bridge, MQTT/database bridge, low-level device nodes (head LED, head motor driver, head homing, arm API protocol) |
| `control/` | Everything that actually moves the robot: `robot_hardware_interface` (ros2_control plugin, real CAN motors), `moveit_api` (REST/WebSocket control server), `robot_skills` + `sequence_executor` (composable motions / FSM), `motion_planner` + `planning_interface` (MoveIt planning glue), `amazing_hand_kinematics` + `gravity_compensation_controller` (hand linkage solver, free-drive), plus trajectory/waypoint helpers |
| `utilities/` | Shared math/computation helpers used by the above |
| `builds/` | Top-level demo/competition bring-up packages (`openarm_demo`, `openarm_test`, `qvic_2026`) |

Package-specific README's worth knowing about:
[`control/moveit_api/README.md`](control/moveit_api/README.md) (REST API),
[`builds/qvic_2026/README.md`](builds/qvic_2026/README.md) (competition app),
[`control/robot_hardware_interface/CALIBRATION.md`](control/robot_hardware_interface/CALIBRATION.md)
(real-motor zero calibration — read before ever touching real hardware).

---

## 6. Try it in simulation first

Before touching the real arm, bring the stack up with fake hardware so you
can see it move in RViz without risk:

```bash
ros2 launch robot_hardware_interface bringup.launch.py \
  arms:=false head:=false use_rviz:=true ee_type:=amazing_hand body_type:=v2
```

or the full REST API + 3D dashboard, also fake:

```bash
ros2 launch moveit_api robot_api.launch.py \
  use_rviz:=false use_moveit:=true use_api:=true use_controllers:=true
```
then open `http://localhost:5050/dashboard/` and `http://localhost:5050/api/docs`.

---

## 7. Control the real robot

This is where you switch to **[`terminal_command.md`](../terminal_command.md)**
for the full, accurate runbook — CAN link checks, motor zero calibration,
the real bringup command (`arms:=auto`), remote RViz from your laptop while
MoveIt runs on the robot, sending motion via the REST API or `ros2_control`
directly, health checks, and `./deploy_to_robot.sh` to push code you changed
locally onto the robot. Read **all of it**, in order, before sending your
first real motion command — it also documents the mechanical safety limits
(e.g. `openarm_left_joint1`'s real range is tighter than the URDF) and the
CAN bus footguns (never `sleep` between disabling torque and `set_zero`).

Short version of the path once you've built and read the docs above:

1. `ssh ubuntu@192.168.1.226` (or work from your own machine and target the
   robot over the network — see `terminal_command.md` step 0 for the DDS env
   vars either way).
2. Check CAN is up, calibrate if needed (`terminal_command.md` steps 1-2).
3. Bring the real stack up (step 3).
4. Send commands via the REST API (`curl .../api/move/joints`, etc.), the 3D
   dashboard, or `ros2_control` topics/services directly (step 5).

---

## Troubleshooting

- **`colcon build` fails on a missing dependency** → re-run
  `rosdep install --from-paths src --ignore-src -r -y`, check step 2's two
  manual exceptions.
- **`doxygen: command not found`** → `sudo apt install doxygen graphviz`.
- **Nothing moves / controllers not active** → `ros2 control list_controllers`
  and `ros2 control list_hardware_components` (see `terminal_command.md`
  step 6).
- **Working from a laptop against the real IQ9075** → DDS/RViz bridging
  gotchas are laptop-specific; ask a teammate for the current
  `ROS_DOMAIN_ID`/CycloneDDS setup for your machine before assuming a plain
  `ros2 launch` is isolated from the live robot (it defaults to sharing the
  same DDS domain — see `terminal_command.md` step 3's "C2" note).
