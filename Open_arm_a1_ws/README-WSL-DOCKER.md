# 🤖 OpenArm — WSL Docker Setup for RViz2 UI Design

> This Docker setup is for **UI designers on Windows/WSL2** who need to run
> RViz2 and MoveIt without installing Ubuntu natively.

## Architecture

```
┌──────────────────────────────┐     ┌──────────────────────────┐
│  Teammate's PC (Windows)     │     │  Hans's PC (Ubuntu 24.04)│
│  ┌────────────────────────┐  │     │  ROS 2 Jazzy (native)    │
│  │ WSL2                   │  │     │                          │
│  │  ┌──────────────────┐  │  │     │  deploy_to_robot.sh      │
│  │  │ Docker Container │  │  │     │          │               │
│  │  │  ROS 2 Jazzy     │  │  │     │          ▼               │
│  │  │  MoveIt 2        │  │  │     │  ┌──────────────────┐   │
│  │  │  RViz2 ◄─── GUI  │  │  │     │  │ Robot (IQ 9075)  │   │
│  │  └──────────────────┘  │  │     │  │ ubuntu@100.92... │   │
│  └────────────────────────┘  │     │  └──────────────────┘   │
└──────────────────────────────┘     └──────────────────────────┘
```

## Prerequisites

1. **Windows 11** (or Windows 10 build 19044+) with **WSL2** enabled
2. **Docker Desktop** for Windows with WSL2 backend enabled
3. **WSLg** working (comes with WSL2 by default — run `wsl --update` to ensure)

### Quick WSLg test

Open WSL terminal and run:
```bash
# Should open a small window with eyes following your mouse
sudo apt install x11-apps -y && xclock
```
If a clock window appears, GUI forwarding is working ✅

---

## Quick Start

### 1. Clone / copy the workspace into WSL

```bash
# From WSL terminal
cd ~
git clone <your-repo-url> Open_arm_a1_ws
cd Open_arm_a1_ws
```

### 2. Build and start the container

```bash
docker compose -f .docker/docker-compose.wsl.yml up -d --build
```

First build takes ~10-15 min (downloads ROS 2 Jazzy + MoveIt 2).  
Subsequent starts are instant.

### 3. Enter the container

```bash
docker exec -it openarm_wsl_ui bash
```

### 4. Build the workspace (first time only)

```bash
# Install any missing rosdep dependencies
sudo apt update
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --symlink-install

# Source
source install/setup.bash
```

### 5. Launch RViz2

```bash
# Option A: Full MoveIt + RViz demo
ros2 launch openarm_moveit_config moveit_bimanual.launch.py

# Option B: Just the robot description viewer
ros2 launch openarm_description display.launch.py

# Option C: Plain RViz2
rviz2
```

---

## Daily Workflow

After the first setup, your daily workflow is just:

```bash
# Start container (if stopped)
docker compose -f .docker/docker-compose.wsl.yml up -d

# Enter container
docker exec -it openarm_wsl_ui bash

# Source and work
source install/setup.bash
rviz2
```

Since `./src` is mounted into the container, any file edits you make in
**Windows (VSCode, etc.)** are instantly reflected inside the container.
After editing URDF/Xacro files, just rebuild:

```bash
colcon build --symlink-install && source install/setup.bash
```

---

## VSCode Integration

You can attach VSCode directly to the running container:

1. Install the **Dev Containers** extension in VSCode
2. Open command palette → `Dev Containers: Attach to Running Container...`
3. Select `openarm_wsl_ui`
4. You now have a full VSCode IDE inside the container with IntelliSense

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `cannot open display` | Run `wsl --update` in PowerShell, then restart WSL |
| RViz2 shows black screen | Set `LIBGL_ALWAYS_SOFTWARE=1` in the container |
| `DRM device` / Session error | Added `/dev/dri` to compose; if it persists, use `LIBGL_ALWAYS_SOFTWARE=1` |
| `rosdep: command not found` | Run `sudo apt install python3-rosdep` inside container |
| Build fails on hardware pkgs | Those packages are robot-only; use `colcon build --packages-select openarm_description openarm_moveit_config moveit_api` |

---

## Stopping / Cleaning Up

```bash
# Stop container
docker compose -f .docker/docker-compose.wsl.yml down

# Full cleanup (removes image too)
docker compose -f .docker/docker-compose.wsl.yml down --rmi all
```