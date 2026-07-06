#!/usr/bin/env bash
# Resolve Isaac Lab / Isaac Sim paths on the training server (stdout: KEY=value lines).
set -euo pipefail

OPENARM_REMOTE_ROOT="${OPENARM_REMOTE_ROOT:-/data21tb/users/huyhoang/openarm_train_ws}"
REMOTE_RL="${OPENARM_REMOTE_ROOT}/Reinforce_Learning"

for d in \
    "${OPENARM_REMOTE_ROOT%/openarm_train_ws}/IsaacLab" \
    "/data21tb/users/huyhoang/IsaacLab" \
    "${HOME}/IsaacLab" \
    "${HOME}/Desktop/IsaacLab"; do
    if [ -f "${d}/isaaclab.sh" ]; then
        echo "ISAACLAB_SH=${d}/isaaclab.sh"
        echo "ISAACLAB_ROOT=${d}"
        break
    fi
    if [ -d "${d}/source/isaaclab" ]; then
        echo "ISAACLAB_ROOT=${d}"
    fi
done

for p in \
    "${OPENARM_REMOTE_ROOT%/openarm_train_ws}/isaacsim/python.sh" \
    "/data21tb/users/huyhoang/isaacsim/python.sh" \
    "/data21tb/huyhoang/isaacsim/python.sh" \
    "${HOME}/isaacsim/python.sh" \
    "${HOME}/Desktop/isaacsim/python.sh"; do
    if [ -f "$p" ]; then
        echo "ISAAC_PYTHON=${p}"
        break
    fi
done

echo "REMOTE_RL=${REMOTE_RL}"

# PYTHONPATH for broken editable installs (pip points to old Desktop path)
if [ -d "${OPENARM_REMOTE_ROOT%/openarm_train_ws}/IsaacLab/source/isaaclab" ]; then
    IL="${OPENARM_REMOTE_ROOT%/openarm_train_ws}/IsaacLab"
elif [ -d "/data21tb/users/huyhoang/IsaacLab/source/isaaclab" ]; then
    IL="/data21tb/users/huyhoang/IsaacLab"
else
    IL=""
fi
if [ -n "$IL" ]; then
    echo "ISAACLAB_PYTHONPATH=${REMOTE_RL}:${IL}/source/isaaclab:${IL}/source/isaaclab_rl:${IL}/source/isaaclab_assets:${IL}/source/isaaclab_tasks"
fi
