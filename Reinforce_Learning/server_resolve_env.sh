#!/usr/bin/env bash
# Resolve Isaac Lab / Isaac Sim paths ON THE SERVER (stdout: KEY=value).
# Gọi từ ./rl.sh train qua SSH — không chạy trên laptop.
set -euo pipefail

OPENARM_DATA_ROOT="${OPENARM_DATA_ROOT:-/data21tb/users}"
OPENARM_WS_NAME="${OPENARM_WS_NAME:-openarm_train_ws}"
OPENARM_REMOTE_ROOT="${OPENARM_REMOTE_ROOT:-${OPENARM_DATA_ROOT}/$(whoami)/${OPENARM_WS_NAME}}"
REMOTE_RL="${OPENARM_REMOTE_ROOT}/Reinforce_Learning"

for d in \
    "${OPENARM_REMOTE_ROOT%/openarm_train_ws}/IsaacLab" \
    "${OPENARM_DATA_ROOT}/$(whoami)/IsaacLab" \
    "/data21tb/$(whoami)/IsaacLab" \
    "${HOME}/IsaacLab" \
    "${HOME}/Desktop/IsaacLab" \
    "${OPENARM_DATA_ROOT}/huyhoang/IsaacLab"; do
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
    "${OPENARM_DATA_ROOT}/$(whoami)/isaacsim/python.sh" \
    "/data21tb/$(whoami)/isaacsim/python.sh" \
    "${HOME}/isaacsim/python.sh" \
    "${HOME}/Desktop/isaacsim/python.sh" \
    "${OPENARM_DATA_ROOT}/huyhoang/isaacsim/python.sh"; do
    if [ -f "$p" ]; then
        echo "ISAAC_PYTHON=${p}"
        break
    fi
done

echo "REMOTE_RL=${REMOTE_RL}"

if [ -d "${OPENARM_REMOTE_ROOT%/openarm_train_ws}/IsaacLab/source/isaaclab" ]; then
    IL="${OPENARM_REMOTE_ROOT%/openarm_train_ws}/IsaacLab"
elif [ -d "${OPENARM_DATA_ROOT}/$(whoami)/IsaacLab/source/isaaclab" ]; then
    IL="${OPENARM_DATA_ROOT}/$(whoami)/IsaacLab"
elif [ -d "${OPENARM_DATA_ROOT}/huyhoang/IsaacLab/source/isaaclab" ]; then
    IL="${OPENARM_DATA_ROOT}/huyhoang/IsaacLab"
else
    IL=""
fi
if [ -n "$IL" ]; then
    echo "ISAACLAB_PYTHONPATH=${REMOTE_RL}:${IL}/source/isaaclab:${IL}/source/isaaclab_rl:${IL}/source/isaaclab_assets:${IL}/source/isaaclab_tasks"
fi
