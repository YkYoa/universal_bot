#!/bin/bash
# =============================================================================
# deploy_training.sh
# Deploy Isaac Lab training code (Manager-based RL) to the RTX 4090 server.
#
# Usage:
#   ./deploy_training.sh [user@server_ip] [python_path]
#
# Example:
#   ./deploy_training.sh naiscorp@192.168.1.122
# =============================================================================
set -e

SERVER="${1:-naiscorp-4090}"
SERVER_USER="$(echo "$SERVER" | cut -d@ -f1)"
SERVER_IP="$(echo "$SERVER" | cut -d@ -f2)"
REMOTE_DIR="/data21tb/huyhoang/openarm_train_ws"

# Combine remote operations (probing path, creating directories, killing old sessions, starting TensorBoard)
echo "Initializing server directories, processes, and path resolution..."
INIT_SCRIPT="
SERVER_ISAAC_PYTHON='$2'
if [ -z \"\$SERVER_ISAAC_PYTHON\" ]; then
    for path in '/data21tb/huyhoang/isaacsim/python.sh' '/home/${SERVER_USER}/Desktop/isaacsim/python.sh' '/home/${SERVER_USER}/isaacsim/python.sh'; do
        if [ -f \"\$path\" ]; then
            SERVER_ISAAC_PYTHON=\"\$path\"
            break
        fi
    done
fi

if [ -z \"\$SERVER_ISAAC_PYTHON\" ] || [ ! -f \"\$SERVER_ISAAC_PYTHON\" ]; then
    echo \"ERROR_PYTHON_NOT_FOUND:\$SERVER_ISAAC_PYTHON\"
    exit 1
fi

mkdir -p '${REMOTE_DIR}/Reinforce_Learning' \
         '${REMOTE_DIR}/Open_arm_a1_ws/src/openarm_description/urdf/robot/v10' \
         '${REMOTE_DIR}/logs_openarm/train' \
         '${REMOTE_DIR}/logs_openarm/tensorboard'

# Safe process killing (grep -v \$\$ prevents killing this remote ssh shell)
pgrep -u \$(whoami) -f 'isaaclab_train.py' | grep -v \"\$\$\" | xargs -r kill 2>/dev/null || true
pgrep -u \$(whoami) -f 'tensorboard.main' | grep -v \"\$\$\" | xargs -r kill 2>/dev/null || true

nohup python3 -m tensorboard.main \
    --logdir '${REMOTE_DIR}/logs_openarm/tensorboard' \
    --host 0.0.0.0 \
    --port 6008 \
    --reload_interval 5 \
    </dev/null >'${REMOTE_DIR}/logs_openarm/tensorboard.log' 2>&1 &
TBOARD_PID=\$!
disown \$TBOARD_PID

echo \"RESOLVED_PYTHON:\$SERVER_ISAAC_PYTHON\"
echo \"TBOARD_PID:\$TBOARD_PID\"
"

# Run initialization in a single SSH connection
SERVER_OUT=$(ssh "$SERVER" "$INIT_SCRIPT" </dev/null)

if echo "$SERVER_OUT" | grep -q "ERROR_PYTHON_NOT_FOUND"; then
    echo "  ❌ Isaac Sim python not found on server."
    echo "  Please set the correct path as the 2nd argument:"
    echo "     ./deploy_training.sh naiscorp@SERVER_IP /path/to/python.sh"
    exit 1
fi

ISAAC_PYTHON=$(echo "$SERVER_OUT" | grep "RESOLVED_PYTHON:" | cut -d: -f2-)
TBOARD_PID=$(echo "$SERVER_OUT" | grep "TBOARD_PID:" | cut -d: -f2-)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$SCRIPT_DIR"

echo "============================================================"
echo "  OpenArm Manager-Based RL Training — Server Ready"
echo "  Server       : $SERVER"
echo "  Remote dir   : $REMOTE_DIR"
echo "  Isaac python : $ISAAC_PYTHON"
echo "  TensorBoard  : http://${SERVER_IP}:6008 (PID: $TBOARD_PID)"
echo "============================================================"

# ── 2. Sync Isaac Lab training source ─────────────────────────────────────────
echo ""
echo "[1/2] Syncing Reinforce_Learning source code..."
rsync -avz --progress --delete \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'logs/' \
    "${WS_DIR}/" \
    "${SERVER}:${REMOTE_DIR}/Reinforce_Learning/"

# ── 3. Sync robot USD ─────────────────────────────────────────────────────────
echo ""
echo "[2/2] Syncing OpenArm A1 v10 USD robot asset..."
rsync -avz --progress \
    --include="*/" \
    --include="*.usd" \
    --include="*.usda" \
    --exclude="*" \
    "${WS_DIR}/../Open_arm_a1_ws/src/openarm_description/urdf/robot/v10/" \
    "${SERVER}:${REMOTE_DIR}/Open_arm_a1_ws/src/openarm_description/urdf/robot/v10/"

# ── 4. Print Launch Instructions ─────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  🚀 Training code deployed and synchronized!"
echo ""
echo "  📊 TensorBoard : http://${SERVER_IP}:6008"
echo "============================================================"
echo ""
echo "  ▶ To run training in the FOREGROUND (interactive):"
echo "     ssh -t $SERVER \\"
echo "         \"${ISAAC_PYTHON} ${REMOTE_DIR}/Reinforce_Learning/isaaclab_train.py \\"
echo "         --headless --log_dir ${REMOTE_DIR}/logs_openarm/train --progress\""
echo ""
echo "  ▶ To run training in the BACKGROUND (headless):"
echo "     ssh $SERVER \\"
echo "         \"nohup ${ISAAC_PYTHON} ${REMOTE_DIR}/Reinforce_Learning/isaaclab_train.py \\"
echo "         --headless --log_dir ${REMOTE_DIR}/logs_openarm/train \\"
echo "         > ${REMOTE_DIR}/logs_openarm/train.log 2>&1 &\""
echo ""
echo "  ▶ To RESUME training in the background:"
echo "     ssh $SERVER \\"
echo "         \"nohup ${ISAAC_PYTHON} ${REMOTE_DIR}/Reinforce_Learning/isaaclab_train.py \\"
echo "         --headless --resume --log_dir ${REMOTE_DIR}/logs_openarm/train \\"
echo "         > ${REMOTE_DIR}/logs_openarm/train_resume.log 2>&1 &\""
echo ""
echo "  📄 Monitor background logs:"
echo "     ssh $SERVER \"tail -f ${REMOTE_DIR}/logs_openarm/train.log\""
echo ""
echo "  🛑 Stop background runs:"
echo "     ssh $SERVER \"pgrep -u \\\$(whoami) -f 'isaaclab_train.py' | grep -v '\\\$\\\$' | xargs -r kill\""
echo ""
echo "  📥 Fetch the trained model:"
echo "     ./fetch_model.sh $SERVER"
echo "============================================================"
