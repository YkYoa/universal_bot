#!/bin/bash
# =============================================================================
# fetch_model.sh
# Download the trained Isaac Lab model from the server to your local machine,
# then optionally view results in TensorBoard locally.
#
# Usage:
#   ./fetch_model.sh [user@server_ip]
#
# Example:
#   ./fetch_model.sh naiscorp-4090
#
# Downloads to: Reinforce_Learning/logs/
# =============================================================================
set -e

SERVER="${1:-naiscorp-4090}"
SERVER_USER="$(echo "$SERVER" | cut -d@ -f1)"
SERVER_IP="$(echo "$SERVER" | cut -d@ -f2)"
REMOTE_DIR="/data21tb/huyhoang/openarm_train_ws/logs_openarm/train"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_LOGS="${SCRIPT_DIR}/logs"

echo "============================================================"
echo "  Fetching Isaac Lab model from server (Manager-Based RL)"
echo "  Server : $SERVER:${REMOTE_DIR}"
echo "  Local  : ${LOCAL_LOGS}"
echo "============================================================"

mkdir -p "${LOCAL_LOGS}/checkpoints"
mkdir -p "${LOCAL_LOGS}/tensorboard"

# ── Check if training is still running ───────────────────────────────────────
echo ""
echo "Checking training status on server..."
IS_RUNNING=$(ssh "$SERVER" "pgrep -f 'isaaclab_train\.p[y]' > /dev/null 2>&1 && echo yes || echo no" </dev/null)
if [ "$IS_RUNNING" = "yes" ]; then
    echo "  ⚠️  Training is still running on the server."
    echo "  You can still fetch intermediate checkpoints below."
else
    echo "  ✅ Training has completed (or was stopped)."
fi

# ── Download best policy ────────────────────────────────────────────────────
echo ""
echo "Downloading best policy checkpoint..."
scp "${SERVER}:${REMOTE_DIR}/best_policy.pt" "${LOCAL_LOGS}/best_policy.pt" && \
    echo "  ✅ best_policy.pt" || echo "  ⚠️  best_policy.pt not found yet"

# ── Download final model ──────────────────────────────────────────────────────
echo ""
echo "Downloading final model checkpoint..."
scp "${SERVER}:${REMOTE_DIR}/final_policy.pt" "${LOCAL_LOGS}/final_policy.pt" && \
    echo "  ✅ final_policy.pt" || echo "  ⚠️  final_policy.pt not found yet"

# ── Download env config (needed to reconstruct env for demo) ─────────────────
echo ""
echo "Downloading environment config..."
scp "${SERVER}:${REMOTE_DIR}/env_cfg.pkl" "${LOCAL_LOGS}/env_cfg.pkl" && \
    echo "  ✅ env_cfg.pkl" || echo "  ⚠️  env_cfg.pkl not found"

# ── Download latest checkpoint (for resume) ───────────────────────────────────
echo ""
echo "Downloading latest checkpoint..."
LATEST=$(ssh "$SERVER" "ls -t ${REMOTE_DIR}/checkpoints/*.zip 2>/dev/null | head -1" </dev/null)
if [ -n "$LATEST" ]; then
    scp "${SERVER}:${LATEST}" "${LOCAL_LOGS}/checkpoints/" && \
        echo "  ✅ $(basename "$LATEST")" || echo "  ⚠️  Failed to download latest checkpoint"
else
    echo "  ⚠️  No checkpoints found yet"
fi

# ── Sync TensorBoard logs ──────────────────────────────────────────────────────
echo ""
echo "Syncing TensorBoard logs..."
rsync -avz --progress \
    "${SERVER}:/data21tb/huyhoang/openarm_train_ws/logs_openarm/tensorboard/" \
    "${LOCAL_LOGS}/tensorboard/" && \
    echo "  ✅ TensorBoard logs synced" || echo "  ⚠️  TensorBoard logs not found yet"

# ── Offer to open TensorBoard locally ─────────────────────────────────────────
echo ""
if [ -d "${LOCAL_LOGS}/tensorboard" ] && [ "$(ls -A "${LOCAL_LOGS}/tensorboard" 2>/dev/null)" ]; then
    read -rp "Launch TensorBoard locally now? [Y/n]: " launch_tb || true
    if [[ ! "$launch_tb" =~ ^[Nn]$ ]]; then
        echo ""
        echo "  🚀 Launching TensorBoard at http://localhost:6007 ..."
        echo "     (Press Ctrl+C to stop)"
        echo ""
        python3 -m tensorboard.main \
            --logdir "${LOCAL_LOGS}/tensorboard" \
            --host 127.0.0.1 \
            --port 6007 \
            --reload_interval 5 2>/dev/null || \
        python3 -m pip install tensorboard -q && \
        python3 -m tensorboard.main \
            --logdir "${LOCAL_LOGS}/tensorboard" \
            --host 127.0.0.1 \
            --port 6007 \
            --reload_interval 5
    fi
fi

echo ""
echo "============================================================"
echo "  ✅ Done! Files saved to:"
echo "     ${LOCAL_LOGS}/"
echo ""
echo "  📁 Files:"
ls "${LOCAL_LOGS}/"*.{pt,pkl} 2>/dev/null | while read f; do
    echo "     $(basename "$f")  ($(du -sh "$f" | cut -f1))"
done || echo "     (no model files yet)"
echo ""
echo "  ▶ Run demo on your laptop with Isaac Sim:"
echo "     cd ${SCRIPT_DIR}/"
echo "     /path/to/isaac-sim-standalone/python.sh isaaclab_demo.py \\"
echo "         --model_path ./logs/best_policy.pt"
echo ""
echo "  📊 View TensorBoard locally:"
echo "     tensorboard --logdir ${LOCAL_LOGS}/tensorboard --port 6007"
echo "     → open http://localhost:6007"
echo "============================================================"
