#!/bin/bash
# =============================================================================
# fetch_model.sh — Download trained policy from server → local logs/
#
# Usage:
#   ./fetch_model.sh [user@server] [run_name]
#
# Examples:
#   ./fetch_model.sh naiscorp-4090 train_osc              # Phase 1 reach
#   ./fetch_model.sh naiscorp-4090 train_osc_phase2    # Phase 2 reach+grasp+lift
#
# Saves:
#   logs/best_policy_<run_name>.pt   — named copy (keeps history)
#   logs/active_policy.pt            — symlink → use this for demo
# =============================================================================
set -euo pipefail

SERVER="${1:-naiscorp@naiscorp-4090}"
RUN_NAME="${2:-train_osc_phase2}"
OPENARM_REMOTE_ROOT="${OPENARM_REMOTE_ROOT:-/data21tb/users/huyhoang/openarm_train_ws}"
REMOTE_DIR="${OPENARM_REMOTE_ROOT}/logs_openarm/${RUN_NAME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_LOGS="${SCRIPT_DIR}/logs"
NAMED_PT="${LOCAL_LOGS}/best_policy_${RUN_NAME}.pt"
ACTIVE_PT="${LOCAL_LOGS}/active_policy.pt"
CKPT_DIR="${LOCAL_LOGS}/checkpoints_${RUN_NAME}"

echo "============================================================"
echo "  Fetch policy: ${RUN_NAME}"
echo "  Server : $SERVER:${REMOTE_DIR}"
echo "  Local  : ${NAMED_PT}"
echo "         → ${ACTIVE_PT} (symlink for demo)"
echo "============================================================"

mkdir -p "${CKPT_DIR}"
mkdir -p "${LOCAL_LOGS}/tensorboard"

IS_RUNNING=$(ssh "$SERVER" "pgrep -f 'isaaclab_train\.p[y]' > /dev/null 2>&1 && echo yes || echo no" </dev/null)
if [ "$IS_RUNNING" = "yes" ]; then
    echo "  ⚠️  Training still running — fetching latest checkpoint."
else
    echo "  ✅ Training not running."
fi

echo ""
echo "Downloading best_policy.pt..."
if scp "${SERVER}:${REMOTE_DIR}/best_policy.pt" "${NAMED_PT}"; then
    ln -sf "$(basename "${NAMED_PT}")" "${ACTIVE_PT}"
    echo "  ✅ ${NAMED_PT}"
    echo "  ✅ active_policy.pt → best_policy_${RUN_NAME}.pt"
else
    echo "  ❌ best_policy.pt not found on server for ${RUN_NAME}"
    exit 1
fi

echo ""
scp "${SERVER}:${REMOTE_DIR}/env_cfg.pkl" "${LOCAL_LOGS}/env_cfg_${RUN_NAME}.pkl" 2>/dev/null && \
    ln -sf "env_cfg_${RUN_NAME}.pkl" "${LOCAL_LOGS}/env_cfg.pkl" && \
    echo "  ✅ env_cfg.pkl" || echo "  ⚠️  env_cfg.pkl missing"

LATEST=$(ssh "$SERVER" "ls -t ${REMOTE_DIR}/checkpoints/*.zip 2>/dev/null | head -1" </dev/null)
if [ -n "$LATEST" ]; then
    scp "${SERVER}:${LATEST}" "${CKPT_DIR}/" && \
        echo "  ✅ latest ckpt → checkpoints_${RUN_NAME}/$(basename "$LATEST")"
fi

echo ""
echo "Run demo with THE FILE YOU JUST FETCHED:"
echo "  ./run_local.sh demo --model ./logs/active_policy.pt --phase 2"
echo ""
echo "  ⚠️  Do NOT use best_policy_osc.pt unless you fetched train_osc (Phase 1 only)."
echo "============================================================"
