#!/bin/bash
# List policy checkpoints on server and locally (avoid v2/v3 confusion).
#
# Usage:
#   ./list_models.sh [user@server]
#
set -euo pipefail

SERVER="${1:-naiscorp@naiscorp-4090}"
OPENARM_REMOTE_ROOT="${OPENARM_REMOTE_ROOT:-/data21tb/users/huyhoang/openarm_train_ws}"
REMOTE_ROOT="${OPENARM_REMOTE_ROOT}/logs_openarm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_LOGS="${SCRIPT_DIR}/logs"

echo "============================================================"
echo "  Policy inventory"
echo "============================================================"

echo ""
echo "── Server (${SERVER}) ──"
ssh "$SERVER" "bash -s" <<EOF
ROOT='${REMOTE_ROOT}'
for dir in "$ROOT"/*/; do
  [ -d "$dir" ] || continue
  name=$(basename "$dir")
  case "$name" in tensorboard) continue ;; esac
  best="$dir/best_policy.pt"
  if [ -f "$best" ]; then
    steps=$(ls -t "$dir/checkpoints"/rl_model_*_steps.zip 2>/dev/null | head -1 | sed 's/.*rl_model_\([0-9]*\)_steps.zip/\1/' || echo '?')
    sz=$(du -h "$best" | cut -f1)
    dt=$(date -r "$best" '+%Y-%m-%d %H:%M')
    md5=$(md5sum "$best" | cut -c1-8)
    echo "  $name  |  ~${steps} steps  |  $sz  |  $dt  |  md5:$md5"
  fi
done | sort
EOF

echo ""
echo "── Local (${LOCAL_LOGS}) ──"
for f in "${LOCAL_LOGS}"/best_policy*.pt "${LOCAL_LOGS}"/active_policy.pt; do
  [ -f "$f" ] || continue
  sz=$(du -h "$f" | cut -f1)
  dt=$(date -r "$f" '+%Y-%m-%d %H:%M')
  md5=$(md5sum "$f" | cut -c1-8)
  echo "  $(basename "$f")  |  $sz  |  $dt  |  md5:$md5"
done

if [ -L "${LOCAL_LOGS}/active_policy.pt" ]; then
  echo ""
  echo "  active_policy.pt → $(readlink -f "${LOCAL_LOGS}/active_policy.pt" 2>/dev/null || readlink "${LOCAL_LOGS}/active_policy.pt")"
fi

echo ""
echo "── Demo command (use active_policy or explicit path) ──"
echo "  ./run_local.sh demo --model ./logs/active_policy.pt --phase 2"
echo "============================================================"
