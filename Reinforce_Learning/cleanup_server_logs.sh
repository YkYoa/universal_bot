#!/bin/bash
# Prune old training artifacts on the 4090 server (keeps latest run per directory).
#
# Usage:
#   ./cleanup_server_logs.sh [user@server]
#   ./cleanup_server_logs.sh naiscorp@naiscorp-4090 --dry-run
#
set -euo pipefail

SERVER="${1:-naiscorp@naiscorp-4090}"
DRY_RUN="${2:-}"
OPENARM_REMOTE_ROOT="${OPENARM_REMOTE_ROOT:-/data21tb/users/huyhoang/openarm_train_ws}"
REMOTE_ROOT="${OPENARM_REMOTE_ROOT}/logs_openarm"
KEEP_CKPT="${KEEP_CKPT:-5}"
KEEP_TB="${KEEP_TB:-2}"

REMOTE_SCRIPT="
REMOTE_ROOT='$REMOTE_ROOT'
KEEP_CKPT=$KEEP_CKPT
KEEP_TB=$KEEP_TB
DRY_RUN='$DRY_RUN'

run() {
  if [ \"\$DRY_RUN\" = '--dry-run' ]; then
    echo \"[dry-run] \$*\"
  else
    eval \"\$@\"
  fi
}

echo '=== Disk usage (before) ==='
du -sh \"\$REMOTE_ROOT\"/* 2>/dev/null | sort -hr || true

for ckpt_dir in \"\$REMOTE_ROOT\"/*/checkpoints; do
  [ -d \"\$ckpt_dir\" ] || continue
  mapfile -t files < <(ls -1 \"\$ckpt_dir\"/rl_model_*_steps.zip 2>/dev/null | sort -t_ -k2 -n)
  n=\${#files[@]}
  if [ \"\$n\" -le \"\$KEEP_CKPT\" ]; then
    continue
  fi
  del_count=\$((n - KEEP_CKPT))
  echo \"Prune \$ckpt_dir: \$n -> \$KEEP_CKPT (delete \$del_count)\"
  for ((i=0; i<del_count; i++)); do
    run rm -f \"\${files[i]}\"
  done
done

TB=\"\$REMOTE_ROOT/tensorboard\"
if [ -d \"\$TB\" ]; then
  mapfile -t ppo < <(ls -1d \"\$TB\"/PPO_* 2>/dev/null | sort -t_ -k2 -n)
  np=\${#ppo[@]}
  if [ \"\$np\" -gt \"\$KEEP_TB\" ]; then
    del=\$((np - KEEP_TB))
    echo \"Prune tensorboard: \$np -> \$KEEP_TB\"
    for ((i=0; i<del; i++)); do
      run rm -rf \"\${ppo[i]}\"
    done
  fi
fi

echo
echo '=== Counts (after) ==='
echo -n 'checkpoints: '
find \"\$REMOTE_ROOT\" -name 'rl_model_*_steps.zip' 2>/dev/null | wc -l
echo -n 'tb runs: '
ls -1d \"\$REMOTE_ROOT/tensorboard\"/PPO_* 2>/dev/null | wc -l
echo
echo '=== Disk usage (after) ==='
du -sh \"\$REMOTE_ROOT\"/* 2>/dev/null | sort -hr || true
"

ssh "$SERVER" "bash -s" <<< "$REMOTE_SCRIPT"
