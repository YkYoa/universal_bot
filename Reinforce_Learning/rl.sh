#!/usr/bin/env bash
# OpenArm RL — single entry point (deploy / train / fetch / demo / …)
#
#   cp openarm.env.example openarm.env
#   ./rl.sh deploy
#   ./rl.sh train train_osc_phase2_v3 --task_phase 2 --assist-schedule --descent-assist --stage all \
#       --num-envs 1024 --timesteps 1000000 --checkpoint logs/best_policy_train_osc_phase2_assist9.pt
#   ./rl.sh train-fg test_run --num-envs 16 --timesteps 2000 --progress
#   ./rl.sh fetch train_osc_phase2
#   ./rl.sh demo --model ./logs/active_policy.pt --phase 2 --stage all
#
set -euo pipefail

RL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RL_ROOT"

# ── Config & paths ───────────────────────────────────────────────────────────
OPENARM_DATA_ROOT="${OPENARM_DATA_ROOT:-/data21tb/users}"
OPENARM_SERVER="${OPENARM_SERVER:-huyhoang-4090}"
OPENARM_WS_NAME="${OPENARM_WS_NAME:-openarm_train_ws}"

load_openarm_env() {
    OPENARM_RL_ROOT="$RL_ROOT"
    OPENARM_LOCAL_LOGS="${OPENARM_LOCAL_LOGS:-${RL_ROOT}/logs}"
    if [ -f "${RL_ROOT}/openarm.env" ]; then
        # shellcheck disable=SC1091
        source "${RL_ROOT}/openarm.env"
    fi
    OPENARM_LOCAL_LOGS="${OPENARM_LOCAL_LOGS:-${RL_ROOT}/logs}"
    mkdir -p "${OPENARM_LOCAL_LOGS}"
    ISAAC_SIM_PYTHON="${ISAAC_SIM_PYTHON:-/home/hans/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh}"
}

openarm_resolve_server() {
    SERVER="${1:-${OPENARM_SERVER}}"
    OPENARM_SERVER="$SERVER"
    REMOTE_USER="$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$SERVER" 'whoami' 2>/dev/null || true)"
    if [ -z "$REMOTE_USER" ]; then
        echo "  ❌ Không SSH được tới ${SERVER}. Kiểm tra openarm.env (OPENARM_SERVER)." >&2
        exit 1
    fi
    OPENARM_REMOTE_ROOT="${OPENARM_REMOTE_ROOT:-${OPENARM_DATA_ROOT}/${REMOTE_USER}/${OPENARM_WS_NAME}}"
    REMOTE_LOGS="${OPENARM_REMOTE_ROOT}/logs_openarm"
    REMOTE_RL="${OPENARM_REMOTE_ROOT}/Reinforce_Learning"
    REMOTE_USD="${OPENARM_REMOTE_ROOT}/Open_arm_a1_ws/src/openarm_description/urdf/robot/v10"
    SERVER_IP="$(ssh -G "$SERVER" 2>/dev/null | awk '$1=="hostname" {print $2; exit}')"
    SERVER_IP="${SERVER_IP:-$(echo "$SERVER" | cut -d@ -f2)}"
    SERVER_IP="${SERVER_IP:-$SERVER}"
}

openarm_print_paths() {
    local tb_port="${OPENARM_TB_PORT:-6008}"
    cat <<EOF
  Server       : ${SERVER} (${REMOTE_USER})
  Remote root  : ${OPENARM_REMOTE_ROOT}
  Remote logs  : ${REMOTE_LOGS}
  Local logs   : ${OPENARM_LOCAL_LOGS}
  TensorBoard  : http://${SERVER_IP}:${tb_port}
EOF
}

usage() {
    cat <<'EOF'
OpenArm RL

  ./rl.sh deploy [server] [python.sh]   Sync code lên server
  ./rl.sh train <run> [args]            Train server (background, mặc định 1024 env / 1M steps)
  ./rl.sh train-fg <run> [args]         Train server (foreground)
  ./rl.sh fetch [run]                   Tải model về logs/ (default: train_osc_phase2)
  ./rl.sh list                          Liệt kê checkpoint
  ./rl.sh cleanup [--dry-run]           Dọn log server
  ./rl.sh demo [args]                   Demo local (4050)
  ./rl.sh local-train [args]            Train local backup

EOF
}

# ── Server commands ────────────────────────────────────────────────────────────
cmd_deploy() {
    if [ -n "${1:-}" ] && [[ "$1" != *.sh ]]; then OPENARM_SERVER="$1"; shift; fi
    local manual_python="${OPENARM_SERVER_PYTHON:-${1:-}}"
    openarm_resolve_server

    local init_script
    init_script="
MANUAL_PYTHON='${manual_python}'
REMOTE_USER=\$(whoami)
SERVER_ISAAC_PYTHON=\"\$MANUAL_PYTHON\"
if [ -z \"\$SERVER_ISAAC_PYTHON\" ]; then
    for path in \
        \"/data21tb/users/\${REMOTE_USER}/isaacsim/python.sh\" \
        \"/data21tb/\${REMOTE_USER}/isaacsim/python.sh\" \
        \"/home/\${REMOTE_USER}/isaacsim/python.sh\" \
        \"/data21tb/users/huyhoang/isaacsim/python.sh\"; do
        [ -f \"\$path\" ] && SERVER_ISAAC_PYTHON=\"\$path\" && break
    done
fi
[ -f \"\$SERVER_ISAAC_PYTHON\" ] || { echo ERROR_PYTHON_NOT_FOUND; exit 1; }
mkdir -p '${OPENARM_REMOTE_ROOT}/Reinforce_Learning' \
         '${OPENARM_REMOTE_ROOT}/logs_openarm/tensorboard' \
         '${OPENARM_REMOTE_ROOT}/.cache/isaaclab_tmp' \
         '${REMOTE_USD}'
pgrep -u \$(whoami) -f 'isaaclab_train.py' | grep -v \"\$\$\" | xargs -r kill 2>/dev/null || true
pgrep -u \$(whoami) -f 'tensorboard.main' | grep -v \"\$\$\" | xargs -r kill 2>/dev/null || true
TB_LOG='${OPENARM_REMOTE_ROOT}/logs_openarm/tensorboard'
TB_PORT=\"\${OPENARM_TB_PORT:-6008}\"
port_in_use() { ss -tln 2>/dev/null | grep -q \":\${1} \"; }
if port_in_use \"\$TB_PORT\"; then
  # Port 6008 thường bị TB user khác (naiscorp) — không đọc được log huyhoang
  for p in 6009 6010 6011; do port_in_use \"\$p\" || { TB_PORT=\$p; break; }; done
fi
nohup python3 -m tensorboard.main --logdir \"\$TB_LOG\" \
    --host 0.0.0.0 --port \"\$TB_PORT\" --reload_interval 5 \
    >'${OPENARM_REMOTE_ROOT}/logs_openarm/tensorboard.log' 2>&1 &
echo \"\$TB_PORT\" >'${OPENARM_REMOTE_ROOT}/logs_openarm/tensorboard.port'
echo RESOLVED_PYTHON:\$SERVER_ISAAC_PYTHON
echo RESOLVED_TB_PORT:\$TB_PORT
"
    local out
    out=$(ssh "$SERVER" "$init_script" </dev/null)
    if echo "$out" | grep -q ERROR_PYTHON_NOT_FOUND; then
        echo "  ❌ Không tìm thấy isaacsim/python.sh trên server."
        echo "     Thêm OPENARM_SERVER_PYTHON vào openarm.env"
        exit 1
    fi
    local isaac_python tb_port
    isaac_python=$(echo "$out" | grep RESOLVED_PYTHON: | cut -d: -f2-)
    tb_port=$(echo "$out" | grep RESOLVED_TB_PORT: | cut -d: -f2-)
    tb_port="${tb_port:-6008}"
    OPENARM_TB_PORT="$tb_port"

    echo "============================================================"
    echo "  Deploy"
    openarm_print_paths
    echo "  Isaac python : ${isaac_python}"
    echo "============================================================"

    echo "[1/2] Sync Reinforce_Learning..."
    rsync -avz --delete \
        --exclude '__pycache__/' --exclude '*.pyc' --exclude 'logs/' --exclude 'openarm.env' \
        "${RL_ROOT}/" "${SERVER}:${REMOTE_RL}/"

    echo "[2/2] Sync robot USD..."
    rsync -avz \
        --include='*/' --include='*.usd' --include='*.usda' --exclude='*' \
        "${RL_ROOT}/../Open_arm_a1_ws/src/openarm_description/urdf/robot/v10/" \
        "${SERVER}:${REMOTE_USD}/"

    if [ "${OPENARM_SYNC_ISAACLAB:-0}" = "1" ]; then
        local local_il="${OPENARM_LOCAL_ISAACLAB:-$HOME/IsaacLab}"
        local remote_il="${OPENARM_REMOTE_ROOT%/openarm_train_ws}/IsaacLab"
        rsync -avz --exclude '.git/' --exclude 'docs/' --exclude '**/logs/' \
            "${local_il}/" "${SERVER}:${remote_il}/"
        ssh "$SERVER" "ln -sfn '$(dirname "${isaac_python}")' '${remote_il}/_isaac_sim'"
    fi
    echo "  ✅ Deploy xong → ./rl.sh train-fg test_run --num-envs 16 --timesteps 2000 --progress"
}

cmd_train() {
    local fg="${1:-0}"; shift
    local run_name="${1:?thiếu run_name}"; shift
    openarm_resolve_server
    local log_file="${REMOTE_LOGS}/${run_name}.log"
    local ssh_t=()
    [ "$fg" = "1" ] && ssh_t=(-t)

    echo "============================================================"
    echo "  Train: ${run_name} ($([ "$fg" = "1" ] && echo foreground || echo background))"
    openarm_print_paths
    echo "  Log: ${log_file}"
    echo "============================================================"

    ssh "${ssh_t[@]}" "$SERVER" \
        OPENARM_REMOTE_ROOT="$OPENARM_REMOTE_ROOT" \
        RUN_NAME="$run_name" LOG_FILE="$log_file" OPENARM_TRAIN_FG="$fg" \
        bash -s -- "$@" <<'REMOTE'
set -euo pipefail
export TMPDIR="${OPENARM_REMOTE_ROOT}/.cache/isaaclab_tmp"
mkdir -p "$TMPDIR" "${OPENARM_REMOTE_ROOT}/logs_openarm/${RUN_NAME}"
eval "$(bash "${OPENARM_REMOTE_ROOT}/Reinforce_Learning/server_resolve_env.sh")"
[ -n "${ISAAC_PYTHON:-}" ] && [ -f "${ISAAC_PYTHON}" ] || { echo "Chạy ./rl.sh deploy trước"; exit 1; }
[ -n "${ISAACLAB_PYTHONPATH:-}" ] && export PYTHONPATH="${ISAACLAB_PYTHONPATH}:${PYTHONPATH:-}"
CMD=("${ISAAC_PYTHON}" "${OPENARM_REMOTE_ROOT}/Reinforce_Learning/isaaclab_train.py" \
     --headless --log_dir "${OPENARM_REMOTE_ROOT}/logs_openarm/${RUN_NAME}" "$@")
if [ "${OPENARM_TRAIN_FG}" = "1" ]; then
    exec "${CMD[@]}"
else
    nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
    echo "PID: $!"
fi
REMOTE
    [ "$fg" = "1" ] || echo "  tail -f: ssh ${SERVER} \"tail -f ${log_file}\""
}

cmd_fetch() {
    local run_name="${1:-train_osc_phase2}"
    openarm_resolve_server
    local remote="${REMOTE_LOGS}/${run_name}"
    local named="${OPENARM_LOCAL_LOGS}/best_policy_${run_name}.pt"
    local active="${OPENARM_LOCAL_LOGS}/active_policy.pt"

    echo "============================================================"
    echo "  Fetch: ${run_name}"
    openarm_print_paths
    echo "============================================================"

    mkdir -p "${OPENARM_LOCAL_LOGS}/checkpoints_${run_name}"
    scp "${SERVER}:${remote}/best_policy.pt" "${named}"
    ln -sf "$(basename "${named}")" "${active}"
    echo "  ✅ ${named} → active_policy.pt"
    scp "${SERVER}:${remote}/env_cfg.pkl" "${OPENARM_LOCAL_LOGS}/env_cfg_${run_name}.pkl" 2>/dev/null && \
        ln -sf "env_cfg_${run_name}.pkl" "${OPENARM_LOCAL_LOGS}/env_cfg.pkl" || true
    local latest
    latest=$(ssh "$SERVER" "ls -t ${remote}/checkpoints/*.zip 2>/dev/null | head -1" || true)
    [ -n "$latest" ] && scp "${SERVER}:${latest}" "${OPENARM_LOCAL_LOGS}/checkpoints_${run_name}/" 2>/dev/null || true
    echo "  Demo: ./rl.sh demo --model ./logs/active_policy.pt --phase 2 --stage all"
}

cmd_list() {
    openarm_resolve_server
    echo "============================================================"
    echo "  Policy inventory"
    openarm_print_paths
    echo "── Server ──"
    ssh "$SERVER" "bash -s" <<EOF
ROOT='${REMOTE_LOGS}'
for dir in "\$ROOT"/*/; do
  [ -d "\$dir" ] || continue
  name=\$(basename "\$dir"); [ "\$name" = tensorboard ] && continue
  best="\$dir/best_policy.pt"
  [ -f "\$best" ] || continue
  steps=\$(ls -t "\$dir/checkpoints"/rl_model_*_steps.zip 2>/dev/null | head -1 | sed 's/.*rl_model_\\([0-9]*\\)_steps.zip/\\1/' || echo '?')
  echo "  \$name | ~\${steps} steps | \$(du -h "\$best" | cut -f1) | \$(date -r "\$best" '+%Y-%m-%d %H:%M')"
done | sort
EOF
    echo "── Local (${OPENARM_LOCAL_LOGS}) ──"
    for f in "${OPENARM_LOCAL_LOGS}"/best_policy*.pt; do
        [ -f "$f" ] && echo "  $(basename "$f") | $(du -h "$f" | cut -f1)"
    done
}

cmd_cleanup() {
    local dry="${1:-}"
    openarm_resolve_server
    local keep_ckpt="${KEEP_CKPT:-5}" keep_tb="${KEEP_TB:-2}"
    ssh "$SERVER" bash -s <<EOF
ROOT='${REMOTE_LOGS}'; KEEP_CKPT=${keep_ckpt}; KEEP_TB=${keep_tb}
for dir in "\$ROOT"/*/; do
  name=\$(basename "\$dir")
  if [ "\$name" = tensorboard ]; then
    ls -1dt "\$dir"/PPO_* 2>/dev/null | tail -n +\$((KEEP_TB+1)) | while read -r o; do
      [ '${dry}' = --dry-run ] && echo "[dry-run] rm -rf \$o" || rm -rf "\$o"
    done
  elif [ -d "\$dir/checkpoints" ]; then
    ls -1t "\$dir/checkpoints"/*.zip 2>/dev/null | tail -n +\$((KEEP_CKPT+1)) | while read -r o; do
      [ '${dry}' = --dry-run ] && echo "[dry-run] rm -f \$o" || rm -f "\$o"
    done
  fi
done
EOF
}

# ── Local commands ─────────────────────────────────────────────────────────────
cmd_local_train() {
    local envs=16 steps=1000000 headless="--headless" resume="" progress=""
    local log_dir="./logs/train" task_phase="" stage="" assist="" checkpoint=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --envs) envs=$2; shift 2 ;;
            --steps) steps=$2; shift 2 ;;
            --gui) headless=""; shift ;;
            --resume) resume="--resume"; shift ;;
            --progress) progress="--progress"; shift ;;
            --phase) task_phase="--task_phase $2"; shift 2 ;;
            --stage) stage="--stage $2"; shift 2 ;;
            --assist-schedule) assist="--assist-schedule"; shift ;;
            --checkpoint) checkpoint="--checkpoint $2"; shift 2 ;;
            *) echo "Option không rõ: $1"; exit 1 ;;
        esac
    done

    PYTHONPATH="$RL_ROOT" "$ISAAC_SIM_PYTHON" ./isaaclab_train.py \
        --num_envs "$envs" --timesteps "$steps" --log_dir "$log_dir" \
        $headless $resume $progress $task_phase $stage $assist $checkpoint
}

cmd_demo() {
    local model="./logs/active_policy.pt" envs=1 stage="" extra=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --model) model=$2; shift 2 ;;
            --envs) envs=$2; shift 2 ;;
            --phase) extra+=(--task_phase "$2"); shift 2 ;;
            --stage) stage=$2; shift 2 ;;
            --assist) stage=all; shift ;;
            *) extra+=("$1"); shift ;;
        esac
    done
    [ -n "$stage" ] && extra+=(--stage "$stage")
    PYTHONPATH="$RL_ROOT" "$ISAAC_SIM_PYTHON" ./isaaclab_demo.py \
        --model_path "$model" --num_envs "$envs" --no-bottle-rand "${extra[@]}"
}

# ── Main ───────────────────────────────────────────────────────────────────────
load_openarm_env
case "${1:-help}" in
    deploy)       shift; cmd_deploy "$@" ;;
    train)        shift; cmd_train 0 "$@" ;;
    train-fg)     shift; cmd_train 1 "$@" ;;
    fetch)        shift; cmd_fetch "$@" ;;
    list)         cmd_list ;;
    cleanup)      shift; cmd_cleanup "$@" ;;
    demo)         shift; cmd_demo "$@" ;;
    local-train|local_train) shift; cmd_local_train "$@" ;;
    help|-h|--help) usage ;;
    *) echo "Lệnh không rõ: $1"; usage; exit 1 ;;
esac
