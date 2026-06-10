#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy_to_server.sh
# Chạy script này từ LAPTOP để copy workspace lên SERVER và build Docker image
#
# Cách dùng:
#   ./deploy_to_server.sh <server_user>@<server_ip>
#   Ví dụ:
#   ./deploy_to_server.sh nvidia@192.168.1.122
# ─────────────────────────────────────────────────────────────────────────────
set -e

SERVER=${1:-"naiscorp@192.168.1.122"}
REMOTE_DIR="/home/$(echo $SERVER | cut -d@ -f1)/openarm_ws"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(dirname $SCRIPT_DIR)"

echo "============================================================"
echo " OpenArm Docker Deploy"
echo " SERVER : $SERVER"
echo " REMOTE : $REMOTE_DIR"
echo " LOCAL  : $WS_DIR"
echo "============================================================"

# ── 1. Tạo thư mục trên server ────────────────────────────────────────────
echo ""
echo "[1/4] Tạo thư mục trên server..."
ssh "$SERVER" "mkdir -p ${REMOTE_DIR}/src ${REMOTE_DIR}/docker"

# ── 2. Sync source packages ───────────────────────────────────────────────
echo ""
echo "[2/4] Sync source packages (rsync)..."
rsync -avz --delete --progress \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'build/' \
    --exclude 'install/' \
    --exclude 'log/' \
    "${WS_DIR}/src/" \
    "${SERVER}:${REMOTE_DIR}/src/"

# ── 3. Sync docker files ──────────────────────────────────────────────────
echo ""
echo "[3/4] Sync docker files..."
rsync -avz --delete --progress \
    "${WS_DIR}/docker/" \
    "${SERVER}:${REMOTE_DIR}/docker/"

# ── 4. Build Docker image trên server ─────────────────────────────────────
echo ""
echo "[4/4] Build Docker image trên server..."
ssh "$SERVER" "cd ${REMOTE_DIR} && docker build -f docker/Dockerfile -t openarm-humble:latest ."

echo ""
echo "============================================================"
echo " ✅ Deploy xong!"
echo ""
echo " Để chạy VLA Bridge trên server:"
echo "   ssh $SERVER"
echo "   cd ${REMOTE_DIR}"
echo "   VLA_INSTRUCTION=\"push the lemon to the box\" \\"
echo "   VLA_HOST=localhost \\"
echo "   docker compose -f docker/docker-compose.yml up vla_bridge"
echo ""
echo " Hoặc debug shell:"
echo "   docker compose -f docker/docker-compose.yml run --rm shell"
echo "============================================================"
