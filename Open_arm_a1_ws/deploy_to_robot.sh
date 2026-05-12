#!/bin/bash

# Configuration
REMOTE_USER="ubuntu"
REMOTE_HOST="100.92.66.138"
REMOTE_WS="~/arm_ws"
LOCAL_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Syncing source code to robot ($REMOTE_HOST)..."

# Syncing src directory, excluding unnecessary files
rsync -avz --delete \
    --exclude 'build/' \
    --exclude 'install/' \
    --exclude 'log/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    "$LOCAL_WS/src/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_WS/src/"

if [ $? -ne 0 ]; then
    echo "❌ Sync failed! Please check your SSH connection."
    exit 1
fi

echo "🏗️ Building on robot..."

# Execute build command on robot
# We use bash -ic to ensure .bashrc is sourced, which typically sets up the ROS 2 environment
ssh "$REMOTE_USER@$REMOTE_HOST" "bash -ic 'cd $REMOTE_WS && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'"

if [ $? -eq 0 ]; then
    echo "✅ Deploy and build successful!"
else
    echo "❌ Build failed on robot!"
    exit 1
fi
