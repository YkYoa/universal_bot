#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Setup base directory (absolute path to the script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== CONFIGURATION ====================
# Path to the source directory to synchronize
SRC_DIR="${SCRIPT_DIR}/Open_arm_a1_ws"

# Path to the company repository workspace root
COMPANY_REPO_ROOT="/home/hans/robot-healthmate/robot-healthmate"

# Destination subdirectory in the company workspace
DEST_DIR="${COMPANY_REPO_ROOT}/robot/Open_arm_a1_ws"
# =======================================================

# Colors for terminal logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# 1. Validation
if [ ! -d "$SRC_DIR" ]; then
    log_error "Source directory '$SRC_DIR' does not exist."
    exit 1
fi

if [ ! -d "$COMPANY_REPO_ROOT" ]; then
    log_error "Company repository root '$COMPANY_REPO_ROOT' does not exist."
    exit 1
fi

# Create destination directory if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
    log_warning "Destination directory '$DEST_DIR' does not exist. Creating it..."
    mkdir -p "$DEST_DIR"
fi

# Detect branch to synchronize
# If an argument is provided, use it; otherwise, detect the current branch of the source repo
SRC_BRANCH=$(git -C "$SCRIPT_DIR" branch --show-current 2>/dev/null || echo "main")
SYNC_BRANCH="${1:-$SRC_BRANCH}"

# Get the last commit message from the source repository to use for the company commit
LAST_COMMIT_MSG=$(git -C "$SCRIPT_DIR" log -1 --pretty=%B 2>/dev/null || echo "manual sync")
LAST_COMMIT_MSG=$(echo "$LAST_COMMIT_MSG" | xargs) # strip whitespace

log_info "Synchronizing branch: '$SYNC_BRANCH'"
log_info "Last commit message: '$LAST_COMMIT_MSG'"

# 2. Update company repository
log_info "Navigating to company repository: '$COMPANY_REPO_ROOT'..."
cd "$COMPANY_REPO_ROOT"

# Check if the target branch exists in the company repo, otherwise fall back to main/master or current
if git show-ref --verify --quiet "refs/heads/$SYNC_BRANCH"; then
    log_info "Switching to branch '$SYNC_BRANCH' in company repository..."
    git checkout "$SYNC_BRANCH"
else
    # Check if branch exists on remote
    if git show-ref --verify --quiet "refs/remotes/origin/$SYNC_BRANCH"; then
        log_info "Checking out remote branch '$SYNC_BRANCH' in company repository..."
        git checkout -b "$SYNC_BRANCH" "origin/$SYNC_BRANCH"
    else
        log_warning "Branch '$SYNC_BRANCH' does not exist in company repository. Using current active branch instead."
        SYNC_BRANCH=$(git branch --show-current)
        log_info "Active branch in company repo: '$SYNC_BRANCH'"
    fi
fi

log_info "Pulling latest changes in company repository..."
git pull origin "$SYNC_BRANCH"

# 3. Synchronize files using rsync
log_info "Copying files from '$SRC_DIR/' to '$DEST_DIR/'..."
rsync -av --delete \
    --exclude="build/" \
    --exclude="install/" \
    --exclude="log/" \
    --exclude="logs/" \
    --exclude=".codegraph/" \
    --exclude=".vscode/" \
    --exclude="__pycache__/" \
    --exclude="*.log" \
    --exclude=".git/" \
    --exclude=".gitattributes" \
    "$SRC_DIR/" "$DEST_DIR/"

# 4. Git status & push in the company repository
log_info "Checking for differences in company workspace..."
if [ -n "$(git status --porcelain)" ]; then
    log_info "Changes detected. Staging changes..."
    git add .
    
    log_info "Committing changes with message: 'sync: $LAST_COMMIT_MSG'..."
    git commit -m "sync: $LAST_COMMIT_MSG"
    
    log_info "Pushing synchronization to company remote..."
    git push origin "$SYNC_BRANCH"
    log_success "Successfully synchronized and pushed to company repository!"
else
    log_success "No changes detected. Company repository is already in sync."
fi
