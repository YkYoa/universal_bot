#!/bin/bash
# /etc/cron.daily/supervisor-log-cleanup
#
# Enforces the supervisor log retention policy:
#   - Delete .jsonl files older than 30 days
#   - If total directory size > 3 GB, remove oldest files until < 2.5 GB
#
# Runs daily as root (cron.daily context). Logs its own actions to syslog.
# To install:
#   sudo cp supervisor_log_cleanup.sh /etc/cron.daily/supervisor-log-cleanup
#   sudo chmod +x /etc/cron.daily/supervisor-log-cleanup

set -euo pipefail

LOG_DIR="/home/ubuntu/robot-healthmate/logs/supervisor"
MAX_AGE_DAYS=30
HARD_LIMIT_GB=3
SOFT_TARGET_GB=2      # trim down to this when over the hard limit

logger -t supervisor-log-cleanup "Starting supervisor log cleanup in $LOG_DIR"

# Create the directory if it doesn't exist yet (first run before any logs)
mkdir -p "$LOG_DIR"

# ── Step 1: Remove files older than MAX_AGE_DAYS ─────────────────────────
OLD_COUNT=0
while IFS= read -r -d '' f; do
    rm -f "$f"
    logger -t supervisor-log-cleanup "Deleted old log: $f"
    OLD_COUNT=$((OLD_COUNT + 1))
done < <(find "$LOG_DIR" -maxdepth 1 -name 'supervisor-*.jsonl' \
         -mtime +"$MAX_AGE_DAYS" -print0 2>/dev/null)

if [ "$OLD_COUNT" -gt 0 ]; then
    logger -t supervisor-log-cleanup "Removed $OLD_COUNT file(s) older than ${MAX_AGE_DAYS} days"
fi

# ── Step 2: Trim total size if over hard limit ────────────────────────────
HARD_LIMIT_BYTES=$(( HARD_LIMIT_GB * 1024 * 1024 * 1024 ))
SOFT_TARGET_BYTES=$(( SOFT_TARGET_GB * 1024 * 1024 * 1024 ))

# Calculate current total size
TOTAL=$(du -sb "$LOG_DIR" 2>/dev/null | awk '{print $1}')
TOTAL=${TOTAL:-0}

if [ "$TOTAL" -gt "$HARD_LIMIT_BYTES" ]; then
    logger -t supervisor-log-cleanup \
        "Total size ${TOTAL} bytes exceeds ${HARD_LIMIT_GB}GB limit — trimming to ${SOFT_TARGET_GB}GB"

    # Remove oldest files first (sorted ascending by name = ascending by date)
    while IFS= read -r -d '' f; do
        CURRENT=$(du -sb "$LOG_DIR" 2>/dev/null | awk '{print $1}')
        CURRENT=${CURRENT:-0}
        if [ "$CURRENT" -le "$SOFT_TARGET_BYTES" ]; then
            break
        fi
        rm -f "$f"
        logger -t supervisor-log-cleanup "Trimmed (size limit): $f"
    done < <(find "$LOG_DIR" -maxdepth 1 -name 'supervisor-*.jsonl' \
             -print0 2>/dev/null | sort -z)
fi

logger -t supervisor-log-cleanup "Cleanup complete"
