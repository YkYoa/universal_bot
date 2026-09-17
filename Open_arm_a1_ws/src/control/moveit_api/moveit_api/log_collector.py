#!/usr/bin/env python3
"""
log_collector.py

Subscribes to /rosout (ROS2's built-in aggregated log topic) and captures
ERROR/FATAL-severity messages from the supervisor node set. Each captured
entry is:
  1. Written as a JSON Line to a daily rotating file under LOG_DIR
  2. Forwarded to an on_log(entry) callback so robot_api_server can push
     it out over the WebSocket in real time.

Log file format: supervisor-YYYY-MM-DD.jsonl
JSON Line schema:
  {
    "ts":   "2026-09-16T13:45:00.123Z",  // ISO-8601 UTC
    "level": "ERROR"|"FATAL",
    "node":  "qvic_fsm_node",
    "file":  "sequence_fsm.cpp",
    "line":  123,
    "msg":   "builtin action failed: ..."
  }

Retention policy:
  - Files older than 30 days are removed by /etc/cron.daily/supervisor-log-cleanup
  - This module only writes; cleanup is a separate script
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                        ReliabilityPolicy)
from rcl_interfaces.msg import Log

# ── configuration ─────────────────────────────────────────────────────────

LOG_DIR = Path(os.path.expanduser('~/robot-healthmate/logs/supervisor'))

# Only these ROS2 node names produce entries. Partial match (substring).
WATCHED_NODES = {
    'sequence_executor_node',
    'robot_skills_node',
    'robot_api_server',
    'qvic_fsm_node',
    'SkillClient',             # rclcpp logger name used inside skill_client.cpp
    'MoveItCppPlannerManager',
}

# rcl_interfaces/msg/Log severity constants
# DEBUG=10 INFO=20 WARN=30 ERROR=40 FATAL=50
_LEVEL_WARN = 30
_LEVEL_ERROR = 40
_LEVEL_FATAL = 50

_LEVEL_MAP = {
    _LEVEL_WARN: 'WARN',
    _LEVEL_ERROR: 'ERROR',
    _LEVEL_FATAL: 'FATAL',
}


def _level_name(severity: int) -> str:
    return _LEVEL_MAP.get(severity, f'LEVEL{severity}')


def _is_watched(name: str) -> bool:
    """True if *name* is or contains any watched node name."""
    for w in WATCHED_NODES:
        if w in name:
            return True
    return False


class LogCollector(Node):
    """ROS2 node that subscribes to /rosout and captures ERROR+ entries."""

    def __init__(self, on_log=None):
        """
        Parameters
        ----------
        on_log : callable(entry_dict) | None
            Called on the ROS executor thread each time an entry is captured.
            Use it to push the entry through SocketIO.  Must not block.
        """
        super().__init__('log_collector')
        self._on_log = on_log
        self._lock = threading.Lock()
        self._current_date: str = ''
        self._file = None
        self._warn_throttle: dict = {}  # (node, msg) -> (last_seen_epoch, suppressed_count)

        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # /rosout is KEEP_LAST depth-100 BEST_EFFORT VOLATILE by convention.
        qos = QoSProfile(
            depth=100,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Log, '/rosout', self._on_msg, qos)
        self.get_logger().info(
            f'LogCollector watching /rosout for {sorted(WATCHED_NODES)}'
        )

    # ── ROS callback ──────────────────────────────────────────────────────

    def _on_msg(self, msg: Log) -> None:
        if msg.level < _LEVEL_WARN:
            return
        if not _is_watched(msg.name):
            return

        # Warning spam throttling: suppress duplicate warnings within 3.0s window per (node, msg)
        if msg.level == _LEVEL_WARN:
            now = time.time()
            throttle_key = (msg.name, msg.msg)
            with self._lock:
                last_time, suppressed_count = self._warn_throttle.get(throttle_key, (0.0, 0))
                if (now - last_time) < 3.0:
                    self._warn_throttle[throttle_key] = (last_time, suppressed_count + 1)
                    return
                # Window expired, record this warning
                self._warn_throttle[throttle_key] = (now, 0)
                if len(self._warn_throttle) > 200:
                    self._warn_throttle = {
                        k: v for k, v in self._warn_throttle.items()
                        if (now - v[0]) < 10.0
                    }

        stamp_ns = msg.stamp.sec * 1_000_000_000 + msg.stamp.nanosec
        ts = datetime.fromtimestamp(stamp_ns / 1e9, tz=timezone.utc)
        entry = {
            'ts':    ts.strftime('%Y-%m-%dT%H:%M:%S.') + f'{ts.microsecond // 1000:03d}Z',
            'level': _level_name(msg.level),
            'node':  msg.name,
            'file':  msg.file,
            'line':  msg.line,
            'msg':   msg.msg,
        }

        self._write(entry, ts)

        if self._on_log:
            try:
                self._on_log(entry)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f'on_log callback raised: {exc}')

    # ── file writing ──────────────────────────────────────────────────────

    def _write(self, entry: dict, ts: datetime) -> None:
        date_str = ts.strftime('%Y-%m-%d')
        with self._lock:
            if date_str != self._current_date:
                self._rotate(date_str)
            if self._file:
                try:
                    self._file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    self._file.flush()
                except OSError as exc:
                    self.get_logger().error(f'log write failed: {exc}')

    def _rotate(self, date_str: str) -> None:
        """Open a new daily log file, closing the previous one."""
        if self._file:
            try:
                self._file.close()
            except OSError:
                pass
        path = LOG_DIR / f'supervisor-{date_str}.jsonl'
        try:
            self._file = open(path, 'a', encoding='utf-8')
            self._current_date = date_str
        except OSError as exc:
            self.get_logger().error(f'could not open log file {path}: {exc}')
            self._file = None

    def destroy_node(self) -> None:
        with self._lock:
            if self._file:
                try:
                    self._file.close()
                except OSError:
                    pass
        super().destroy_node()

    # ── REST helper: read historical entries ──────────────────────────────

    @staticmethod
    def read_entries(since=None, limit=200, level=None, node=None):
        """
        Read up to *limit* entries from the .jsonl files, newest first.

        Parameters
        ----------
        since : ISO timestamp string (UTC) or None — exclude entries older than this
        limit : max entries to return
        level : filter to this severity (case-insensitive), e.g. 'ERROR'
        node  : filter to entries whose node field contains this string
        """
        files = sorted(LOG_DIR.glob('supervisor-*.jsonl'), reverse=True)
        results = []
        since_ts = None
        if since:
            try:
                since_ts = datetime.fromisoformat(since.replace('Z', '+00:00'))
            except ValueError:
                pass

        level_upper = level.upper() if level else None

        for path in files:
            try:
                lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
            except OSError:
                continue
            for raw in reversed(lines):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if since_ts:
                    try:
                        entry_ts = datetime.fromisoformat(
                            entry['ts'].replace('Z', '+00:00')
                        )
                        if entry_ts <= since_ts:
                            continue
                    except (KeyError, ValueError):
                        pass
                if level_upper and entry.get('level', '').upper() != level_upper:
                    continue
                if node and node.lower() not in entry.get('node', '').lower():
                    continue
                results.append(entry)
                if len(results) >= limit:
                    return results
        return results

    @staticmethod
    def list_files():
        """List available log files with date, size (bytes)."""
        files = sorted(LOG_DIR.glob('supervisor-*.jsonl'), reverse=True)
        out = []
        for path in files:
            try:
                stat = path.stat()
                size = stat.st_size
                date_str = path.stem.replace('supervisor-', '')
                out.append({'date': date_str, 'size': size, 'file': path.name})
            except OSError:
                continue
        return out
