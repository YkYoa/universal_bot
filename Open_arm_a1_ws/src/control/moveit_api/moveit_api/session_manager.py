#!/usr/bin/env python3
"""
session_manager.py

Thread-safe single-operator leased control manager with FIFO queuing.
Ensures only one operator has control of supervisory FSM actions at any time.
Any other connected client is placed in a view-only waiting queue.

Key properties:
  - Strict exclusive control: Only active controller can issue commands.
  - FIFO queue: When the active controller leaves, the next in line is promoted.
  - Inactivity timeout: 5 minutes (300 seconds) of no user activity revokes lease.
  - Disconnect grace period: 10 seconds allows page reload (F5) without losing turn.
  - Voluntary release: Operator can yield control immediately.
"""

import threading
import time
from typing import Callable, Dict, List, Optional


class SessionManager:
    def __init__(
        self,
        on_broadcast: Optional[Callable[[dict], None]] = None,
        afk_timeout: float = 300.0,
        grace_period: float = 10.0,
    ):
        self._lock = threading.RLock()
        self.on_broadcast = on_broadcast
        self.afk_timeout = afk_timeout
        self.grace_period = grace_period

        # Active controller record:
        # { 'session_id': str, 'socket_id': str, 'last_active': float, 'disconnect_ts': float | None }
        self._active_controller: Optional[dict] = None

        # FIFO queue:
        # [ { 'session_id': str, 'socket_id': str, 'joined_ts': float, 'last_seen': float, 'disconnect_ts': float | None }, ... ]
        self._queue: List[dict] = []

        # Background reaper for grace period and AFK expiration
        self._running = True
        self._reaper_thread = threading.Thread(target=self._reaper_loop, daemon=True)
        self._reaper_thread.start()

    def get_state(self) -> dict:
        """Return public snapshot of session manager state."""
        with self._lock:
            now = time.time()
            ctrl_id = None
            ctrl_last_active = None
            if self._active_controller:
                ctrl_id = self._active_controller['session_id']
                ctrl_last_active = self._active_controller['last_active']

            queue_ids = [item['session_id'] for item in self._queue]

            return {
                'controller_id': ctrl_id,
                'controller_last_active': ctrl_last_active,
                'queue': queue_ids,
                'queue_count': len(queue_ids),
                'afk_timeout_s': int(self.afk_timeout),
                'grace_period_s': int(self.grace_period),
                'timestamp': now,
            }

    def _broadcast(self) -> None:
        """Emit state update via callback if configured."""
        if self.on_broadcast:
            try:
                state = self.get_state()
                self.on_broadcast(state)
            except Exception:
                pass

    def join(self, session_id: str, socket_id: str = '') -> dict:
        """Register a client session entering /fsm."""
        if not session_id:
            return self.get_state()

        with self._lock:
            now = time.time()

            # 1. Is this already the active controller?
            if self._active_controller and self._active_controller['session_id'] == session_id:
                self._active_controller['socket_id'] = socket_id
                self._active_controller['disconnect_ts'] = None
                self._broadcast()
                return self.get_state()

            # 2. If no active controller, acquire lock immediately
            if self._active_controller is None:
                self._active_controller = {
                    'session_id': session_id,
                    'socket_id': socket_id,
                    'last_active': now,
                    'disconnect_ts': None,
                }
                # Ensure not also in queue
                self._queue = [q for q in self._queue if q['session_id'] != session_id]
                self._broadcast()
                return self.get_state()

            # 3. Otherwise, enqueue in waiting list if not already present
            existing = next((q for q in self._queue if q['session_id'] == session_id), None)
            if existing:
                existing['socket_id'] = socket_id
                existing['last_seen'] = now
                existing['disconnect_ts'] = None
            else:
                self._queue.append({
                    'session_id': session_id,
                    'socket_id': socket_id,
                    'joined_ts': now,
                    'last_seen': now,
                    'disconnect_ts': None,
                })

            self._broadcast()
            return self.get_state()

    def heartbeat(self, session_id: str, user_active: bool = False, socket_id: str = '') -> dict:
        """Update activity timestamp from client periodic ping."""
        if not session_id:
            return self.get_state()

        with self._lock:
            now = time.time()

            # Active controller
            if self._active_controller and self._active_controller['session_id'] == session_id:
                if socket_id:
                    self._active_controller['socket_id'] = socket_id
                self._active_controller['disconnect_ts'] = None
                if user_active:
                    self._active_controller['last_active'] = now
                return self.get_state()

            # Queued client
            existing = next((q for q in self._queue if q['session_id'] == session_id), None)
            if existing:
                if socket_id:
                    existing['socket_id'] = socket_id
                existing['last_seen'] = now
                existing['disconnect_ts'] = None

            return self.get_state()

    def leave(self, session_id: str) -> dict:
        """Client explicitly left /fsm or navigated away."""
        if not session_id:
            return self.get_state()

        with self._lock:
            if self._active_controller and self._active_controller['session_id'] == session_id:
                self._active_controller = None
                self._promote_next()
                self._broadcast()
                return self.get_state()

            prev_len = len(self._queue)
            self._queue = [q for q in self._queue if q['session_id'] != session_id]
            if len(self._queue) != prev_len:
                self._broadcast()

            return self.get_state()

    def release_control(self, session_id: str) -> bool:
        """Voluntary handover by active controller."""
        with self._lock:
            if self._active_controller and self._active_controller['session_id'] == session_id:
                self._active_controller = None
                self._promote_next()
                self._broadcast()
                return True
            return False

    def on_socket_disconnect(self, socket_id: str) -> None:
        """Mark socket as disconnected; grace period timer begins."""
        with self._lock:
            now = time.time()
            if self._active_controller and self._active_controller['socket_id'] == socket_id:
                self._active_controller['disconnect_ts'] = now

            for q in self._queue:
                if q['socket_id'] == socket_id:
                    q['disconnect_ts'] = now

    def is_controller(self, session_id: str) -> bool:
        """Check whether given session_id holds valid, non-expired control lease."""
        if not session_id:
            return False

        with self._lock:
            if not self._active_controller:
                return False

            if self._active_controller['session_id'] != session_id:
                return False

            now = time.time()
            # Check AFK timeout
            if (now - self._active_controller['last_active']) > self.afk_timeout:
                return False

            # Check disconnect grace period
            disc_ts = self._active_controller.get('disconnect_ts')
            if disc_ts and (now - disc_ts) > self.grace_period:
                return False

            return True

    def _promote_next(self) -> None:
        """Promote head of FIFO queue to active controller."""
        now = time.time()
        while self._queue:
            candidate = self._queue.pop(0)
            # Skip candidates whose disconnect grace period already expired
            disc_ts = candidate.get('disconnect_ts')
            if disc_ts and (now - disc_ts) > self.grace_period:
                continue

            self._active_controller = {
                'session_id': candidate['session_id'],
                'socket_id': candidate['socket_id'],
                'last_active': now,
                'disconnect_ts': candidate.get('disconnect_ts'),
            }
            return

        self._active_controller = None

    def _reaper_loop(self) -> None:
        """Background thread checking AFK timeout and disconnect grace period."""
        while self._running:
            try:
                time.sleep(1.0)
                now = time.time()
                need_broadcast = False

                with self._lock:
                    # 1. Check active controller disconnect grace period
                    if self._active_controller:
                        disc_ts = self._active_controller.get('disconnect_ts')
                        if disc_ts and (now - disc_ts) > self.grace_period:
                            # Disconnected past grace period -> yield to queue
                            self._active_controller = None
                            self._promote_next()
                            need_broadcast = True
                        elif (now - self._active_controller['last_active']) > self.afk_timeout:
                            # AFK timeout (5m) expired -> yield to queue
                            self._active_controller = None
                            self._promote_next()
                            need_broadcast = True

                    # 2. Prune expired queue candidates
                    valid_queue = []
                    for q in self._queue:
                        disc_ts = q.get('disconnect_ts')
                        if disc_ts and (now - disc_ts) > self.grace_period:
                            need_broadcast = True
                            continue
                        valid_queue.append(q)
                    self._queue = valid_queue

                if need_broadcast:
                    self._broadcast()

            except Exception:
                pass
