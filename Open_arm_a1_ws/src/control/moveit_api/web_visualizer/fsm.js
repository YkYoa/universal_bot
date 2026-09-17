/* FSM viewer + real-time error console.
 *
 * Draws both layers of the state machine from /api/fsm/graph and lights up
 * whichever node the robot is in, following `fsm_state` socket events.
 *
 * Also subscribes to `log_event` socket events from the server-side
 * LogCollector (/rosout -> robot_api_server -> SocketIO) and renders them
 * in the collapsible Console section below Detail.
 *
 * Deliberately plain: no framework, no build step, no graph library.
 * app.js is a separate, minified bundle - this file is not part of it.
 */
(function () {
  'use strict';

  var KIND_COLOR = {
    start: '#38bdf8', normal: '#475569', active: '#38bdf8',
    success: '#10b981', warning: '#f59e0b', error: '#f43f5e', special: '#a855f7'
  };

  var el = function (id) { return document.getElementById(id); }
  var layers = {};
  var lastState = null;
  var lastNodeByLayer = {};

  // ── layout ────────────────────────────────────────────────────────────────

  function layout(spec, width, height) {
    var cx = width / 2, cy = height / 2;
    var radius = Math.min(width, height) / 2 - 58;
    var positions = {};
    spec.nodes.forEach(function (node, i) {
      var angle = (i / spec.nodes.length) * Math.PI * 2 - Math.PI / 2;
      positions[node.id] = { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
    });
    return positions;
  }

  function svgEl(name, attrs) {
    var node = document.createElementNS('http://www.w3.org/2000/svg', name);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }

  function drawLayer(svg, spec) {
    var width = 720, height = 380;
    var pos = layout(spec, width, height);
    svg.innerHTML = '';

    var defs = svgEl('defs');
    var marker = svgEl('marker', {
      id: 'arrow-' + spec.id, viewBox: '0 0 10 10', refX: 9, refY: 5,
      markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse'
    });
    marker.appendChild(svgEl('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: 'rgba(255,255,255,0.28)' }));
    defs.appendChild(marker);
    svg.appendChild(defs);

    var edgeLayer = svgEl('g');
    var nodeLayer = svgEl('g');
    svg.appendChild(edgeLayer);
    svg.appendChild(nodeLayer);

    var radius = 30;
    var edges = {};
    spec.edges.forEach(function (edge) {
      var a = pos[edge.from], b = pos[edge.to];
      if (!a || !b) { return; }
      var dx = b.x - a.x, dy = b.y - a.y;
      var len = Math.hypot(dx, dy) || 1;
      var ux = dx / len, uy = dy / len;
      var ox = -uy * 5, oy = ux * 5;
      var line = svgEl('line', {
        x1: a.x + ux * radius + ox, y1: a.y + uy * radius + oy,
        x2: b.x - ux * radius + ox, y2: b.y - uy * radius + oy,
        stroke: 'rgba(255,255,255,0.16)', 'stroke-width': 1.2,
        'marker-end': 'url(#arrow-' + spec.id + ')'
      });
      var title = svgEl('title');
      title.textContent = edge.from + ' → ' + edge.to + ': ' + edge.label;
      line.appendChild(title);
      edgeLayer.appendChild(line);
      edges[edge.from + '>' + edge.to] = line;
    });

    var nodes = {};
    spec.nodes.forEach(function (node) {
      var p = pos[node.id];
      var group = svgEl('g');
      var circle = svgEl('circle', {
        cx: p.x, cy: p.y, r: radius,
        fill: '#23262c', stroke: KIND_COLOR[node.kind] || KIND_COLOR.normal,
        'stroke-width': 1.5
      });
      var title = svgEl('title');
      title.textContent = node.label + ' — ' + node.description;
      circle.appendChild(title);
      var label = svgEl('text', {
        x: p.x, y: p.y + 4, 'text-anchor': 'middle', class: 'node-label'
      });
      label.textContent = node.label;
      group.appendChild(circle);
      group.appendChild(label);
      nodeLayer.appendChild(group);
      nodes[node.id] = { circle: circle, color: KIND_COLOR[node.kind] || KIND_COLOR.normal };
    });

    return { spec: spec, nodes: nodes, edges: edges };
  }

  function highlight(layerId, activeId) {
    var layer = layers[layerId];
    if (!layer) { return; }
    Object.keys(layer.nodes).forEach(function (id) {
      var node = layer.nodes[id];
      var on = id === activeId;
      node.circle.setAttribute('fill', on ? node.color : '#0f172a');
      node.circle.setAttribute('stroke', on ? '#38bdf8' : node.color);
      node.circle.setAttribute('stroke-width', on ? 3 : 1.5);
      node.circle.setAttribute('opacity', on || !activeId ? 1 : 0.45);
    });
    var previous = lastNodeByLayer[layerId];
    if (previous && activeId && previous !== activeId) {
      var edge = layer.edges[previous + '>' + activeId];
      if (edge) {
        edge.setAttribute('stroke', '#38bdf8');
        edge.setAttribute('stroke-width', 2.4);
        setTimeout(function () {
          edge.setAttribute('stroke', 'rgba(255,255,255,0.16)');
          edge.setAttribute('stroke-width', 1.2);
        }, 900);
      }
    }
    lastNodeByLayer[layerId] = activeId;
  }

  // ── state rendering ───────────────────────────────────────────────────────

  function render(state) {
    lastState = state;
    highlight('robot', state.robot_state);
    highlight('sequence', state.sequence_state || null);

    var robotStateEl = el('fsm-robot-state');
    var robotPill = el('fsm-robot-pill');
    if (robotStateEl) {
      robotStateEl.textContent = state.robot_state || 'IDLE';
    }
    if (robotPill) {
      if (state.robot_state === 'RUNNING' || state.robot_state === 'IDLE') {
        robotPill.className = 'pill pill-live';
      } else if (state.robot_state === 'PAUSED' || state.robot_state === 'BOOTING') {
        robotPill.className = 'pill pill-degraded';
      } else {
        robotPill.className = 'pill pill-dead';
      }
    }

    var modeEl = el('mode');
    if (modeEl) {
      modeEl.textContent = state.control_mode_active || 'position';
    }

    var seqEl = el('seq');
    if (seqEl) {
      seqEl.textContent = state.sequence_name
        ? state.sequence_name + (state.loop_total === -1
            ? ' (loop ' + (state.loop_index + 1) + ', forever)'
            : ' (loop ' + (state.loop_index + 1) + '/' + state.loop_total + ')')
        : 'no sequence';
    }

    var fault = el('fault');
    if (state.fault_reason) {
      fault.textContent = state.fault_reason;
      fault.classList.add('show');
    } else {
      fault.classList.remove('show');
    }

    el('progressBar').style.width = Math.round((state.progress || 0) * 100) + '%';

    var running = state.robot_state === 'RUNNING';
    var paused = state.robot_state === 'PAUSED';
    var faulted = state.robot_state === 'FAULT';

    var motorEl = el('fsm-motors-state');
    var motorPill = el('fsm-motor-pill');
    if (motorEl) {
      if (state.motors_enabled === false) {
        motorEl.textContent = 'DISABLED';
        motorEl.style.color = '#f87171';
        if (motorPill) {
          motorPill.className = 'pill pill-interactive pill-motor-disabled';
          motorPill.title = 'Click to enable motors';
        }
      } else {
        motorEl.textContent = 'ENABLED';
        motorEl.style.color = '#34d399';
        if (motorPill) {
          motorPill.className = 'pill pill-interactive pill-motor-enabled';
          motorPill.title = 'Motors active';
        }
      }
    }

    var stopBtn = el('btnStop');
    var abortBtn = el('btnAbortHome');
    var pauseBtn = el('btnPause');
    var resumeBtn = el('btnResume');
    var clearBtn = el('btnClear');
    var runBtn = el('btnRun');
    var dryBtn = el('btnDry');

    if (!_isController) {
      if (pauseBtn) { pauseBtn.disabled = true; }
      if (resumeBtn) { resumeBtn.disabled = true; }
      if (clearBtn) { clearBtn.disabled = true; }
      if (stopBtn) { stopBtn.disabled = true; }
      if (abortBtn) { abortBtn.disabled = true; }
      if (runBtn) { runBtn.disabled = true; }
      if (dryBtn) { dryBtn.disabled = true; }
    } else {
      if (pauseBtn) { pauseBtn.disabled = !running; }
      if (resumeBtn) { resumeBtn.disabled = !paused; }
      if (clearBtn) { clearBtn.disabled = !faulted; }
      if (stopBtn) { stopBtn.disabled = false; }
      if (abortBtn) { abortBtn.disabled = false; }
      if (runBtn) { runBtn.disabled = running || paused; }
      if (dryBtn) { dryBtn.disabled = running || paused; }
    }

    renderSteps(state);
    renderDetail(state);
  }

  function renderSteps(state) {
    var body = el('steps');
    body.innerHTML = '';
    if (!state.step_total) {
      body.innerHTML = '<tr><td class="idx">—</td><td>nothing running</td></tr>';
      return;
    }
    for (var i = 0; i < state.step_total; i++) {
      var row = document.createElement('tr');
      if (i === state.step_index) { row.className = 'current'; }
      else if (i < state.step_index) { row.className = 'done'; }
      var idx = document.createElement('td');
      idx.className = 'idx';
      idx.textContent = i;
      var name = document.createElement('td');
      if (i === state.step_index) {
        name.innerHTML = '<strong></strong><br><span class="type"></span>';
        name.querySelector('strong').textContent = state.step_name || '(step ' + i + ')';
        name.querySelector('.type').textContent = state.step_type || '';
      } else {
        name.textContent = i < state.step_index ? 'done' : '';
      }
      row.appendChild(idx);
      row.appendChild(name);
      body.appendChild(row);
    }
  }

  function renderDetail(state) {
    var pairs = [
      ['Robot', state.robot_state],
      ['Sequence', state.sequence_state || '—'],
      ['Step', state.step_total ? (state.step_index + 1) + ' / ' + state.step_total : '—'],
      ['Progress', Math.round((state.progress || 0) * 100) + '%'],
      ['Control mode', state.control_mode_active || '?']
    ];
    var dl = el('detail');
    dl.innerHTML = '';
    pairs.forEach(function (pair) {
      var dt = document.createElement('dt');
      dt.textContent = pair[0];
      var dd = document.createElement('dd');
      dd.textContent = pair[1];
      dl.appendChild(dt);
      dl.appendChild(dd);
    });
  }

  // ── Console ───────────────────────────────────────────────────────────────

  var MAX_LOG_ENTRIES = 50;
  var _logEntries = [];    // [{level, node, ts, msg, file, line}, ...]
  var _activeFilters = { WARN: true, ERROR: true, FATAL: true };
  var _userScrolled = false;  // true when user has scrolled up in the console

  function initConsole() {
    var header = el('consoleHeader');
    var section = el('consoleSection');
    var body = el('consoleBody');

    // Toggle expand/collapse
    function toggleConsole() {
      var open = section.classList.toggle('open');
      header.setAttribute('aria-expanded', open ? 'true' : 'false');
      if (open) { scrollConsoleToBottom(); }
    }
    header.addEventListener('click', toggleConsole);
    header.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleConsole(); }
    });

    // Detect user scroll (so we don't force scroll to bottom while they're reading)
    body.addEventListener('scroll', function () {
      var atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 20;
      _userScrolled = !atBottom;
    });

    // Severity filter buttons
    document.querySelectorAll('.sev-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var lvl = btn.getAttribute('data-level');
        _activeFilters[lvl] = !_activeFilters[lvl];
        btn.classList.toggle('active', _activeFilters[lvl]);
        rebuildConsole();
      });
    });
  }

  function scrollConsoleToBottom() {
    var body = el('consoleBody');
    if (body) { body.scrollTop = body.scrollHeight; }
  }

  function formatTs(tsStr) {
    // "2026-09-16T13:45:00.123Z" → "13:45:00"
    try {
      return tsStr.substring(11, 19);
    } catch (e) { return tsStr || ''; }
  }

  function buildEntryEl(entry) {
    var div = document.createElement('div');
    div.className = 'log-entry';

    var ts = document.createElement('span');
    ts.className = 'log-ts';
    ts.textContent = formatTs(entry.ts);

    var lvl = document.createElement('span');
    lvl.className = 'log-lvl ' + (entry.level || 'ERROR');
    lvl.textContent = entry.level || 'ERROR';

    var node = document.createElement('span');
    node.className = 'log-node';
    // Shorten long node names
    var nodeName = (entry.node || '').replace('sequence_executor_node', 'seq_exec')
                                     .replace('MoveItCppPlannerManager', 'planner');
    node.textContent = '[' + nodeName + ']';

    var msg = document.createElement('span');
    msg.className = 'log-msg';
    msg.textContent = entry.msg || '';

    div.appendChild(ts);
    div.appendChild(lvl);
    div.appendChild(node);
    div.appendChild(msg);
    return div;
  }

  function rebuildConsole() {
    var body = el('consoleBody');
    var emptyEl = el('consoleEmpty');
    if (!body) return;

    // Clear all entries (keep the empty placeholder)
    var children = Array.from(body.querySelectorAll('.log-entry'));
    children.forEach(function (c) { c.remove(); });

    var visible = _logEntries.filter(function (e) {
      return _activeFilters[e.level] !== false;
    });

    if (visible.length === 0) {
      if (emptyEl) emptyEl.style.display = '';
      return;
    }
    if (emptyEl) emptyEl.style.display = 'none';

    visible.forEach(function (entry) {
      body.appendChild(buildEntryEl(entry));
    });

    if (!_userScrolled) { scrollConsoleToBottom(); }
  }

  function addLogEntry(entry) {
    // Add to buffer, cap at MAX_LOG_ENTRIES
    _logEntries.push(entry);
    if (_logEntries.length > MAX_LOG_ENTRIES) {
      _logEntries.shift();
    }

    // Only append to DOM if we pass the active filter
    if (_activeFilters[entry.level] === false) return;

    var body = el('consoleBody');
    var emptyEl = el('consoleEmpty');
    if (!body) return;

    if (emptyEl) emptyEl.style.display = 'none';

    body.appendChild(buildEntryEl(entry));

    // Prune DOM nodes beyond MAX_LOG_ENTRIES
    var entries = body.querySelectorAll('.log-entry');
    while (entries.length > MAX_LOG_ENTRIES) {
      body.removeChild(entries[0]);
      entries = body.querySelectorAll('.log-entry');
    }

    // Update console unread issue count badge
    var countBadge = el('consoleBadge');
    if (countBadge) {
      countBadge.textContent = _logEntries.length + (_logEntries.length === 1 ? ' issue' : ' issues');
      countBadge.style.background = 'rgba(244, 63, 94, 0.25)';
      countBadge.style.color = '#fb7185';
    }

    if (!_userScrolled) { scrollConsoleToBottom(); }

    // Auto-expand the console on ERROR or FATAL (if collapsed, never on WARN)
    if (entry.level === 'ERROR' || entry.level === 'FATAL') {
      var section = el('consoleSection');
      var header = el('consoleHeader');
      if (section && !section.classList.contains('open')) {
        section.classList.add('open');
        header && header.setAttribute('aria-expanded', 'true');
      }
    }
  }

  // ── session management & transport ─────────────────────────────────────────

  var SESSION_STORAGE_KEY = 'oa_fsm_session_id';
  var _sessionId = sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (!_sessionId) {
    _sessionId = 'op_' + Math.random().toString(36).substring(2, 10);
    sessionStorage.setItem(SESSION_STORAGE_KEY, _sessionId);
  }
  var _isController = false;
  var _userLastActive = Date.now();
  var _socket = null;

  function post(path, body) {
    return fetch(path, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': _sessionId
      },
      body: JSON.stringify(body || {})
    }).then(function (r) { return r.json().catch(function () { return {}; }); });
  }

  function recordUserActivity() {
    _userLastActive = Date.now();
    var toast = el('afkWarningToast');
    if (toast) { toast.style.display = 'none'; }
  }

  function handleSessionState(state) {
    if (!state) return;
    var ctrlId = state.controller_id;
    var isMe = (ctrlId === _sessionId);
    _isController = isMe;

    var overlay = el('fsmViewOnlyOverlay');
    var ctrlPill = el('sessionStatus');
    var ctrlDot = el('sessionDot');
    var relBtn = el('btnReleaseControl');
    var overlayCtrl = el('overlayCtrlId');
    var overlayQueue = el('overlayQueuePos');

    if (isMe) {
      if (overlay) {
        overlay.style.display = 'none';
        overlay.setAttribute('aria-hidden', 'true');
      }
      if (ctrlPill) { ctrlPill.textContent = 'Control: You'; }
      if (ctrlDot) { ctrlDot.style.background = '#34d399'; }
      if (relBtn) { relBtn.style.display = 'inline-block'; }
    } else {
      if (overlay) {
        overlay.style.display = 'flex';
        overlay.setAttribute('aria-hidden', 'false');
      }
      var shortCtrl = ctrlId ? ctrlId.substring(0, 8) : 'None';
      if (ctrlPill) { ctrlPill.textContent = 'Control: ' + shortCtrl; }
      if (ctrlDot) { ctrlDot.style.background = '#f59e0b'; }
      if (relBtn) { relBtn.style.display = 'none'; }
      if (overlayCtrl) { overlayCtrl.textContent = 'Operator [' + shortCtrl + ']'; }

      var q = state.queue || [];
      var myIdx = q.indexOf(_sessionId);
      if (overlayQueue) {
        if (myIdx >= 0) {
          overlayQueue.textContent = '#' + (myIdx + 1) + ' in line';
        } else {
          overlayQueue.textContent = 'Spectating';
        }
      }
    }

    if (lastState) {
      render(lastState);
    }
  }

  function initSessionTracking(socket) {
    _socket = socket;
    ['pointermove', 'mousedown', 'keydown', 'touchstart', 'scroll'].forEach(function (ev) {
      window.addEventListener(ev, recordUserActivity, { passive: true });
    });

    socket.emit('fsm_session_join', { session_id: _sessionId });
    socket.on('fsm_session_state', handleSessionState);

    fetch('/api/fsm/session_state').then(function (r) { return r.json(); }).then(function (res) {
      if (res && res.success && res.state) {
        handleSessionState(res.state);
      }
    }).catch(function () {});

    setInterval(function () {
      if (socket && socket.connected) {
        var userActive = (Date.now() - _userLastActive < 30000);
        socket.emit('fsm_session_heartbeat', {
          session_id: _sessionId,
          user_active: userActive
        });
      }
    }, 10000);

    setInterval(function () {
      if (!_isController) {
        var toast = el('afkWarningToast');
        if (toast) toast.style.display = 'none';
        return;
      }
      var idleMs = Date.now() - _userLastActive;
      var remainingMs = 300000 - idleMs;
      var toast = el('afkWarningToast');
      var countdown = el('afkCountdown');

      if (remainingMs > 0 && remainingMs <= 30000) {
        if (toast) toast.style.display = 'flex';
        if (countdown) countdown.textContent = Math.ceil(remainingMs / 1000);
      } else {
        if (toast) toast.style.display = 'none';
      }
    }, 1000);

    var leaveBtn = el('btnLeaveQueue');
    if (leaveBtn) {
      leaveBtn.onclick = function () {
        if (socket && socket.connected) {
          socket.emit('fsm_session_leave', { session_id: _sessionId });
        }
        window.location.href = '/';
      };
    }

    var relBtn = el('btnReleaseControl');
    if (relBtn) {
      relBtn.onclick = function () {
        if (confirm('Yield robot control to the next operator in queue?')) {
          post('/api/fsm/release_control').then(function (res) {
            if (res && res.message) {
              el('runMsg').textContent = res.message;
            }
          });
        }
      };
    }

    var keepBtn = el('btnKeepControl');
    if (keepBtn) {
      keepBtn.onclick = function () {
        recordUserActivity();
      };
    }

    window.addEventListener('beforeunload', function () {
      if (socket && socket.connected) {
        socket.emit('fsm_session_leave', { session_id: _sessionId });
      }
    });
  }

  function command(name) {
    post('/api/fsm/command', { command: name }).then(function (res) {
      if (res && res.message) {
        el('runMsg').textContent = res.message;
      } else if (!res.success) {
        el('runMsg').textContent = 'refused';
      }
    });
  }

  function setConn(ok, text) {
    var badge = el('conn');
    if (badge) {
      badge.textContent = text;
      badge.className = 'badge ' + (ok ? 'live' : 'dead');
    }
    var connWrapper = el('conn-wrapper');
    if (connWrapper) {
      connWrapper.className = 'pill ' + (ok ? 'pill-live' : 'pill-dead');
    }
    // Also update unified nav badge
    if (window.oaNav) { window.oaNav.setBadge(ok, text); }
  }

  function loadSequences() {
    fetch('/api/sequences').then(function (r) { return r.json(); }).then(function (res) {
      var picker = el('seqPicker');
      var names = res.success ? res.sequences.map(function (s) { return s.name; }) : [];
      return fetch('/api/actions').then(function (r) { return r.json(); }).then(function (act) {
        if (act.success) {
          act.actions.forEach(function (a) { names.push(a.name); });
        }
        picker.innerHTML = '';
        names.forEach(function (name) {
          var option = document.createElement('option');
          option.value = name;
          option.textContent = name;
          picker.appendChild(option);
        });
        if (!names.length) {
          picker.innerHTML = '<option value="">no sequences found</option>';
        }
      });
    }).catch(function () { /* picker stays empty */ });
  }

  function start() {
    fetch('/api/fsm/graph').then(function (r) { return r.json(); }).then(function (res) {
      (res.graph.layers || []).forEach(function (spec) {
        var svg = el(spec.id === 'robot' ? 'robotSvg' : 'sequenceSvg');
        if (svg) { layers[spec.id] = drawLayer(svg, spec); }
      });
      return fetch('/api/fsm/state');
    }).then(function (r) { return r.json(); }).then(function (res) {
      if (res.success) { setConn(true, 'live'); render(res.state); }
      else { setConn(false, res.message || 'executor not reachable'); }
    }).catch(function (err) {
      setConn(false, 'API unreachable: ' + err.message);
    });

    loadSequences();
    initConsole();

    var socket = io();
    socket.on('connect',    function () {
      setConn(true,  'live');
      socket.emit('fsm_session_join', { session_id: _sessionId });
    });
    socket.on('disconnect', function () { setConn(false, 'socket lost'); });
    socket.on('fsm_state',  render);

    initSessionTracking(socket);

    // ── Real-time log events from LogCollector ──
    socket.on('log_event', function (entry) {
      addLogEntry(entry);
    });

    // Wire up nav badge
    if (window.oaNav) { window.oaNav.setSocket(socket); }

    if (el('btnPause')) { el('btnPause').onclick = function () { command('pause'); }; }
    if (el('btnResume')) { el('btnResume').onclick = function () { command('resume'); }; }
    if (el('btnClear')) { el('btnClear').onclick = function () { command('clear_fault'); }; }
    if (el('btnStop')) {
      el('btnStop').onclick = function () { command('stop'); };
    }
    if (el('btnEstop')) {
      el('btnEstop').onclick = function () { command('stop'); };
    }

    var motorPill = el('fsm-motor-pill');
    function triggerMotorEnable() {
      if (!_isController) {
        el('runMsg').textContent = 'Spectator mode: control lease required to enable motors';
        return;
      }
      if (lastState && lastState.motors_enabled === false) {
        el('runMsg').textContent = 'Enabling motors…';
        command('enable');
      } else {
        el('runMsg').textContent = 'Motors are already enabled';
      }
    }
    if (motorPill) {
      motorPill.onclick = triggerMotorEnable;
      motorPill.onkeydown = function (e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          triggerMotorEnable();
        }
      };
    }

    var btnAbortHome = el('btnAbortHome');
    if (btnAbortHome) {
      btnAbortHome.onclick = function () {
        btnAbortHome.disabled = true;
        var prevText = btnAbortHome.textContent;
        btnAbortHome.textContent = 'Aborting ➔ Home...';
        el('runMsg').textContent = 'Issuing abort & returning to home posture...';
        post('/api/fsm/abort_home').then(function (res) {
          btnAbortHome.disabled = false;
          btnAbortHome.textContent = prevText;
          if (res && res.message) {
            el('runMsg').textContent = res.message;
          }
        }).catch(function (err) {
          btnAbortHome.disabled = false;
          btnAbortHome.textContent = prevText;
          el('runMsg').textContent = 'Abort failed: ' + (err.message || err);
        });
      };
    }

    function run(dry) {
      var name = el('seqPicker').value;
      if (!name) { return; }
      el('runMsg').textContent = 'starting ' + name + '…';
      post('/api/sequence/run', { name: name, dry_run: dry }).then(function (res) {
        el('runMsg').textContent = res.message || '';
      });
    }
    el('btnRun').onclick = function () { run(false); };
    el('btnDry').onclick = function () { run(true); };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
