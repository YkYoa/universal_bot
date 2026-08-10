/* FSM viewer.
 *
 * Draws both layers of the state machine from /api/fsm/graph and lights up
 * whichever node the robot is in, following `fsm_state` socket events.
 *
 * Deliberately plain: no framework, no build step, no graph library, and the
 * only script it loads is the socket.io client already vendored for the 3D
 * dashboard. It is served off the robot to a phone on the same LAN, so it has
 * to work with no internet and no npm.
 *
 * app.js is a separate, minified bundle - this file is intentionally not part
 * of it.
 */
(function () {
  'use strict';

  var KIND_COLOR = {
    start: '#4a5058', normal: '#4a5058', active: '#3a7bd5',
    success: '#2ecc71', warning: '#e2b93b', error: '#d9534f', special: '#9b6dd6'
  };

  var el = function (id) { return document.getElementById(id); };
  var layers = {};        // layer id -> {spec, nodes: {id -> {circle, label}}}
  var lastState = null;
  var lastNodeByLayer = {};

  // ── layout ────────────────────────────────────────────────────────────────
  //
  // Nodes go on a circle, edges are chords. With seven and nine states the
  // circle stays legible, every edge is a straight line, and the layout is
  // fully determined by the node order in fsm_graph.json - so the diagram
  // looks the same on every load and after every redeploy.

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
      // Trim the line to the circle edges so the arrowhead lands on the rim
      // rather than under the node.
      var dx = b.x - a.x, dy = b.y - a.y;
      var len = Math.hypot(dx, dy) || 1;
      var ux = dx / len, uy = dy / len;
      // Offset perpendicular so A->B and B->A do not draw on top of each other.
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
      node.circle.setAttribute('fill', on ? node.color : '#23262c');
      node.circle.setAttribute('stroke-width', on ? 3 : 1.5);
      node.circle.setAttribute('opacity', on || !activeId ? 1 : 0.55);
    });

    // Briefly brighten the edge just traversed, so a fast transition is still
    // visible rather than only showing up as the destination lighting.
    var previous = lastNodeByLayer[layerId];
    if (previous && activeId && previous !== activeId) {
      var edge = layer.edges[previous + '>' + activeId];
      if (edge) {
        edge.setAttribute('stroke', 'rgba(255,255,255,0.75)');
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

    el('mode').textContent = 'control mode: ' + (state.control_mode_active || '?');
    el('seq').textContent = state.sequence_name
      ? state.sequence_name + (state.loop_total === -1
          ? ' (loop ' + (state.loop_index + 1) + ', forever)'
          : ' (loop ' + (state.loop_index + 1) + '/' + state.loop_total + ')')
      : 'no sequence';

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
    var faulted = state.robot_state === 'FAULT' || state.robot_state === 'ESTOP';
    el('btnPause').disabled = !running;
    el('btnResume').disabled = !paused;
    el('btnStep').disabled = !paused;
    el('btnCancel').disabled = !(running || paused);
    el('btnClear').disabled = !faulted;
    el('btnRun').disabled = running || paused || faulted;
    el('btnDry').disabled = running || paused || faulted;

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

  // ── transport ─────────────────────────────────────────────────────────────

  function post(path, body) {
    return fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {})
    }).then(function (r) { return r.json().catch(function () { return {}; }); });
  }

  function command(name) {
    post('/api/fsm/command', { command: name }).then(function (res) {
      if (!res.success) { el('runMsg').textContent = res.message || 'refused'; }
    });
  }

  function setConn(ok, text) {
    var badge = el('conn');
    badge.textContent = text;
    badge.className = 'badge ' + (ok ? 'live' : 'dead');
  }

  function loadSequences() {
    // The CRUD endpoints come from the project package's blueprint, which a
    // workspace may not have - fall back to the builtin actions, which the
    // generic server always exposes.
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
    }).catch(function () { /* the picker just stays empty */ });
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

    var socket = io();
    socket.on('connect', function () { setConn(true, 'live'); });
    socket.on('disconnect', function () { setConn(false, 'socket lost'); });
    socket.on('fsm_state', render);

    el('btnPause').onclick = function () { command('pause'); };
    el('btnResume').onclick = function () { command('resume'); };
    el('btnStep').onclick = function () { command('step'); };
    el('btnCancel').onclick = function () { command('cancel'); };
    el('btnClear').onclick = function () { command('clear_fault'); };
    el('btnEstop').onclick = function () { command('estop'); };

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
