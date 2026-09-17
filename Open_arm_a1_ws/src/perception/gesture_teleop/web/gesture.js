// Gesture teleop mockup client. Deliberately plain: no framework, no build
// step - same precedent as web_visualizer/fsm.js (has to run offline on a
// phone on the same LAN). This page NEVER commands the real robot; it only
// shows the camera feed, the MediaPipe bone overlay, and a drawn mock of
// the retargeted arm - all computed server-side by mock_server.py using
// the exact same gesture_teleop.core algorithm the real teleop_node uses,
// so the overlay and the arm mock can never disagree with production
// logic (see the plan: "never reimplement FK in JS").
(function () {
  "use strict";

  const SEND_WIDTH = 320;
  const SEND_HEIGHT = 240;
  const JPEG_QUALITY = 0.8;
  const HEARTBEAT_INTERVAL_MS = 100; // well under the server's 150ms dead-man timeout

  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const overlayCtx = overlay.getContext("2d");
  const armCanvas = document.getElementById("arm-canvas");
  const armCtx = armCanvas.getContext("2d");
  const connStatus = document.getElementById("conn-status");
  const cameraOverlay = document.getElementById("camera-overlay");
  const cameraBtn = document.getElementById("camera-btn");
  const cameraMessage = document.getElementById("camera-message");
  const engageBtn = document.getElementById("engage-btn");
  const mirrorSelect = document.getElementById("mirror-mode");
  const stateLeftEl = document.getElementById("state-left");
  const stateRightEl = document.getElementById("state-right");
  const latencyEl = document.getElementById("latency");
  const rejectReasonsEl = document.getElementById("reject-reasons");

  const sendCanvas = document.createElement("canvas");
  sendCanvas.width = SEND_WIDTH;
  sendCanvas.height = SEND_HEIGHT;
  const sendCtx = sendCanvas.getContext("2d");

  let inFlight = false; // client-side 1-in-flight pacing (see plan's transport guidance)
  let lastSendTime = 0;
  let engaged = false;
  let heartbeatTimer = null;

  // -- Socket.IO -----------------------------------------------------------

  const socket = io({ transports: ["websocket"] });

  socket.on("connect", () => {
    connStatus.textContent = "connected";
    connStatus.className = "conn-status ok";
  });
  socket.on("disconnect", () => {
    connStatus.textContent = "disconnected";
    connStatus.className = "conn-status bad";
    setEngaged(false);
  });
  socket.on("error", (data) => {
    rejectReasonsEl.textContent = "server: " + (data && data.message ? data.message : "unknown error");
  });

  socket.on("result", (payload) => {
    inFlight = false;
    const latency = Math.round(performance.now() - lastSendTime);
    latencyEl.textContent = latency + " ms";

    drawSkeleton(payload.skeleton);
    drawArms(payload.arms);
    updateState(payload.state, payload.arms);
  });

  socket.on("engage_result", (result) => {
    const reasons = [];
    if (!result.left.ok && result.left.reason) reasons.push("left: " + result.left.reason);
    if (!result.right.ok && result.right.reason) reasons.push("right: " + result.right.reason);
    rejectReasonsEl.textContent = reasons.join(" | ");
  });

  // -- Camera capture + send loop -------------------------------------------
  //
  // Deliberately NOT auto-started on page load:
  //  1. navigator.mediaDevices (and therefore getUserMedia) only exists in
  //     a secure context - https:// or http://localhost. Reached by LAN IP
  //     over plain http://, the whole API is undefined - not a permission
  //     problem, a browser policy one - so an unconditional auto-call would
  //     throw immediately with zero visible feedback beyond a small header
  //     badge (the exact bad UX this replaces). Check window.isSecureContext
  //     up front and say so, with a link to the https:// version if the
  //     server offers one (see mock_server.py's --https flag).
  //  2. Some browsers (notably Safari/iOS) refuse getUserMedia unless it is
  //     called from within a user-gesture handler (a click), so it can
  //     never be reliably auto-started anyway.
  //  3. An explicit button also gives "permission denied" a working retry
  //     path, instead of a dead page the user has to reload.

  function showCameraOverlay(message, isError) {
    cameraOverlay.hidden = false;
    cameraMessage.textContent = message || "";
    cameraMessage.className = "camera-message" + (isError ? " error" : "");
  }

  function hideCameraOverlay() {
    cameraOverlay.hidden = true;
  }

  function httpsUrl() {
    if (location.protocol === "https:") return null;
    return `https://${location.hostname}:${location.port || 443}${location.pathname}`;
  }

  function cameraApiUnavailableMessage() {
    const upgrade = httpsUrl();
    if (!upgrade) {
      return "This browser has no camera API available.";
    }
    return `Your browser only allows camera access on a secure (HTTPS) connection. `
      + `Reload this page at ${upgrade} and accept the one-time "not trusted" `
      + `warning (expected for a self-signed LAN certificate) - see mock_server.py --https.`;
  }

  function friendlyCameraError(err) {
    switch (err && err.name) {
      case "NotAllowedError":
      case "SecurityError":
        return "Camera permission was denied. Click Enable Camera and allow access when your browser asks.";
      case "NotFoundError":
      case "OverconstrainedError":
        return "No camera was found on this device.";
      case "NotReadableError":
        return "The camera is already in use by another application.";
      default:
        return "Could not access the camera: " + (err && err.message ? err.message : String(err));
    }
  }

  async function startCamera() {
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showCameraOverlay(cameraApiUnavailableMessage(), true);
      cameraBtn.hidden = true;
      return;
    }

    cameraBtn.disabled = true;
    cameraBtn.textContent = "Requesting camera…";
    showCameraOverlay("Requesting camera access…", false);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      video.addEventListener("loadedmetadata", () => {
        overlay.width = video.videoWidth || 640;
        overlay.height = video.videoHeight || 480;
      }, { once: true });
      hideCameraOverlay();
      requestAnimationFrame(sendLoop);
    } catch (err) {
      showCameraOverlay(friendlyCameraError(err), true);
      cameraBtn.hidden = false;
      cameraBtn.disabled = false;
      cameraBtn.textContent = "Try Again";
    }
  }

  function initCameraUI() {
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showCameraOverlay(cameraApiUnavailableMessage(), true);
      cameraBtn.hidden = true;
      return;
    }
    showCameraOverlay("Your webcam is only used on this page - it is sent to this server's own process and nowhere else.", false);
    cameraBtn.hidden = false;
  }

  cameraBtn.addEventListener("click", startCamera);

  function sendLoop() {
    requestAnimationFrame(sendLoop);
    if (inFlight || !socket.connected || video.readyState < 2) return;

    sendCtx.drawImage(video, 0, 0, SEND_WIDTH, SEND_HEIGHT);
    sendCanvas.toBlob(
      (blob) => {
        if (!blob) return;
        blob.arrayBuffer().then((buf) => {
          inFlight = true;
          lastSendTime = performance.now();
          socket.emit("frame", buf);
        });
      },
      "image/jpeg",
      JPEG_QUALITY
    );
  }

  // -- Bone-segment overlay (never a bounding box - see the plan) ----------

  function drawSkeleton(skeleton) {
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    if (!skeleton || !skeleton.landmarks) return;
    const lm = skeleton.landmarks;
    const w = overlay.width;
    const h = overlay.height;

    overlayCtx.strokeStyle = "#38bdf8";
    overlayCtx.lineWidth = 3;
    overlayCtx.lineCap = "round";
    for (const [a, b] of skeleton.connections) {
      const pa = lm[a];
      const pb = lm[b];
      if (!pa || !pb) continue;
      overlayCtx.beginPath();
      overlayCtx.moveTo(pa.x * w, pa.y * h);
      overlayCtx.lineTo(pb.x * w, pb.y * h);
      overlayCtx.globalAlpha = Math.min(pa.visibility, pb.visibility);
      overlayCtx.stroke();
    }
    overlayCtx.globalAlpha = 1;

    overlayCtx.fillStyle = "#fbbf24";
    for (const key in lm) {
      const p = lm[key];
      overlayCtx.beginPath();
      overlayCtx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
      overlayCtx.globalAlpha = p.visibility;
      overlayCtx.fill();
    }
    overlayCtx.globalAlpha = 1;
  }

  // -- Drawn robot arm mock (orthographic projection of server-computed FK) -

  // Simple top-down-ish orthographic projection: robot arm-base frame is
  // (x=forward, y=left, z=up); project onto screen (y_left -> screen x,
  // z_up -> screen y), which reads naturally for a person facing the robot.
  function projectPoint(p, cx, cy, scale) {
    return [cx - p[1] * scale, cy - p[2] * scale];
  }

  function drawArms(arms) {
    const w = armCanvas.width;
    const h = armCanvas.height;
    armCtx.clearRect(0, 0, w, h);
    armCtx.fillStyle = "#0f172a";
    armCtx.fillRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h * 0.25;
    const scale = w * 0.55; // arm reach is ~0.32m; leaves margin for a 2x reach picture

    // Torso reference blob.
    armCtx.fillStyle = "#1e293b";
    armCtx.beginPath();
    armCtx.arc(cx, cy, 10, 0, Math.PI * 2);
    armCtx.fill();

    drawOneArm(arms.left, "#38bdf8", cx, cy, scale);
    drawOneArm(arms.right, "#f472b6", cx, cy, scale);
  }

  function drawOneArm(arm, color, cx, cy, scale) {
    if (!arm || !arm.points || arm.points.length === 0) return;
    armCtx.strokeStyle = color;
    armCtx.fillStyle = color;
    armCtx.lineWidth = arm.valid ? 4 : 2;
    armCtx.globalAlpha = arm.valid ? 1.0 : 0.35; // dim when not actively engaged/valid

    armCtx.beginPath();
    arm.points.forEach((p, i) => {
      const [x, y] = projectPoint(p, cx, cy, scale);
      if (i === 0) armCtx.moveTo(x, y);
      else armCtx.lineTo(x, y);
    });
    armCtx.stroke();

    arm.points.forEach((p) => {
      const [x, y] = projectPoint(p, cx, cy, scale);
      armCtx.beginPath();
      armCtx.arc(x, y, 4, 0, Math.PI * 2);
      armCtx.fill();
    });
    armCtx.globalAlpha = 1;
  }

  // -- Status tiles ----------------------------------------------------------

  function stateClass(state) {
    if (state === "ENGAGED") return "engaged";
    if (state === "HOLD") return "hold";
    return "idle";
  }

  function updateState(state, arms) {
    stateLeftEl.textContent = state.left;
    stateLeftEl.className = "status-value " + stateClass(state.left);
    stateRightEl.textContent = state.right;
    stateRightEl.className = "status-value " + stateClass(state.right);

    const reasons = [];
    if (arms.left.reason && arms.left.reason !== "none") reasons.push("left: " + arms.left.reason);
    if (arms.right.reason && arms.right.reason !== "none") reasons.push("right: " + arms.right.reason);
    rejectReasonsEl.textContent = reasons.join(" | ");
  }

  // -- Engage dead-man control -----------------------------------------------
  //
  // A key-hold alone is NOT a dead-man switch (plan interlock 3): this
  // sends a heartbeat on an interval while held, and disengages
  // immediately on blur/visibilitychange/pagehide, so a stuck pointerup
  // or a lost event can never leave the mock "engaged" indefinitely.

  function setEngaged(next) {
    if (next === engaged) return;
    engaged = next;
    engageBtn.classList.toggle("active", engaged);
    engageBtn.textContent = engaged ? "Engaged (release to stop)" : "Hold to Engage";
    if (engaged) {
      socket.emit("engage", {});
      heartbeatTimer = setInterval(() => socket.emit("heartbeat", {}), HEARTBEAT_INTERVAL_MS);
    } else {
      socket.emit("disengage", {});
      if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
      }
    }
  }

  engageBtn.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    setEngaged(true);
  });
  ["pointerup", "pointerleave", "pointercancel"].forEach((evt) =>
    engageBtn.addEventListener(evt, () => setEngaged(false))
  );
  window.addEventListener("blur", () => setEngaged(false));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) setEngaged(false);
  });
  window.addEventListener("pagehide", () => setEngaged(false));

  mirrorSelect.addEventListener("change", () => {
    socket.emit("set_mirror_mode", { mode: mirrorSelect.value });
  });

  initCameraUI();
})();
