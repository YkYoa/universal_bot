"""Self-signed TLS cert for the gesture mockup's optional HTTPS listener.

Why this exists at all: `navigator.mediaDevices` (and therefore
`getUserMedia`) does not exist in an insecure context - Chrome/Firefox/Edge
only expose the camera API on `https://` origins or `http://localhost`.
This mockup is reached over the LAN by IP (`http://<robot-ip>:5055/`),
which is an insecure context, so `navigator.mediaDevices` is `undefined`
there and gesture.js's `getUserMedia` call throws before the browser ever
shows a permission prompt - not a JS bug, a browser security requirement.
Serving over HTTPS is what actually unlocks the API; a self-signed cert
(with the one-time "unsafe, proceed anyway" browser warning that is
unavoidable for any unregistered private-LAN device) is the only kind of
cert a robot on a local network can have.

This module never raises: any failure (no `openssl` on PATH, a locked-down
cert dir, an unexpected non-zero exit) returns None so the caller can fall
back to plain HTTP rather than take the whole mockup down over a TLS
nicety that was never a hard requirement of "nothing real, only mockup".
"""
from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Tuple

CERT_FILENAME = "cert.pem"
KEY_FILENAME = "key.pem"


def local_lan_ip() -> Optional[str]:
    """Best-effort primary LAN IP, so the generated cert's SAN list
    matches the address people actually type into the browser. Uses the
    connect-a-UDP-socket trick (no packet is actually sent - UDP `connect`
    only picks a route/local address) so it works with no external
    connectivity, e.g. an isolated robot LAN."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def ensure_self_signed_cert(
    cert_dir: Path,
    common_name: str = "openarm-gesture",
    extra_sans: Optional[Sequence[str]] = None,
) -> Optional[Tuple[str, str]]:
    """Returns (cert_path, key_path) as strings. Reuses an existing pair
    under cert_dir untouched (so the browser's trust exception survives a
    service restart - a fresh cert every restart would force re-trusting
    it every time), generating a new 10-year self-signed RSA-2048 pair
    only when neither file exists yet. Returns None on any failure."""
    cert_path = cert_dir / CERT_FILENAME
    key_path = cert_dir / KEY_FILENAME
    if cert_path.is_file() and key_path.is_file():
        return str(cert_path), str(key_path)

    if shutil.which("openssl") is None:
        return None

    try:
        cert_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    sans = ["DNS:localhost", "IP:127.0.0.1"]
    lan_ip = local_lan_ip()
    if lan_ip:
        sans.append(f"IP:{lan_ip}")
    for extra in extra_sans or ():
        if extra not in sans:
            sans.append(extra)

    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-days", "3650",
        "-keyout", str(key_path),
        "-out", str(cert_path),
        "-subj", f"/CN={common_name}",
        "-addext", f"subjectAltName={','.join(sans)}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        # Clean up a possible half-written pair so the next attempt (or a
        # human) doesn't find a cert with no matching key or vice versa.
        cert_path.unlink(missing_ok=True)
        key_path.unlink(missing_ok=True)
        return None

    try:
        key_path.chmod(0o600)
    except OSError:
        pass

    if not (cert_path.is_file() and key_path.is_file()):
        return None
    return str(cert_path), str(key_path)
