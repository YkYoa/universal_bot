"""Unit tests for tls_cert.py - the self-signed cert helper behind
mock_server.py's --https flag. No ROS, no hardware; the "generates a real
cert" test shells out to the system openssl and is skipped if it's absent
(e.g. a minimal CI image) rather than failing, matching this module's own
never-crash-the-mockup-over-TLS philosophy."""
import shutil
import subprocess

import pytest

from gesture_teleop import tls_cert


def test_reuses_existing_pair_without_invoking_openssl(tmp_path, monkeypatch):
    cert_dir = tmp_path / "tls"
    cert_dir.mkdir()
    cert_path = cert_dir / tls_cert.CERT_FILENAME
    key_path = cert_dir / tls_cert.KEY_FILENAME
    cert_path.write_text("existing cert")
    key_path.write_text("existing key")

    def _boom(*args, **kwargs):
        raise AssertionError("openssl should not be invoked when a pair already exists")

    monkeypatch.setattr(subprocess, "run", _boom)

    result = tls_cert.ensure_self_signed_cert(cert_dir)
    assert result == (str(cert_path), str(key_path))


def test_returns_none_without_openssl_on_path(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    result = tls_cert.ensure_self_signed_cert(tmp_path / "tls")
    assert result is None


def test_returns_none_and_cleans_up_on_openssl_failure(tmp_path, monkeypatch):
    def _fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "openssl")

    monkeypatch.setattr(subprocess, "run", _fail)
    cert_dir = tmp_path / "tls"

    result = tls_cert.ensure_self_signed_cert(cert_dir)
    assert result is None
    assert not (cert_dir / tls_cert.CERT_FILENAME).exists()
    assert not (cert_dir / tls_cert.KEY_FILENAME).exists()


def test_returns_none_on_unwritable_cert_dir(tmp_path, monkeypatch):
    # A file where the cert dir should be - mkdir(parents=True) on top of
    # an existing regular file always raises FileExistsError (an OSError
    # subclass), on every platform, which is exactly the "can't create the
    # directory" case this guards - no os.chmod/permission-bit dependence
    # needed, so this passes on Windows too.
    blocker = tmp_path / "tls"
    blocker.write_text("not a directory")

    result = tls_cert.ensure_self_signed_cert(blocker)
    assert result is None


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl not on PATH")
def test_generates_a_real_self_signed_cert(tmp_path):
    cert_dir = tmp_path / "tls"
    result = tls_cert.ensure_self_signed_cert(cert_dir, common_name="test-openarm-gesture")
    assert result is not None
    cert_path, key_path = result
    assert (cert_dir / tls_cert.CERT_FILENAME).is_file()
    assert (cert_dir / tls_cert.KEY_FILENAME).is_file()

    # Cross-check with openssl itself rather than trusting our own writer.
    out = subprocess.run(
        ["openssl", "x509", "-in", cert_path, "-noout", "-subject", "-ext", "subjectAltName"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "test-openarm-gesture" in out
    assert "DNS:localhost" in out
    assert "IP Address:127.0.0.1" in out


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl not on PATH")
def test_second_call_reuses_the_first_certs_bytes(tmp_path):
    cert_dir = tmp_path / "tls"
    first = tls_cert.ensure_self_signed_cert(cert_dir)
    first_bytes = (cert_dir / tls_cert.CERT_FILENAME).read_bytes()

    second = tls_cert.ensure_self_signed_cert(cert_dir)
    second_bytes = (cert_dir / tls_cert.CERT_FILENAME).read_bytes()

    assert first == second
    assert first_bytes == second_bytes


def test_local_lan_ip_returns_a_string_or_none():
    result = tls_cert.local_lan_ip()
    assert result is None or isinstance(result, str)
