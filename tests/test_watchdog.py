import shutil
import socket
import tempfile

import pytest

from mirror import watchdog


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(watchdog, "_last_ping", None)
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    monkeypatch.delenv("WATCHDOG_USEC", raising=False)


@pytest.fixture
def sock(monkeypatch):
    d = tempfile.mkdtemp(prefix="wd", dir="/tmp")  # AF_UNIX paths are short on macOS
    path = d + "/n.sock"
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    s.bind(path)
    s.settimeout(1)
    monkeypatch.setenv("NOTIFY_SOCKET", path)
    yield s
    s.close()
    shutil.rmtree(d, ignore_errors=True)


def test_ready_and_ping_reach_the_socket(sock):
    assert watchdog.ready() is True
    assert sock.recv(64) == b"READY=1"
    assert watchdog.ping() is True
    assert sock.recv(64) == b"WATCHDOG=1"


def test_abstract_socket(monkeypatch):
    name = "mirror-test-watchdog-abstract"
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        try:
            s.bind("\0" + name)
        except OSError:
            pytest.skip("no abstract unix sockets here")
        s.settimeout(1)
        monkeypatch.setenv("NOTIFY_SOCKET", "@" + name)
        assert watchdog.ready() is True
        assert s.recv(64) == b"READY=1"
    finally:
        s.close()


def test_unset_env_is_a_noop():
    assert watchdog.notify("READY=1") is False
    assert watchdog.ready() is False
    assert watchdog.ping() is False


def test_dead_socket_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTIFY_SOCKET", str(tmp_path / "missing.sock"))
    assert watchdog.ready() is False


def test_ping_is_rate_limited(sock, monkeypatch):
    monkeypatch.setenv("WATCHDOG_USEC", "8000000")  # -> every 2 s
    now = [100.0]
    monkeypatch.setattr(watchdog.time, "monotonic", lambda: now[0])
    assert watchdog.ping_interval_s() == 2.0
    assert watchdog.ping() is True
    now[0] += 1.9
    assert watchdog.ping() is False
    now[0] += 0.2
    assert watchdog.ping() is True
    sock.recv(64)
    sock.recv(64)
    with pytest.raises(socket.timeout):
        sock.settimeout(0.05)
        sock.recv(64)


def test_default_interval_is_15s():
    assert watchdog.ping_interval_s() == 15.0
