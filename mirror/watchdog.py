"""systemd watchdog: sd_notify over $NOTIFY_SOCKET, stdlib only.

A no-op (returns False) when NOTIFY_SOCKET is unset, e.g. on the Mac. Under
systemd (Type=notify, WatchdogSec=...) the process calls ready() once it is up
and ping() from its main loop; a loop that stops pinging is restarted.
"""
import os
import socket
import time

DEFAULT_WATCHDOG_USEC = 60_000_000
_last_ping = None  # time.monotonic() of the last ping that was sent


def notify(msg: str) -> bool:
    """Send msg to $NOTIFY_SOCKET (a path, or an abstract name starting with '@').
    False if the variable is unset or the send failed; never raises."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.send(msg.encode())
        return True
    except OSError:
        return False


def ready() -> bool:
    return notify("READY=1")


def ping_interval_s() -> float:
    """A quarter of the watchdog timeout (WATCHDOG_USEC), 15 s by default."""
    try:
        usec = int(os.environ["WATCHDOG_USEC"])
    except (KeyError, ValueError):
        usec = DEFAULT_WATCHDOG_USEC
    return (usec if usec > 0 else DEFAULT_WATCHDOG_USEC) / 4 / 1e6


def ping() -> bool:
    """Send WATCHDOG=1, at most once per ping_interval_s(); True only if it was sent."""
    global _last_ping
    now = time.monotonic()
    if _last_ping is not None and now - _last_ping < ping_interval_s():
        return False
    if not notify("WATCHDOG=1"):
        return False
    _last_ping = now
    return True
