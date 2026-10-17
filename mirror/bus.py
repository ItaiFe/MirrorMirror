"""Face bus: plain UDP on 127.0.0.1 carrying (State, mouth, viseme) messages as JSON.

The viseme is a mouth-shape letter (see mirror.visemes); a message without one
(an older sender) means "X", the resting mouth.
"""
import json
import socket

from mirror.states import State

HOST = "127.0.0.1"


class FaceBus:
    def __init__(self, sock: socket.socket, port: int):
        self._sock = sock
        self._port = port

    @classmethod
    def sender(cls, port: int) -> "FaceBus":
        return cls(socket.socket(socket.AF_INET, socket.SOCK_DGRAM), port)

    @classmethod
    def receiver(cls, port: int) -> "FaceBus":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((HOST, port))
        sock.setblocking(False)
        return cls(sock, port)

    def send(self, state: State, mouth: float, viseme: str = "X") -> None:
        msg = json.dumps({"s": State(state).value, "m": float(mouth), "v": str(viseme)}).encode()
        self._sock.sendto(msg, (HOST, self._port))

    def poll(self):
        """Drain queued messages; return the latest (State, float, str) or None."""
        latest = None
        while True:
            try:
                data, _ = self._sock.recvfrom(1024)
            except BlockingIOError:
                break
            try:
                d = json.loads(data)
                viseme = d.get("v", "X")
                if not isinstance(viseme, str):
                    raise TypeError("viseme must be a string")
                latest = (State(d["s"]), float(d["m"]), viseme)
            except (ValueError, KeyError, TypeError, AttributeError):
                continue  # ignore malformed datagrams
        return latest

    def close(self) -> None:
        self._sock.close()
