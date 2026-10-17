"""Face bus: plain UDP on 127.0.0.1 carrying (State, mouth) messages as JSON."""
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

    def send(self, state: State, mouth: float) -> None:
        msg = json.dumps({"s": State(state).value, "m": float(mouth)}).encode()
        self._sock.sendto(msg, (HOST, self._port))

    def poll(self):
        """Drain queued messages; return the latest (State, float) or None."""
        latest = None
        while True:
            try:
                data, _ = self._sock.recvfrom(1024)
            except BlockingIOError:
                break
            try:
                d = json.loads(data)
                latest = (State(d["s"]), float(d["m"]))
            except (ValueError, KeyError, TypeError):
                continue  # ignore malformed datagrams
        return latest

    def close(self) -> None:
        self._sock.close()
