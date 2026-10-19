import json
import socket
import time

from mirror.bus import FaceBus
from mirror.states import State


def test_bus_roundtrip():
    rx = FaceBus.receiver(17777)
    tx = FaceBus.sender(17777)
    try:
        tx.send(State.THINKING, 0.0)
        tx.send(State.SPEAKING, 0.42)
        time.sleep(0.05)
        assert rx.poll() == (State.SPEAKING, 0.42, "X")  # latest wins
        assert rx.poll() is None
    finally:
        tx.close()
        rx.close()


def test_bus_viseme_roundtrip():
    rx = FaceBus.receiver(17778)
    tx = FaceBus.sender(17778)
    try:
        tx.send(State.SPEAKING, 0.7, viseme="D")
        time.sleep(0.05)
        assert rx.poll() == (State.SPEAKING, 0.7, "D")
    finally:
        tx.close()
        rx.close()


def test_bus_old_message_defaults_x():
    rx = FaceBus.receiver(17779)
    raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        raw.sendto(json.dumps({"s": "speaking", "m": 0.5}).encode(), ("127.0.0.1", 17779))
        time.sleep(0.05)
        assert rx.poll() == (State.SPEAKING, 0.5, "X")
    finally:
        raw.close()
        rx.close()


def test_bus_long_datagram_is_read_whole():
    rx = FaceBus.receiver(17780)
    raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:  # over the old 1024-byte read, which truncated it into invalid JSON
        msg = {"s": "speaking", "m": 0.5, "v": "B", "pad": "x" * 2000}
        raw.sendto(json.dumps(msg).encode(), ("127.0.0.1", 17780))
        time.sleep(0.05)
        assert rx.poll() == (State.SPEAKING, 0.5, "B")
    finally:
        raw.close()
        rx.close()
