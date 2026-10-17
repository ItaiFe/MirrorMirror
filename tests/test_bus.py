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
        assert rx.poll() == (State.SPEAKING, 0.42)  # latest wins
        assert rx.poll() is None
    finally:
        tx.close()
        rx.close()
