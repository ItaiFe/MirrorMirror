"""Audio I/O (sounddevice, imported lazily) and the mouth-level envelope."""
from __future__ import annotations

import math
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

CHUNK = 512  # ~32 ms at 16 kHz, matches the VAD frame size


def mouth_envelope(samples: np.ndarray, sr: int, hop_s: float = 0.02) -> np.ndarray:
    """RMS per hop, normalised by the 95th percentile, clipped to 0..1."""
    hop = max(1, int(round(sr * hop_s)))
    n = math.ceil(len(samples) / hop)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    x = np.asarray(samples, dtype=np.float32)
    pad = np.zeros(n * hop, dtype=np.float32)
    pad[: len(x)] = x
    rms = np.sqrt((pad.reshape(n, hop) ** 2).mean(axis=1))
    ref = np.percentile(rms, 95)
    if ref <= 0:
        return np.zeros(n, dtype=np.float32)
    return np.clip(rms / ref, 0.0, 1.0).astype(np.float32)


def find_device(name_substr: Optional[str], kind: str) -> Optional[int]:
    """Index of the first device whose name contains name_substr (case-insensitive)
    and that supports `kind` ("input" or "output"); None means the default."""
    if not name_substr:
        return None
    import sounddevice as sd

    key = f"max_{kind}_channels"
    needle = name_substr.lower()
    for i, dev in enumerate(sd.query_devices()):
        if dev[key] > 0 and needle in dev["name"].lower():
            return i
    return None


class Mic:
    """Context manager yielding ~32 ms float32 mono chunks via read()."""

    def __init__(self, device: Optional[int], sample_rate: int = 16000):
        self.device = device
        self.sample_rate = sample_rate
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:
        self._q.put(indata[:, 0].copy())

    def __enter__(self) -> "Mic":
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=CHUNK,
            device=self.device,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self) -> np.ndarray:
        return self._q.get()


class Player:
    """Plays mono float32 audio on one persistent output stream.

    Opening a CoreAudio/ALSA output can take seconds on a cold device, so the
    stream is opened once (open(), or the first play) and reused; it is reopened
    only when the sample rate changes or after a write error.
    """

    def __init__(self, device: Optional[int]):
        self.device = device
        self._stream = None
        self._rate: Optional[int] = None

    def open(self, sr: int) -> None:
        """Open and start the output stream at sr (a no-op if it already is)."""
        if self._stream is not None and self._rate == sr:
            return
        import sounddevice as sd

        self.close()
        stream = sd.OutputStream(samplerate=sr, device=self.device, channels=1,
                                 dtype="float32")
        stream.start()
        self._stream, self._rate = stream, sr

    def close(self) -> None:
        if self._stream is not None:
            stream, self._stream, self._rate = self._stream, None, None
            try:
                stream.stop()
            finally:
                stream.close()

    def play(self, samples: np.ndarray, sr: int, on_level: Callable[[float], None]) -> None:
        """Play samples, calling on_level once per 20 ms hop in step with the audio,
        then once with 0.0; returns when playback has finished."""
        hop_s = 0.02
        env = mouth_envelope(samples, sr, hop_s)
        self.open(sr)
        stream = self._stream
        errors: list[BaseException] = []

        def write() -> None:
            try:
                stream.write(np.ascontiguousarray(samples, dtype=np.float32).reshape(-1, 1))
            except Exception as e:  # noqa: BLE001 - re-raised on the calling thread
                errors.append(e)

        writer = threading.Thread(target=write, name="player", daemon=True)
        writer.start()
        start = time.monotonic()
        for i, level in enumerate(env):
            delay = start + i * hop_s - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            on_level(float(level))
        writer.join()
        if errors:
            self.close()
            raise errors[0]
        # write() returns once the last samples are queued; wait until they are heard
        tail = start + len(samples) / sr + float(stream.latency) - time.monotonic()
        if tail > 0:
            time.sleep(tail)
        on_level(0.0)
