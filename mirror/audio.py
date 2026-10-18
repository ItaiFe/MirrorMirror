"""Audio I/O (sounddevice, imported lazily) and the mouth-level envelope.

A raw ALSA hw: device (the ReSpeaker on the Orin) neither resamples nor remixes:
it may accept only 16 kHz or 48 kHz, and only 2 channels. So Mic and Player
negotiate a format the device takes and convert in numpy: the rest of the
pipeline always sees 16 kHz mono in, and plays each TTS clip at its own rate.

Threads: mirror-voice opens the Mic on its audio thread and the Player on the
main thread, so two PortAudio streams are opened/closed from different threads.
Each stream is only ever touched by one thread at a time (the Player's writer
thread is joined before its stream is aborted or closed); this has worked on
CoreAudio and ALSA, but if a host API ever objects, open both on one thread.
"""
from __future__ import annotations

import logging
import math
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

CHUNK = 512  # ~32 ms at 16 kHz, matches the VAD frame size
WRITE_CHUNK_S = 0.05  # playback is written in short blocks so it can be stopped
MIC_READ_TIMEOUT_S = 2.0  # no audio for this long: the mic has stalled
PLAY_JOIN_GRACE_S = 2.0  # the writer may lag the clip's length by this much

log = logging.getLogger("mirror.audio")


class MicStalled(RuntimeError):
    """The input stream delivered no audio within the read timeout."""


class PlayerStalled(RuntimeError):
    """A write to the output stream did not finish in time."""


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


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """A whole clip from sr_from to sr_to (polyphase with scipy, else linear)."""
    x = np.asarray(x, dtype=np.float32)
    if sr_from == sr_to or len(x) == 0:
        return x
    try:
        from scipy.signal import resample_poly  # optional
    except ImportError:
        n_out = int(round(len(x) * sr_to / sr_from))
        t = np.arange(n_out) * (sr_from / sr_to)
        return np.interp(t, np.arange(len(x)), x).astype(np.float32)
    g = math.gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


class StreamResampler:
    """Block-by-block resampling of a continuous stream, with no seams between
    blocks. An integer down-factor averages groups of samples (a cheap
    anti-alias); any other ratio interpolates linearly."""

    def __init__(self, sr_in: int, sr_out: int):
        self._step = sr_in / sr_out
        self._factor = sr_in // sr_out if sr_in % sr_out == 0 else None
        self._buf = np.zeros(0, dtype=np.float32)
        self._pos = 0.0  # position of the next output sample in _buf (linear path)

    def feed(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self._factor == 1:
            return x
        buf = np.concatenate([self._buf, x]) if len(self._buf) else x
        if self._factor is not None:
            k = self._factor
            n = len(buf) // k
            self._buf = buf[n * k:]
            return buf[: n * k].reshape(n, k).mean(axis=1).astype(np.float32)
        last = len(buf) - 1
        if last < self._pos:
            self._buf = buf
            return np.zeros(0, dtype=np.float32)
        n = int((last - self._pos) // self._step) + 1
        t = self._pos + np.arange(n) * self._step
        out = np.interp(t, np.arange(len(buf)), buf).astype(np.float32)
        nxt = self._pos + n * self._step
        drop = min(int(nxt), len(buf))  # nxt may lie past this block's end
        self._buf, self._pos = buf[drop:], nxt - drop
        return out


def _device_info(sd, device: Optional[int], kind: str) -> dict:
    try:
        return dict(sd.query_devices(device, kind))
    except Exception:  # noqa: BLE001 - a default we can't query: no extra candidate
        return {}


def _pick_format(sd, device: Optional[int], kind: str,
                 candidates: list[tuple[int, int]]) -> tuple[int, int]:
    """The first (rate, channels) the device accepts, then its own default
    rate with up to 2 channels; the first candidate if nothing checks out (the
    open then raises the device's real error)."""
    info = _device_info(sd, device, kind)
    default_sr = info.get("default_samplerate")
    max_ch = int(info.get(f"max_{kind}_channels") or 0)
    tries = list(candidates)
    if default_sr and max_ch > 0:
        tries.append((int(default_sr), min(2, max_ch)))
    check = getattr(sd, f"check_{kind}_settings", None)
    if check is None:
        return tries[0]
    for rate, ch in tries:
        try:
            check(device=device, samplerate=rate, channels=ch, dtype="float32")
        except Exception:  # noqa: BLE001 - not supported; try the next one
            continue
        return rate, ch
    return tries[0]


_PLUG_HINTS = ("plughw", "pulse", "pipewire", "default")


def find_device(name_substr: Optional[str], kind: str,
                prefer_plug: bool = False) -> Optional[int]:
    """Index of a device whose name contains name_substr (case-insensitive) and
    that supports `kind` ("input" or "output"); None means the default. With
    prefer_plug, and several matches, one whose name says it converts formats
    (ALSA plughw, PulseAudio/PipeWire, "default") wins over a raw hw: device."""
    if not name_substr:
        return None
    import sounddevice as sd

    key = f"max_{kind}_channels"
    needle = name_substr.lower()
    matches = [(i, dev["name"]) for i, dev in enumerate(sd.query_devices())
               if dev[key] > 0 and needle in dev["name"].lower()]
    if not matches:
        log.warning("no %s device matches %r; using the default", kind, name_substr)
        return None
    chosen = matches[0]
    if prefer_plug:
        for hint in _PLUG_HINTS:
            hit = next((m for m in matches if hint in m[1].lower()), None)
            if hit is not None:
                chosen = hit
                break
    log.info("%s device: [%d] %s", kind, chosen[0], chosen[1])
    return chosen[0]


class Mic:
    """Context manager yielding CHUNK-sample float32 mono chunks at sample_rate via
    read(). The device is opened at a format it supports; channel 0 is taken and
    resampled to sample_rate. read() raises MicStalled after read_timeout_s
    without audio."""

    def __init__(self, device: Optional[int], sample_rate: int = 16000,
                 read_timeout_s: float = MIC_READ_TIMEOUT_S):
        self.device = device
        self.sample_rate = sample_rate
        self.read_timeout_s = read_timeout_s
        self.device_rate = sample_rate
        self.channels = 1
        self.reads = 0  # chunks returned since the stream was opened
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)  # ~6 s at 16 kHz
        self._stream = None
        self._resampler: Optional[StreamResampler] = None
        self._out = np.zeros(0, dtype=np.float32)

    def _callback(self, indata, frames, time_info, status) -> None:
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:  # the reader is gone or far behind: drop, never block
            pass

    def __enter__(self) -> "Mic":
        import sounddevice as sd

        sr = self.sample_rate
        rate, ch = _pick_format(sd, self.device, "input",
                                [(sr, 1), (sr, 2), (48000, 2)])
        self.device_rate, self.channels = rate, ch
        self._resampler = StreamResampler(rate, sr)
        self._out = np.zeros(0, dtype=np.float32)
        self.reads = 0
        self._stream = sd.InputStream(
            samplerate=rate,
            blocksize=max(1, round(CHUNK * rate / sr)),
            device=self.device,
            channels=ch,
            dtype="float32",
            callback=self._callback,
        )
        try:
            self._stream.start()
            log.info("mic open: %d Hz x %d ch (channel 0 -> %d Hz mono)", rate, ch, sr)
        except BaseException:
            # __exit__ never runs for a failed __enter__: don't leak the stream
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            stream, self._stream = self._stream, None
            try:
                stream.abort()  # input: nothing to drain
            finally:
                stream.close()

    def read(self) -> np.ndarray:
        while len(self._out) < CHUNK:
            try:
                block = self._q.get(timeout=self.read_timeout_s)
            except queue.Empty:
                raise MicStalled(
                    f"no audio from the mic for {self.read_timeout_s:.1f} s") from None
            self._out = np.concatenate([self._out, self._resampler.feed(block)])
        chunk, self._out = self._out[:CHUNK], self._out[CHUNK:]
        self.reads += 1
        return chunk


class Player:
    """Plays mono float32 audio on one persistent output stream.

    Opening a CoreAudio/ALSA output can take seconds on a cold device, so the
    stream is opened once (open(), or the first play) and reused; it is reopened
    only when the clip sample rate changes or after a write error. The device is
    opened at a format it supports: the clip's rate in mono if it can, else
    48 kHz / 16 kHz stereo or its default; clips are resampled (and duplicated
    to stereo) to match.
    """

    def __init__(self, device: Optional[int]):
        self.device = device
        self._stream = None
        self._rate: Optional[int] = None  # the clip rate the stream was opened for
        self._out_rate: Optional[int] = None  # the device's rate
        self._channels = 1

    def open(self, sr: int) -> None:
        """Open and start the output stream for clips at sr (a no-op if it already is)."""
        if self._stream is not None and self._rate == sr:
            return
        import sounddevice as sd

        self.close()
        rate, ch = _pick_format(sd, self.device, "output",
                                [(sr, 1), (48000, 2), (16000, 2)])
        stream = sd.OutputStream(samplerate=rate, device=self.device, channels=ch,
                                 dtype="float32")
        stream.start()
        self._stream, self._rate, self._out_rate, self._channels = stream, sr, rate, ch
        log.info("audio output open: %d Hz x %d ch (clips at %d Hz)", rate, ch, sr)

    def close(self) -> None:
        if self._stream is not None:
            stream, self._stream, self._rate = self._stream, None, None
            try:
                stream.stop()
            finally:
                stream.close()

    def _abort(self, stream, writer: threading.Thread, cancel: threading.Event) -> None:
        """Stop the writer at its next chunk, then abort and discard the stream.

        PortAudio isn't thread-safe and, on CoreAudio, abort() doesn't wake a
        blocked write(), so the writer is stopped first and the stream is only
        touched once it has exited."""
        cancel.set()
        if self._stream is stream:
            self._stream, self._rate = None, None
        writer.join(timeout=1.0)
        if writer.is_alive():  # never touch a stream a write() is still inside
            # Known leak: the wedged stream stays open (it is already detached, so
            # the next play opens a fresh one) until the write returns or the
            # process exits.
            log.warning("audio writer did not stop; leaving the output stream open")
            return
        for op in (stream.abort, stream.close):
            try:
                op()
            except Exception:  # noqa: BLE001 - best effort while already failing
                pass

    def play(self, samples: np.ndarray, sr: int, on_level: Callable[[float], None]) -> None:
        """Play samples, calling on_level once per 20 ms hop in step with the audio,
        then once with 0.0; returns when playback has finished. Raises
        PlayerStalled if the device stops taking audio."""
        hop_s = 0.02
        env = mouth_envelope(samples, sr, hop_s)  # on the clip's own timeline
        self.open(sr)
        stream = self._stream
        out_rate, channels = self._out_rate, self._channels
        errors: list[BaseException] = []
        cancel = threading.Event()
        mono = resample(np.asarray(samples, dtype=np.float32), sr, out_rate)
        data = np.ascontiguousarray(np.repeat(mono.reshape(-1, 1), channels, axis=1))
        chunk = max(1, int(out_rate * WRITE_CHUNK_S))
        duration = len(samples) / sr

        def write() -> None:
            try:
                for i in range(0, len(data), chunk):
                    if cancel.is_set():
                        return
                    stream.write(data[i:i + chunk])
            except Exception as e:  # noqa: BLE001 - re-raised on the calling thread
                errors.append(e)

        writer = threading.Thread(target=write, name="player", daemon=True)
        writer.start()
        start = time.monotonic()
        try:
            for i, level in enumerate(env):
                if errors:
                    break  # the device failed: stop animating a silent mouth
                delay = start + i * hop_s - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                on_level(float(level))
            writer.join(max(0.0, start + duration + PLAY_JOIN_GRACE_S - time.monotonic()))
        except BaseException:
            # Interrupted (shutdown signal, failing on_level): stop the blocked write
            # before anyone closes the stream under it (PortAudio isn't thread-safe).
            self._abort(stream, writer, cancel)
            raise
        if writer.is_alive():  # a write() wedged: give up on this stream
            self._abort(stream, writer, cancel)
            raise PlayerStalled(f"audio output stalled playing a {duration:.1f} s clip")
        if errors:
            self.close()
            raise errors[0]
        # write() returns once the last samples are queued; wait until they are heard
        tail = start + duration + float(stream.latency) - time.monotonic()
        if tail > 0:
            time.sleep(tail)
        on_level(0.0)
