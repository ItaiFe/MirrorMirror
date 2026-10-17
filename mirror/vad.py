"""Speech segmentation: frame-level VAD decisions -> complete utterances."""
import math
from collections import deque
from typing import Callable, List

import numpy as np


class Segmenter:
    """Cuts a 16 kHz mono float32 stream into utterances.

    A segment = preroll + speech + trailing silence (up to min_silence_ms).
    Segments with less than min_speech_ms of speech are dropped; segments
    are force-split at max_segment_s.
    """

    def __init__(self, is_speech: Callable[[np.ndarray], bool], frame_samples: int = 512,
                 sample_rate: int = 16000, min_silence_ms: int = 700,
                 min_speech_ms: int = 250, max_segment_s: float = 15.0,
                 preroll_ms: int = 300):
        self._is_speech = is_speech
        self._n = frame_samples
        frame_ms = 1000.0 * frame_samples / sample_rate
        self._silence_frames = max(1, int(min_silence_ms / frame_ms))  # floor: never exceed min_silence_ms
        self._speech_frames_min = math.ceil(min_speech_ms / frame_ms)
        self._max_samples = int(max_segment_s * sample_rate)
        self._preroll_frames = int(preroll_ms / frame_ms)
        self.reset()

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll: deque = deque(maxlen=max(self._preroll_frames, 1))
        self._seg: List[np.ndarray] = []
        self._in_speech = False
        self._speech_count = 0
        self._silence_count = 0

    def feed(self, chunk: np.ndarray) -> List[np.ndarray]:
        self._pending = np.concatenate([self._pending, np.asarray(chunk, dtype=np.float32)])
        out: List[np.ndarray] = []
        n = self._n
        pos = 0
        while len(self._pending) - pos >= n:
            seg = self._step(self._pending[pos:pos + n].copy())
            pos += n
            if seg is not None:
                out.append(seg)
        self._pending = self._pending[pos:]
        return out

    def _step(self, frame: np.ndarray):
        speech = bool(self._is_speech(frame))
        if not self._in_speech:
            if not speech:
                self._preroll.append(frame)
                return None
            self._in_speech = True
            self._seg = list(self._preroll) + [frame]
            self._preroll.clear()
            self._speech_count = 1
            self._silence_count = 0
        else:
            self._seg.append(frame)
            if speech:
                self._speech_count += 1
                self._silence_count = 0
            else:
                self._silence_count += 1
        total = len(self._seg) * self._n
        if self._silence_count >= self._silence_frames or total >= self._max_samples:
            keep = self._speech_count >= self._speech_frames_min
            seg = np.concatenate(self._seg) if keep else None
            self._seg = []
            self._in_speech = False
            self._speech_count = self._silence_count = 0
            return seg
        return None


class SileroVad:
    """Silero VAD v5 ONNX on 512-sample, 16 kHz frames; keeps its own state."""

    _CONTEXT = 64

    def __init__(self, model_path: str, threshold: float = 0.5):
        import onnxruntime as ort
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._threshold = threshold
        self._sr = np.array(16000, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> bool:
        x = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        x = np.concatenate([self._context, x], axis=1)
        prob, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -self._CONTEXT:]
        return float(prob.reshape(-1)[0]) > self._threshold
