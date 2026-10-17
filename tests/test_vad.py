import numpy as np

from mirror.vad import Segmenter

SR = 16000


def fake_is_speech(frame):
    return np.abs(frame).mean() > 0.1


def silence(s):
    return np.zeros(int(SR * s), dtype=np.float32)


def tone(s, amp=0.5):
    t = np.arange(int(SR * s)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def run(audio, chunk=512):
    seg = Segmenter(fake_is_speech)
    out = []
    for i in range(0, len(audio), chunk):
        out.extend(seg.feed(audio[i:i + chunk]))
    return out


def test_single_utterance():
    out = run(np.concatenate([silence(0.5), tone(1.0), silence(1.0)]))
    assert len(out) == 1
    assert SR * 1.0 <= len(out[0]) <= SR * (1.0 + 0.3 + 0.7)
    assert out[0].dtype == np.float32


def test_short_blip_dropped():
    assert run(np.concatenate([silence(0.5), tone(0.1), silence(1.5)])) == []


def test_max_length_split():
    out = run(tone(20.0))
    assert len(out) >= 1
    assert SR * 15.0 <= len(out[0]) <= SR * 15.0 + 512


def test_arbitrary_chunk_sizes():
    audio = np.concatenate([silence(0.5), tone(1.0), silence(1.0), tone(0.8), silence(1.0)])
    counts = {c: len(run(audio, c)) for c in (100, 512, 3000)}
    assert counts == {100: 2, 512: 2, 3000: 2}


def test_reset_discards_partial():
    seg = Segmenter(fake_is_speech)
    out = list(seg.feed(tone(0.5)))
    seg.reset()
    out += seg.feed(silence(1.0))
    assert out == []
