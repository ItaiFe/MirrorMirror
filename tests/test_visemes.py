from pathlib import Path

import numpy as np
import pytest

from mirror.visemes import PHONEME_TO_SHAPE, VisemeTrack, rms_shape, track_from_alignments

SR = 22050
FRAME_S = 1 / 30
PIPER_MODEL = Path("models/en_US-lessac-medium.onnx")


def durations(track: VisemeTrack) -> list[float]:
    return list(np.diff(np.append(track.times, track.end)))


def test_rescale_to_audio_length():
    sr = 1000  # 1000 samples = 1 s, so every segment is long enough to keep
    track = track_from_alignments([("m", 250), ("a", 500)], sr=sr, total_samples=1000)
    assert track.shapes[-1] == "D"
    assert track.end == pytest.approx(1000 / sr)  # the final segment ends with the audio
    assert track.at(999 / sr) == "D"
    assert track.at(1001 / sr) == "X"
    # proportions are kept: "m" covers the first third of the rescaled audio
    assert track.at(0.25) == "A" and track.at(0.4) == "D"


def test_merge_and_min_hold():
    al = [("p", 300), ("a", 4000), ("a", 1000), ("t", 100), ("o", 3000)]
    track = track_from_alignments(al, sr=SR, total_samples=8400)
    assert track.shapes == ["A", "D", "E"]  # the 100-sample "t" is absorbed
    assert track.times[2] == pytest.approx(5400 / SR)


def test_protected_shapes_survive():
    track = track_from_alignments([("a", 3000), ("m", 600), ("a", 3000)],
                                  sr=SR, total_samples=6600)
    assert track.shapes == ["D", "A", "D"]
    assert durations(track)[1] >= FRAME_S - 1e-9
    assert track.end == pytest.approx(6600 / SR)  # the time came from the next "a"


def test_protected_shape_borrows_from_previous_at_end():
    track = track_from_alignments([("a", 3000), ("f", 300)], sr=SR, total_samples=3300)
    assert track.shapes == ["D", "G"]
    assert durations(track)[1] == pytest.approx(FRAME_S)


def test_short_unprotected_still_absorbed_after_protected():
    # "l" is protected and kept; the short "t" after it is absorbed into it
    track = track_from_alignments([("a", 3000), ("l", 2000), ("t", 200), ("o", 3000)],
                                  sr=SR, total_samples=8200)
    assert track.shapes == ["D", "H", "E"]


def test_protected_cluster_never_shrinks_a_protected_donor():
    # f-u-l then a short t: G and H can borrow legally, F has no legal donor
    al = [("a", 3000), ("f", 500), ("u", 500), ("l", 500), ("t", 300)]
    track = track_from_alignments(al, sr=SR, total_samples=4800)
    assert track.shapes == ["D", "G", "F", "H"]
    d = durations(track)
    assert d[1] == pytest.approx(FRAME_S)       # G took its time from the "a"
    assert d[2] == pytest.approx(500 / SR)      # F kept as it was, not shrunk
    assert d[3] >= FRAME_S - 1e-9               # H took the "t" (then absorbed it)
    assert track.end == pytest.approx(4800 / SR)


def test_protected_pair_at_track_end_keeps_both():
    # A then G at the end: G has no next segment and A is at its minimum
    track = track_from_alignments([("a", 3000), ("m", 300), ("f", 300)],
                                  sr=SR, total_samples=3600)
    assert track.shapes == ["D", "A", "G"]
    d = durations(track)
    assert d[1] == pytest.approx(FRAME_S)
    assert d[2] == pytest.approx(300 / SR)  # no legal donor: kept as-is


def test_protected_without_any_legal_donor_kept_as_is():
    track = track_from_alignments([("m", 300)], sr=SR, total_samples=300)
    assert track.shapes == ["A"] and track.end == pytest.approx(300 / SR)


def test_insert_b_between_a_and_d():
    track = track_from_alignments([("m", 3000), ("a", 3000)], sr=SR, total_samples=6000)
    assert track.shapes == ["A", "B", "D"]
    assert durations(track)[1] >= FRAME_S - 1e-9


def test_hebrew_phonemes():
    assert PHONEME_TO_SHAPE["ʁ"] == "B"
    assert PHONEME_TO_SHAPE["χ"] == "B"


def test_unknown_and_silence_shapes():
    track = track_from_alignments([("^", 3000), ("ʘ", 3000), (".", 3000)],
                                  sr=SR, total_samples=9000)
    assert track.shapes == ["X", "B", "X"]


def test_stress_mark_joins_next_phoneme():
    # piper gives the stress mark its own (often long) duration; it belongs to the vowel
    track = track_from_alignments([("m", 3000), ("ˈ", 2000), ("a", 1000), ("ː", 2000)],
                                  sr=SR, total_samples=8000)
    assert track.shapes == ["A", "B", "D"]
    assert track.at(3500 / SR) == "D" and track.at(7900 / SR) == "D"


def test_at_outside_track_is_x():
    track = VisemeTrack(np.array([0.0, 0.1]), ["X", "D"], end=0.5)
    assert track.at(-0.01) == "X"
    assert track.at(0.05) == "X"
    assert track.at(0.1) == "D"
    assert track.at(0.5) == "X"
    assert VisemeTrack(np.array([]), []).at(0.0) == "X"


def test_rms_shape_thresholds():
    assert [rms_shape(v) for v in (0.0, 0.079, 0.08, 0.349, 0.35, 0.649, 0.65, 1.0)] == [
        "X", "X", "B", "B", "C", "C", "D", "D"]


@pytest.mark.slow
def test_piper_alignments_real():
    pytest.importorskip("piper")
    pytest.importorskip("onnx")
    if not PIPER_MODEL.exists():
        pytest.skip("piper model missing")
    from mirror.tts import PiperTTS

    audio, sr, track = PiperTTS(str(PIPER_MODEL)).synth_visemes(
        "Mirror, mirror on the wall. Who is the fairest of them all?")
    assert track is not None
    duration = len(audio) / sr
    assert track.end - track.times[0] >= 0.9 * duration
    assert set(track.shapes) <= set("ABCDEFGHX") and "A" in track.shapes
