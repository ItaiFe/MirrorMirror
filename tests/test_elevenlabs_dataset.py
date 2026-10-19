import json
import os
import sys
import wave
from unittest.mock import MagicMock

import numpy as np
import pytest
import responses

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tools"))
import elevenlabs_dataset as ed  # noqa: E402

VOICE = "voice123"
TTS_URL = ed.API_URL.format(voice_id=VOICE)
SETTINGS = {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0,
            "use_speaker_boost": True}
SR = ed.SAMPLE_RATE


def write_sentences(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def tone(seconds, amp=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return np.clip(np.rint(amp * 32767 * np.sin(2 * np.pi * 220 * t)),
                   -32768, 32767).astype("<i2")


def save_wav(out_dir, sid, pcm, sr=SR):
    os.makedirs(os.path.join(out_dir, "wavs"), exist_ok=True)
    with open(ed.wav_path(out_dir, sid), "wb") as f:
        f.write(ed.pcm_to_wav(pcm.astype("<i2").tobytes(), sr))


def no_sleep(_s):
    pass


# --- sentences, ids, manifest ---------------------------------------------------

def test_load_sentences_skips_comments_blanks_and_dedupes(tmp_path):
    a = write_sentences(tmp_path, "a.txt", ["# header", "שלום עולם", "", "  מה  שלומך? ",
                                            "שלום עולם"])
    b = write_sentences(tmp_path, "b.txt", ["מה שלומך?", "\tלילה\tטוב", "א|ב"])
    assert ed.load_sentences([a, b]) == ["שלום עולם", "מה שלומך?", "לילה טוב", "א ב"]


def test_manifest_ids_are_stable_and_new_sentences_get_next_ids(tmp_path):
    out = str(tmp_path / "ds")
    first = ed.update_manifest(out, ["א", "ב", "ג"])
    assert first == [("he_0001", "א"), ("he_0002", "ב"), ("he_0003", "ג")]
    # reordered, one dropped, one new: old ids keep, the new one gets he_0004
    again = ed.update_manifest(out, ["ד", "ג", "א"])
    assert again == first + [("he_0004", "ד")]
    assert ed.read_manifest(os.path.join(out, "sentences.tsv")) == again
    assert ed.select(again, ["ד", "ג", "א"], None) == [
        ("he_0001", "א"), ("he_0003", "ג"), ("he_0004", "ד")]
    assert ed.select(again, ["ד", "ג", "א"], 2) == [("he_0001", "א"), ("he_0003", "ג")]


def test_dry_run_prints_estimate_and_writes_nothing(tmp_path, capsys):
    path = write_sentences(tmp_path, "s.txt", ["א" * 140, "ב" * 140])
    out = tmp_path / "ds"
    assert ed.main([path, "--out", str(out), "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "sentences: 2" in text
    assert "characters: 280" in text
    assert "credits: ~280" in text and "1 credit per character" in text
    assert "check your plan" in text
    assert "~0.3 min" in text and "rough estimate" in text  # 280 / 14 = 20 s
    assert not out.exists()


def test_estimate_per_model():
    entries = [("he_0001", "abcd")]
    assert ed.estimate(entries, "eleven_flash_v2_5")["credits"] == 2.0
    unknown = ed.estimate(entries, "some_new_model")
    assert unknown["credits"] == 4.0 and not unknown["known_model"]


# --- requests -------------------------------------------------------------------

@responses.activate
def test_request_url_headers_and_body():
    responses.post(TTS_URL, body=b"\x01\x00\x02\x00")
    pcm = ed.synthesize("KEY", VOICE, "שלום", "eleven_multilingual_v2", SETTINGS,
                        sleep=no_sleep)
    assert pcm == b"\x01\x00\x02\x00"
    req = responses.calls[0].request
    assert req.url == TTS_URL + "?output_format=pcm_22050"
    assert req.headers["xi-api-key"] == "KEY"
    assert json.loads(req.body) == {"text": "שלום", "model_id": "eleven_multilingual_v2",
                                    "voice_settings": SETTINGS}


@responses.activate
def test_language_code_only_when_given():
    responses.post(TTS_URL, body=b"\x00\x00")
    ed.synthesize("KEY", VOICE, "x", "eleven_flash_v2_5", SETTINGS, "he", sleep=no_sleep)
    assert json.loads(responses.calls[0].request.body)["language_code"] == "he"


@responses.activate
def test_retries_429_then_succeeds():
    responses.post(TTS_URL, status=429, json={"detail": {"status": "system_busy"}})
    responses.post(TTS_URL, status=503, body="busy")
    responses.post(TTS_URL, body=b"\x05\x00")
    sleeps = []
    assert ed.synthesize("KEY", VOICE, "x", "m", SETTINGS, sleep=sleeps.append) == b"\x05\x00"
    assert len(responses.calls) == 3
    assert sleeps == [1.0, 2.0]


@responses.activate
def test_gives_up_after_max_tries():
    responses.post(TTS_URL, status=500, body="oops")
    with pytest.raises(ed.SentenceError):
        ed.synthesize("KEY", VOICE, "x", "m", SETTINGS, sleep=no_sleep)
    assert len(responses.calls) == ed.MAX_TRIES


@responses.activate
def test_quota_error_is_fatal():
    responses.post(TTS_URL, status=400,
                   json={"detail": {"status": "quota_exceeded", "message": "no credits"}})
    with pytest.raises(ed.FatalAPIError):
        ed.synthesize("KEY", VOICE, "x", "m", SETTINGS, sleep=no_sleep)
    assert len(responses.calls) == 1


@responses.activate
def test_401_aborts_run_cleanly(tmp_path, monkeypatch, capsys):
    responses.post(TTS_URL, status=401, json={"detail": {"status": "invalid_api_key"}})
    path = write_sentences(tmp_path, "s.txt", ["א", "ב", "ג"])
    out = tmp_path / "ds"
    monkeypatch.setenv("ELEVENLABS_API_KEY", "SECRET")
    code = ed.main([path, "--out", str(out), "--voice-id", VOICE, "--concurrency", "1"])
    assert code == 3
    err = capsys.readouterr().err
    assert "stopping" in err and "HTTP 401" in err and "SECRET" not in err
    assert len(responses.calls) == 1  # the rest were not sent
    assert (out / "metadata.csv").read_text(encoding="utf-8") == ""


def test_missing_key_exits(tmp_path, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    path = write_sentences(tmp_path, "s.txt", ["א"])
    assert ed.main([path, "--out", str(tmp_path / "ds"), "--voice-id", VOICE]) == 2


# --- generation, resume, metadata ------------------------------------------------

@responses.activate
def test_generate_resumes_and_writes_metadata(tmp_path, monkeypatch, capsys):
    pcm = tone(1.0).tobytes()
    responses.post(TTS_URL, body=pcm)
    path = write_sentences(tmp_path, "s.txt", ["שלום עולם", "לילה טוב", "בוקר טוב"])
    out = str(tmp_path / "ds")
    ed.update_manifest(out, ed.load_sentences([path]))
    save_wav(out, "he_0002", tone(2.0))  # already made: must be skipped
    monkeypatch.setenv("ELEVENLABS_API_KEY", "KEY")
    assert ed.main([path, "--out", out, "--voice-id", VOICE]) == 0
    sent = sorted(json.loads(c.request.body)["text"] for c in responses.calls)
    assert sent == sorted(["שלום עולם", "בוקר טוב"])
    with wave.open(ed.wav_path(out, "he_0001"), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SR)
        assert w.getnframes() == SR
    meta = open(os.path.join(out, "metadata.csv"), encoding="utf-8").read()
    assert meta == "he_0001|שלום עולם\nhe_0002|לילה טוב\nhe_0003|בוקר טוב\n"
    assert "3 clips, 0.1 min" in capsys.readouterr().out  # 4 s
    assert not [f for f in os.listdir(os.path.join(out, "wavs")) if ".tmp" in f]

    responses.calls.reset()
    assert ed.main([path, "--out", out, "--voice-id", VOICE]) == 0
    assert len(responses.calls) == 0  # everything exists now


def test_metadata_only_lists_sentences_with_wavs(tmp_path):
    out = str(tmp_path / "ds")
    entries = ed.update_manifest(out, ["א", "ב", "ג"])
    save_wav(out, "he_0003", tone(0.5))
    with open(ed.wav_path(out, "he_0001"), "wb"):
        pass  # empty file: not a clip
    clips, seconds = ed.write_metadata(entries, out)
    assert (clips, seconds) == (1, pytest.approx(0.5, abs=1e-3))
    assert open(os.path.join(out, "metadata.csv"), encoding="utf-8").read() == "he_0003|ג\n"


# --- check ----------------------------------------------------------------------

def test_check_flags(tmp_path, capsys):
    out = str(tmp_path / "ds")
    silence = np.zeros(int(1.5 * SR), dtype="<i2")
    text_ok = "א" * 21  # 21 chars / 1.5 s = 14 cps
    entries = ed.update_manifest(out, [text_ok, "ב" * 5, "ג", "ד" * 50, "ה" * 21, "ו" * 21])
    save_wav(out, "he_0001", tone(1.5))                             # fine
    save_wav(out, "he_0002", tone(1.5))                             # 3.3 cps: slow
    save_wav(out, "he_0003", tone(0.2))                             # short
    save_wav(out, "he_0004", tone(1.5))                             # 33 cps: fast
    save_wav(out, "he_0005", np.concatenate([silence, tone(1.5), silence]))  # silences
    save_wav(out, "he_0006", tone(1.5, amp=1.2))                    # clipping
    rows = {r["id"]: r for r in ed.run_check(entries, out)}
    assert rows["he_0001"]["flags"] == []
    assert rows["he_0002"]["flags"] == ["slow"]
    assert "short" in rows["he_0003"]["flags"]
    assert rows["he_0004"]["flags"] == ["fast"]
    assert set(rows["he_0005"]["flags"]) == {"lead_silence", "trail_silence"}
    assert rows["he_0005"]["lead_s"] == pytest.approx(1.5, abs=0.02)
    assert rows["he_0006"]["flags"] == ["clipping"]
    tsv = open(os.path.join(out, "check.tsv"), encoding="utf-8").read().splitlines()
    assert tsv[0].split("\t")[:3] == ["id", "duration_s", "cps"]
    assert len(tsv) == 7
    ed.print_check_summary(list(rows.values()), out)
    assert "5 flagged" in capsys.readouterr().out


def test_cer():
    assert ed.cer("שָׁלוֹם, עולם!", "שלום עולם") == 0.0  # niqqud and punctuation ignored
    assert ed.cer("abcd", "abxd") == pytest.approx(0.25)
    assert ed.cer("abcd", "") == 1.0


def test_check_with_stt_flags_misread(tmp_path):
    out = str(tmp_path / "ds")
    entries = ed.update_manifest(out, ["א" * 21, "ב" * 21])
    save_wav(out, "he_0001", tone(1.5))
    save_wav(out, "he_0002", tone(1.5))
    stt = MagicMock()
    stt.transcribe.side_effect = [MagicMock(text="א" * 21), MagicMock(text="גגג")]
    rows = ed.run_check(entries, out, stt)
    assert rows[0]["cer"] == 0.0 and rows[0]["flags"] == []
    assert rows[1]["cer"] == 1.0 and rows[1]["flags"] == ["cer"]
    _, kwargs = stt.transcribe.call_args
    assert kwargs == {"sr": 16000, "language": "he"}


@responses.activate
def test_check_with_whisper_server(tmp_path, capsys):
    responses.post("http://stt:8081/inference", json={"text": "שלום עולם", "language": "he"})
    path = write_sentences(tmp_path, "s.txt", ["שלום עולם"])
    out = str(tmp_path / "ds")
    ed.update_manifest(out, ["שלום עולם"])
    save_wav(out, "he_0001", tone(0.7))
    assert ed.main([path, "--out", out, "--check", "--stt-url", "http://stt:8081"]) == 0
    assert b'name="language"\r\n\r\nhe' in responses.calls[0].request.body
    assert "0 flagged" in capsys.readouterr().out


def test_check_ignores_digit_transcripts_and_slow_oracle_speech():
    import numpy as np
    import elevenlabs_dataset as ed

    class STT:
        def __init__(self, text):
            self.text = text

        def transcribe(self, samples, sr=16000, language=None):
            from mirror.stt import Transcript
            return Transcript(self.text, "he")

    sr = 22050
    pcm = (8000 * np.sin(np.arange(int(4.4 * sr)) * 0.05)).astype(np.int16)
    text = "בשנת אלפיים עשרים ושש המראה התחילה לדבר."  # 41 chars in 4.4 s: ~9.3 cps
    row = ed.check_clip(text, pcm, sr, STT("בשנת 2026 המראה התחילה לדבר."))
    assert row["cer"] > ed.MAX_CER and row["flags"] == []
    assert "cer" in ed.check_clip(text, pcm, sr, STT("משהו אחר לגמרי בכלל."))["flags"]
