import io
import wave

import numpy as np
import pytest
import requests
import responses

from mirror.stt import STTError, Transcript, WhisperServerSTT

URL = "http://127.0.0.1:8080"
SAMPLES = np.zeros(1600, dtype=np.float32)


def make():
    return WhisperServerSTT(URL, timeout_s=2.0)


@responses.activate
def test_parses_hebrew():
    responses.post(URL + "/inference", json={"text": " מראה מראה ", "language": "he"})
    assert make().transcribe(SAMPLES) == Transcript("מראה מראה", "he")


@responses.activate
def test_detected_language_fallback_field():
    responses.post(URL + "/inference", json={"text": "hi", "detected_language": "he"})
    assert make().transcribe(SAMPLES).lang == "he"


@responses.activate
def test_unknown_lang_maps_to_en():
    responses.post(URL + "/inference", json={"text": "x", "language": "ar"})
    assert make().transcribe(SAMPLES).lang == "en"


@responses.activate
def test_timeout_raises():
    responses.post(URL + "/inference", body=requests.exceptions.Timeout())
    with pytest.raises(STTError):
        make().transcribe(SAMPLES)


@responses.activate
def test_connection_error_raises():
    responses.post(URL + "/inference", body=requests.exceptions.ConnectionError())
    with pytest.raises(STTError):
        make().transcribe(SAMPLES)


@responses.activate
def test_http_error_raises():
    responses.post(URL + "/inference", status=500, body="boom")
    with pytest.raises(STTError):
        make().transcribe(SAMPLES)


@responses.activate
def test_malformed_json_raises():
    responses.post(URL + "/inference", body="not json")
    with pytest.raises(STTError):
        make().transcribe(SAMPLES)


@responses.activate
def test_wav_payload_and_form_fields():
    responses.post(URL + "/inference", json={"text": "a", "language": "en"})
    samples = np.array([0.0, 0.5, -0.5, 2.0, -2.0], dtype=np.float32)
    make().transcribe(samples)
    req = responses.calls[0].request
    body = req.body if isinstance(req.body, bytes) else req.body.read()
    assert b"RIFF" in body
    for field in (b'name="response_format"', b"verbose_json", b'name="language"',
                  b"auto", b'name="temperature"', b'name="file"'):
        assert field in body
    wav = body[body.index(b"RIFF"):]
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert pcm[3] == 32767 and pcm[4] == -32767  # clipped
    assert abs(int(pcm[1]) - 16383) <= 1


FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name, text, lang", [
    ("whisper_verbose_en.json", "Mirror, mirror on the wall, who is the fairest of them all?", "en"),
    ("whisper_verbose_he.json", "מה אמר אש על הקיר מי הכי יפה?", "he"),
])
@responses.activate
def test_real_whisper_server_response(name, text, lang):
    # Trimmed copies of real whisper-server (whisper.cpp 1.9.5) verbose_json replies:
    # the language field carries the full name ("hebrew"), not the ISO code.
    body = (FIXTURES / name).read_text(encoding="utf-8")
    responses.post(URL + "/inference", body=body, content_type="application/json")
    assert make().transcribe(SAMPLES) == Transcript(text, lang)
