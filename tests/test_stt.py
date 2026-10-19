import io
import re
import wave

import numpy as np
import pytest
import requests
import responses

from mirror.stt import STTError, Transcript, WhisperServerSTT, script_language

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
    responses.post(URL + "/inference", json={"text": "42", "detected_language": "he"})
    assert make().transcribe(SAMPLES).lang == "he"


@responses.activate
def test_unknown_lang_maps_to_en():
    responses.post(URL + "/inference", json={"text": "x", "language": "ar"})
    assert make().transcribe(SAMPLES).lang == "en"


@pytest.mark.parametrize("server_lang", ["HE", "he-IL", "he_IL", " Hebrew "])
@responses.activate
def test_server_language_ignores_case_and_region(server_lang):
    responses.post(URL + "/inference", json={"text": "42", "language": server_lang})
    assert make().transcribe(SAMPLES).lang == "he"


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


@responses.activate
def test_null_text_is_empty():
    responses.post(URL + "/inference", json={"text": None, "language": "en"})
    assert make().transcribe(SAMPLES) == Transcript("", "en")


# --- language from the transcript's script ---------------------------------------------

@pytest.mark.parametrize("text, lang", [
    ("מראה מראה שעל הקיר", "he"),
    ("Mirror, mirror on the wall", "en"),
    ("מראה מראה, tell me", "he"),  # mixed, Hebrew majority
    ("mirror mirror, ספרי", "en"),  # mixed, Latin majority
    ("מַרְאָה", "he"),  # niqqud marks are not letters, the base letters count
    ("42 - 7", None),
    ("", None),
])
def test_script_language(text, lang):
    assert script_language(text) == lang


@pytest.mark.parametrize("text, server_lang, lang", [
    ("Mirror mirror, who is the fairest?", "hebrew", "en"),  # ivrit-ai says hebrew for all
    ("מראה מראה, מה שלומך?", "english", "he"),
    # transliterated English ("מיר מיר"): the script wins, the reply comes in Hebrew
    ("מיר מיר, תגיד לי משהו", "english", "he"),
    ("2024", "english", "en"),  # no letters: the server's language field decides
    ("2024", "hebrew", "he"),
])
@responses.activate
def test_transcribe_trusts_script_over_server_language(text, server_lang, lang):
    responses.post(URL + "/inference", json={"text": text, "language": server_lang})
    assert make().transcribe(SAMPLES).lang == lang


# --- whisper-server flags ----------------------------------------------------------------

def _repo_file(*parts):
    import os
    with open(os.path.join(os.path.dirname(__file__), os.pardir, *parts),
              encoding="utf-8") as f:
        return f.read()


def _stt_flags(cmd):
    """The whisper-server flags, minus the model path and host/port."""
    toks = cmd.split()
    flags = []
    i = 1
    while i < len(toks):
        if toks[i] in ("-m", "--host", "--port"):
            i += 2
            continue
        flags.append(toks[i])
        i += 1
    return flags


def test_whisper_server_flags_match_in_dev_and_deploy():
    """-ac 768: a 15.36 s audio context (the VAD caps segments at 15 s) makes the
    turbo model ~2x faster; both launchers must run the same server."""
    make = next(l for l in _repo_file("Makefile").splitlines()
                if l.startswith("WHISPER :="))
    unit = next(l for l in _repo_file("deploy", "systemd", "mirror-stt.service")
                .splitlines() if l.startswith("ExecStart="))
    make_cmd = make.split(":=", 1)[1]
    unit_cmd = unit.split("=", 1)[1]
    assert re.search(r"-ac 768\b", make_cmd)
    assert _stt_flags(make_cmd) == _stt_flags(unit_cmd)
    assert "-m $(STT_MODEL)" in make_cmd
    assert "-m %h/mirror-models/${STT_MODEL} " in unit_cmd


def test_stt_model_defaults_to_ivrit_turbo():
    assert re.search(r"^STT_MODEL \?= \$\(MODELS\)/ivrit-turbo-q5_0\.bin$",
                     _repo_file("Makefile"), re.MULTILINE)
    unit = _repo_file("deploy", "systemd", "mirror-stt.service")
    assert "Environment=STT_MODEL=ivrit-turbo-q5_0.bin" in unit
