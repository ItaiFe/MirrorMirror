import csv
import json
import os
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tools"))
import prepare_piper_dataset as ppd  # noqa: E402

SR = 22050
ID_MAP = {"_": [0], "^": [1], "$": [2], " ": [3], ",": [8], ".": [10],
          "a": [14], "l": [24], "m": [25], "o": [27], "s": [31], "ʃ": [96], "ˈ": [120]}

# Fake G2P: a lookup table, so no model is needed.
PHONEMES = {
    "שלום": "ʃalˈom.",
    "שלום שלום": "ʃalˈom ʃalˈom.",
    "ריק": "",
    "זר": "ʃalˈom ʁ.",          # ʁ is not in the test map
    "מעורב": "ʃalˈom.",         # the full IPA has one more (Latin) word
    "מסומן": "ʃalˈom.",
}
FULL_IPA = {"מעורב": "ʃalˈom hello."}


class FakeG2P:
    def phonemize(self, text):
        return PHONEMES.get(text, "ʃalˈom.")  # anything else (the sample lines)

    def ipa(self, text):
        return FULL_IPA.get(text, self.phonemize(text))


def write_wav(path, seconds=1.0, sr=SR, channels=1):
    n = int(seconds * sr)
    pcm = (0.2 * 32767 * np.sin(np.arange(n * channels) * 0.05)).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


@pytest.fixture
def dataset(tmp_path):
    ds = tmp_path / "dataset_he"
    (ds / "wavs").mkdir(parents=True)
    rows = [("he_0001", "שלום"), ("he_0002", "שלום שלום"), ("he_0003", "ריק"),
            ("he_0004", "זר"), ("he_0005", "מעורב"), ("he_0006", "מסומן"),
            ("he_0007", "שלום"), ("he_0008", "שלום")]
    (ds / "metadata.csv").write_text(
        "".join(f"{i}|{t}\n" for i, t in rows), encoding="utf-8")
    for sid, _ in rows:
        if sid == "he_0007":
            continue  # not generated yet
        if sid == "he_0008":
            write_wav(ds / "wavs" / f"{sid}.wav", sr=16000)
        else:
            write_wav(ds / "wavs" / f"{sid}.wav", seconds=1.5)
    (ds / "check.tsv").write_text(
        "id\tduration_s\tcps\tlead_s\ttrail_s\tclipped\tcer\tflags\ttext\theard\n"
        "he_0001\t1.5\t8\t0\t0\t0\t0.00\t\tשלום\tשלום\n"
        "he_0006\t1.5\t8\t0\t0\t0\t0.60\tmisread\tמסומן\tמשהו\n", encoding="utf-8")
    config = {"audio": {"sample_rate": SR, "quality": "train"}, "phoneme_type": "raw",
              "phoneme_id_map": ID_MAP, "num_symbols": 256, "dataset": ""}
    cfg = tmp_path / "voice.onnx.json"
    cfg.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return ds, cfg


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter="|"))


def test_phoneme_ids_match_piper_framing():
    assert ppd.phoneme_ids("ʃa", ID_MAP) == [1, 0, 96, 0, 14, 0, 2]


def test_phoneme_ids_match_installed_piper():
    piper_ids = pytest.importorskip("piper.phoneme_ids")
    ph = "ʃalˈom, ʃalˈom."
    assert ppd.phoneme_ids(ph, ID_MAP) == piper_ids.phonemes_to_ids(list(ph), ID_MAP)


def test_prepare_keeps_good_clips_and_reports_the_rest(dataset, tmp_path):
    ds, cfg = dataset
    out = tmp_path / "train_he"
    lines = []
    s = ppd.prepare(str(ds), str(out), str(cfg), FakeG2P(), log=lines.append)

    assert (s["total"], s["kept"], s["dropped"]) == (8, 2, 6)
    assert s["minutes"] == round(3.0 / 60, 1)
    rows = read_rows(out / "metadata_ids.csv")
    assert [r[0] for r in rows] == ["he_0001", "he_0002"]
    assert rows[0][1] == "שלום"
    assert rows[0][2] == " ".join(map(str, ppd.phoneme_ids("ʃalˈom.", ID_MAP)))
    assert read_rows(out / "metadata_phonemes.csv") == [
        ["he_0001", "ʃalˈom."], ["he_0002", "ʃalˈom ʃalˈom."]]
    assert sorted(os.listdir(out / "wavs")) == ["he_0001.wav", "he_0002.wav"]

    with open(out / "dropped.tsv", encoding="utf-8") as f:
        reasons = {r["id"]: r["reason"] for r in csv.DictReader(f, delimiter="\t")}
    assert reasons == {
        "he_0003": "empty phonemes",
        "he_0004": "symbols not in phoneme_id_map",
        "he_0005": "G2P dropped word(s)",
        "he_0006": "flagged in check.tsv",
        "he_0007": "missing wav",
        "he_0008": reasons["he_0008"],
    }
    assert reasons["he_0008"].startswith("wav is 1 ch, 16-bit, 16000 Hz")
    assert json.loads((out / "phoneme_id_map.json").read_text("utf-8")) == ID_MAP
    voice = json.loads((out / "config.json").read_text("utf-8"))
    assert voice["phoneme_id_map"] == ID_MAP and voice["dataset"] == "dataset_he"
    assert any("2 of 8 clips kept" in line for line in lines)
    samples = read_rows(out / "samples_ids.csv")
    assert [r[1] for r in samples] == ppd.SAMPLE_SENTENCES
    assert samples[0][2] == rows[0][2]


def test_keep_flagged_and_no_copy(dataset, tmp_path):
    ds, cfg = dataset
    out = tmp_path / "train_he"
    s = ppd.prepare(str(ds), str(out), str(cfg), FakeG2P(), keep_flagged=True,
                    copy_wavs=False, log=lambda _m: None)
    assert s["kept"] == 3
    assert not (out / "wavs").exists()


def test_quoted_text_round_trips(tmp_path):
    ds = tmp_path / "ds"
    (ds / "wavs").mkdir(parents=True)
    text = '"שלום" אמר'
    (ds / "metadata.csv").write_text(f"he_0001|{text}\n", encoding="utf-8")
    write_wav(ds / "wavs" / "he_0001.wav")
    cfg = tmp_path / "v.json"
    cfg.write_text(json.dumps({"audio": {"sample_rate": SR}, "phoneme_id_map": ID_MAP}),
                   encoding="utf-8")

    class G2P:
        def phonemize(self, _t):
            return "ʃalˈom"

    ppd.prepare(str(ds), str(tmp_path / "out"), str(cfg), G2P(), log=lambda _m: None)
    assert read_rows(tmp_path / "out" / "metadata_ids.csv")[0][1] == text


def test_runtime_filter_is_shared():
    from mirror.tts import keep_voice_symbols
    assert keep_voice_symbols(" ʃalˈom! ", set(ID_MAP)) == "ʃalˈom"


def test_main_without_metadata(tmp_path, capsys):
    assert ppd.main(["--dataset", str(tmp_path)]) == 2
