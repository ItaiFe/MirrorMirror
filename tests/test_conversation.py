import numpy as np
from mirror.config import ConversationConfig
from mirror.conversation import Controller
from mirror.llm import LLMError
from mirror.persona import FALLBACK
from mirror.states import State
from mirror.stt import STTError, Transcript
from mirror.visemes import VisemeTrack

SEG = np.zeros(1600, dtype=np.float32)


class FakeSTT:
    def __init__(self, script):
        self.script = list(script)  # Transcript or Exception instances
        self.calls = 0

    def transcribe(self, samples, sr=16000):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeLLM:
    def __init__(self, replies=("Thou art fairest.",)):
        self.replies = list(replies)
        self.calls = []

    def reply(self, history, user_text, lang):
        self.calls.append((list(history), user_text, lang))
        item = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(item, Exception):
            raise item
        return item


class FakeTTS:
    def __init__(self, track=None):
        self.calls = []
        self.track = track

    def synth_visemes(self, text, lang):
        self.calls.append((text, lang))
        return np.zeros(2205, dtype=np.float32), 22050, self.track  # 0.1 s


class FakePlayer:
    def __init__(self, levels=(0.5, 0.9, 0.0), during=None):
        self.levels = levels
        self.during = during  # callable run mid-playback
        self.plays = 0

    def play(self, samples, sr, on_level):
        self.plays += 1
        for lvl in self.levels:
            on_level(lvl)
            if self.during is not None:
                self.during()


class FakeLights:
    def __init__(self):
        self.states = []

    def set_state(self, state):
        self.states.append(state)


class FakeFace:
    def __init__(self):
        self.msgs = []

    def send(self, state, mouth, viseme="X"):
        self.msgs.append((state, mouth, viseme))


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(script, llm=None, player=None, tts=None, face=None):
    parts = dict(stt=FakeSTT(script), llm=llm or FakeLLM(), tts=tts or FakeTTS(),
                 player=player or FakePlayer(), lights=FakeLights(), face=face or FakeFace())
    clock = Clock()
    ctl = Controller(cfg=ConversationConfig(), clock=clock, **parts)
    return ctl, parts, clock


def test_wake_and_question_one_breath():
    ctl, p, _ = make([Transcript("Mirror, mirror, who is fairest?", "en")])
    ctl.on_segment(SEG)
    assert p["llm"].calls == [([], "who is fairest", "en")]
    assert p["lights"].states == [State.THINKING, State.SPEAKING, State.LISTENING]
    assert p["tts"].calls == [("Thou art fairest.", "en")]
    assert ctl.state == State.LISTENING


def test_wake_then_question_next_segment():
    ctl, p, _ = make([Transcript("מראה מראה", "he"), Transcript("מי הכי יפה", "he")])
    ctl.on_segment(SEG)
    assert ctl.state == State.LISTENING
    assert p["llm"].calls == []
    ctl.on_segment(SEG)
    assert p["llm"].calls == [([], "מי הכי יפה", "he")]


def test_listening_timeout_returns_idle():
    ctl, p, clock = make([Transcript("mirror mirror", "en")])
    ctl.on_segment(SEG)
    clock.t += 7.9
    ctl.tick()
    assert ctl.state == State.LISTENING
    clock.t += 0.2
    ctl.tick()
    assert ctl.state == State.IDLE
    assert p["lights"].states[-1] == State.IDLE
    assert p["face"].msgs[-1] == (State.IDLE, 0.0, "X")


def test_followup_without_wake():
    ctl, p, clock = make([Transcript("mirror mirror who is fairest", "en"),
                          Transcript("And who is second?", "en")],
                         llm=FakeLLM(["You are.", "Also you."]))
    ctl.on_segment(SEG)
    clock.t += 5.0
    ctl.tick()
    assert ctl.state == State.LISTENING
    ctl.on_segment(SEG)
    history, text, lang = p["llm"].calls[1]
    assert text == "And who is second?" and lang == "en"
    assert history == [{"role": "user", "content": "who is fairest"},
                       {"role": "assistant", "content": "You are."}]


def test_followup_window_expires_and_clears_history():
    ctl, p, clock = make([Transcript("mirror mirror who is fairest", "en"),
                          Transcript("mirror mirror again", "en")])
    ctl.on_segment(SEG)
    clock.t += 6.1
    ctl.tick()
    assert ctl.state == State.IDLE
    ctl.on_segment(SEG)
    assert p["llm"].calls[1] == ([], "again", "en")


def test_history_keeps_last_six():
    script = [Transcript("mirror mirror q0", "en")] + [
        Transcript(f"q{i}", "en") for i in range(1, 5)]
    ctl, p, _ = make(script)
    for _ in script:
        ctl.on_segment(SEG)
    history = p["llm"].calls[-1][0]
    assert len(history) == 6
    assert history[0] == {"role": "user", "content": "q1"}


def test_wake_phrase_in_listening_uses_remainder():
    ctl, p, _ = make([Transcript("mirror mirror", "en"),
                      Transcript("mirror mirror", "en"),
                      Transcript("Mirror mirror, who?", "en")])
    ctl.on_segment(SEG)
    ctl.on_segment(SEG)
    assert ctl.state == State.LISTENING and p["llm"].calls == []
    ctl.on_segment(SEG)
    assert p["llm"].calls == [([], "who", "en")]


def test_non_wake_ignored_in_idle():
    ctl, p, _ = make([Transcript("hello there", "en")])
    ctl.on_segment(SEG)
    assert p["llm"].calls == []
    assert ctl.state == State.IDLE
    assert p["lights"].states == []


def test_hallucination_in_listening_ignored():
    ctl, p, _ = make([Transcript("mirror mirror", "en"), Transcript("Thank you.", "en")])
    ctl.on_segment(SEG)
    ctl.on_segment(SEG)
    assert ctl.state == State.LISTENING
    assert p["llm"].calls == []


def test_segments_during_speaking_dropped():
    holder = {}
    player = FakePlayer(during=lambda: holder["ctl"].on_segment(SEG))
    ctl, p, _ = make([Transcript("mirror mirror who is fairest", "en")], player=player)
    holder["ctl"] = ctl
    ctl.on_segment(SEG)
    assert p["stt"].calls == 1
    assert ctl.state == State.LISTENING


def test_llm_failure_speaks_fallback():
    ctl, p, _ = make([Transcript("מראה מראה מי הכי יפה", "he")],
                     llm=FakeLLM([LLMError("down")]))
    ctl.on_segment(SEG)
    assert p["tts"].calls == [(FALLBACK["he"], "he")]
    assert p["player"].plays == 1
    assert ctl.state == State.IDLE
    assert p["lights"].states == [State.THINKING, State.SPEAKING, State.IDLE]


def test_stt_failure_stays_idle():
    ctl, p, _ = make([STTError("boom")])
    ctl.on_segment(SEG)
    assert ctl.state == State.IDLE
    assert p["player"].plays == 0
    assert p["lights"].states == []


def test_stt_failure_in_listening_stays_listening():
    ctl, p, _ = make([Transcript("mirror mirror", "en"), STTError("boom")])
    ctl.on_segment(SEG)
    ctl.on_segment(SEG)
    assert ctl.state == State.LISTENING
    assert p["player"].plays == 0


def test_tts_failure_goes_idle_without_raising():
    class BrokenOnSecondTTS(FakeTTS):
        def synth_visemes(self, text, lang):
            if self.calls:
                raise RuntimeError("no voice")
            return super().synth_visemes(text, lang)

    ctl, p, _ = make([Transcript("mirror mirror who", "en"),
                      Transcript("and then?", "en"),
                      Transcript("mirror mirror again", "en")],
                     tts=BrokenOnSecondTTS())
    ctl.on_segment(SEG)  # answered, history now has 2 messages
    ctl.on_segment(SEG)  # follow-up; TTS raises
    assert ctl.state == State.IDLE
    assert p["lights"].states[-1] == State.IDLE
    ctl.on_segment(SEG)
    assert p["llm"].calls[-1][0] == []  # history was cleared


class RaisingFace(FakeFace):
    def send(self, state, mouth, viseme="X"):
        super().send(state, mouth, viseme)
        raise OSError("bus down")


def test_failing_face_bus_never_escapes():
    ctl, p, clock = make([Transcript("mirror mirror who", "en"),
                          Transcript("mirror mirror", "en")], face=RaisingFace())
    ctl.on_segment(SEG)  # full answer with every face send raising
    assert ctl.state == State.LISTENING
    assert p["player"].plays == 1
    assert (State.SPEAKING, 0.9, "D") in p["face"].msgs  # playback not aborted
    assert p["lights"].states == [State.THINKING, State.SPEAKING, State.LISTENING]
    clock.t += 6.1
    ctl.tick()
    assert ctl.state == State.IDLE


def test_failing_face_bus_in_error_path_never_escapes():
    ctl, p, _ = make([Transcript("mirror mirror who", "en")],
                     llm=FakeLLM([RuntimeError("unexpected")]), face=RaisingFace())
    ctl.on_segment(SEG)
    assert ctl.state == State.IDLE


def test_mouth_levels_forwarded():
    ctl, p, _ = make([Transcript("mirror mirror who is fairest", "en")],
                     player=FakePlayer(levels=(0.3, 0.8, 0.0)))
    ctl.on_segment(SEG)
    msgs = p["face"].msgs
    start = msgs.index((State.SPEAKING, 0.0, "X"))
    assert [m[:2] for m in msgs[start + 1:start + 4]] == [
        (State.SPEAKING, 0.3), (State.SPEAKING, 0.8), (State.SPEAKING, 0.0)]
    assert msgs[-1] == (State.LISTENING, 0.0, "X")


def speaking_msgs(face):
    """Face messages sent during playback (after the SPEAKING state change)."""
    start = face.msgs.index((State.SPEAKING, 0.0, "X"))
    return [m for m in face.msgs[start + 1:] if m[0] == State.SPEAKING]


def test_controller_forwards_visemes():
    track = VisemeTrack(np.array([0.0, 0.1]), ["X", "D"])
    levels = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    ctl, p, _ = make([Transcript("mirror mirror who is fairest", "en")],
                     tts=FakeTTS(track=track), player=FakePlayer(levels=levels))
    ctl.on_segment(SEG)
    msgs = speaking_msgs(p["face"])
    assert msgs[5] == (State.SPEAKING, 0.6, "D")  # hop 5: t = 0.10 s + 0.05 s lead
    assert msgs[2] == (State.SPEAKING, 0.3, "X")  # hop 2: t = 0.09 s
    assert msgs[3] == (State.SPEAKING, 0.4, "D")  # hop 3: t = 0.11 s


def test_controller_rms_fallback_when_no_track():
    levels = (0.0, 0.1, 0.5, 0.9)
    ctl, p, _ = make([Transcript("mirror mirror who is fairest", "en")],
                     tts=FakeTTS(track=None), player=FakePlayer(levels=levels))
    ctl.on_segment(SEG)
    assert speaking_msgs(p["face"]) == [(State.SPEAKING, 0.0, "X"), (State.SPEAKING, 0.1, "B"),
                                        (State.SPEAKING, 0.5, "C"), (State.SPEAKING, 0.9, "D")]


def test_hop_count_restarts_for_each_reply():
    track = VisemeTrack(np.array([0.0, 0.1]), ["D", "X"])
    ctl, p, _ = make([Transcript("mirror mirror who", "en"), Transcript("and then?", "en")],
                     tts=FakeTTS(track=track), player=FakePlayer(levels=(0.5, 0.5, 0.5, 0.5)))
    ctl.on_segment(SEG)
    ctl.on_segment(SEG)
    shapes = [m[2] for m in p["face"].msgs if m[0] == State.SPEAKING and m[1] == 0.5]
    assert shapes == ["D", "D", "D", "X"] * 2
