import re
import threading

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

    def reply_stream(self, history, user_text, lang, cancel=None):
        yield from re.split(r"(?<=[.!?…])\s+", self.reply(history, user_text, lang))


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
        self.played = []  # the samples of each play, in order

    def play(self, samples, sr, on_level):
        self.plays += 1
        self.played.append(samples)
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


def make(script, llm=None, player=None, tts=None, face=None, fillers=None, cfg=None,
         first_sentence_timeout_s=20.0):
    parts = dict(stt=FakeSTT(script), llm=llm or FakeLLM(), tts=tts or FakeTTS(),
                 player=player or FakePlayer(), lights=FakeLights(), face=face or FakeFace())
    clock = Clock()
    ctl = Controller(cfg=cfg or ConversationConfig(), clock=clock, fillers=fillers,
                     first_sentence_timeout_s=first_sentence_timeout_s, **parts)
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


# --- fillers and streamed replies ---------------------------------------------

FILLER_CLIP = np.full(441, 0.25, dtype=np.float32)


class FakeFillers:
    def __init__(self, track=None):
        self.track = track
        self.picks = []

    def pick(self, lang):
        self.picks.append(lang)
        return FILLER_CLIP, 22050, self.track


class TextTTS(FakeTTS):
    """Synthesizes each text to a clip that carries the text, so plays can be identified."""

    def synth_visemes(self, text, lang):
        self.calls.append((text, lang))
        return TextClip(text), 22050, self.track


class TextClip(np.ndarray):
    def __new__(cls, text):
        obj = np.zeros(2205, dtype=np.float32).view(cls)
        obj.text = text
        return obj

    def __array_finalize__(self, obj):
        self.text = getattr(obj, "text", None)


def played_texts(player):
    return [getattr(s, "text", None) if s is not FILLER_CLIP else "FILLER"
            for s in player.played]


class ScriptedLLM:
    """reply_stream yields the given sentences; an Exception item is raised."""

    def __init__(self, items, before=None):
        self.items = items
        self.before = before or {}  # index -> callable run before yielding that item
        self.calls = []
        self.finished = threading.Event()

    def reply_stream(self, history, user_text, lang, cancel=None):
        self.calls.append((list(history), user_text, lang))
        try:
            for i, item in enumerate(self.items):
                if i in self.before:
                    self.before[i]()
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            self.finished.set()


def test_filler_plays_before_reply():
    llm = ScriptedLLM(["Stars align.", "You will dance."])
    ctl, p, _ = make([Transcript("mirror mirror what will happen", "en")], llm=llm,
                     tts=TextTTS(), fillers=FakeFillers())
    ctl.on_segment(SEG)
    assert played_texts(p["player"]) == ["FILLER", "Stars align.", "You will dance."]
    states = p["lights"].states
    assert states[:2] == [State.THINKING, State.SPEAKING]
    assert states[-1] == State.LISTENING
    assert State.IDLE not in states
    assert p["tts"].calls == [("Stars align.", "en"), ("You will dance.", "en")]


def test_filler_visemes_forwarded():
    track = VisemeTrack(np.array([0.0]), ["D"])
    ctl, p, _ = make([Transcript("mirror mirror who", "en")], llm=ScriptedLLM(["Hm."]),
                     fillers=FakeFillers(track=track), player=FakePlayer(levels=(0.4,)))
    ctl.on_segment(SEG)
    assert speaking_msgs(p["face"])[0] == (State.SPEAKING, 0.4, "D")


def test_filler_disabled_skips():
    fillers = FakeFillers()
    ctl, p, _ = make([Transcript("mirror mirror who", "en")], llm=ScriptedLLM(["Yes."]),
                     tts=TextTTS(), fillers=fillers,
                     cfg=ConversationConfig(filler_enabled=False))
    ctl.on_segment(SEG)
    assert played_texts(p["player"]) == ["Yes."]
    assert fillers.picks == []
    assert p["lights"].states == [State.THINKING, State.SPEAKING, State.LISTENING]


def test_filler_skipped_when_sentence_arrives_within_min_wait():
    ctl, p, _ = make([Transcript("mirror mirror who", "en")], llm=ScriptedLLM(["Yes."]),
                     tts=TextTTS(), fillers=FakeFillers(),
                     cfg=ConversationConfig(filler_min_wait_s=5.0))
    ctl.on_segment(SEG)
    assert played_texts(p["player"]) == ["Yes."]


def test_back_to_thinking_after_filler_while_waiting():
    gate = threading.Event()
    llm = ScriptedLLM(["Late answer."], before={0: lambda: gate.wait(2.0)})
    player = FakePlayer(levels=(0.1,))
    orig = player.play

    def play(samples, sr, on_level):
        orig(samples, sr, on_level)
        gate.set()  # the filler is over; now let the LLM answer

    player.play = play
    ctl, p, _ = make([Transcript("mirror mirror who", "en")], llm=llm, tts=TextTTS(),
                     fillers=FakeFillers(), player=player)
    ctl.on_segment(SEG)
    assert p["lights"].states == [State.THINKING, State.SPEAKING, State.THINKING,
                                  State.SPEAKING, State.LISTENING]
    assert played_texts(p["player"]) == ["FILLER", "Late answer."]


def test_first_sentence_synth_while_second_generates():
    first_played = threading.Event()
    seen = {}

    def before_second():
        seen["first_played"] = first_played.wait(2.0)

    llm = ScriptedLLM(["One.", "Two."], before={1: before_second})
    player = FakePlayer()
    orig = player.play

    def play(samples, sr, on_level):
        orig(samples, sr, on_level)
        if getattr(samples, "text", None) == "One.":
            first_played.set()

    player.play = play
    ctl, p, _ = make([Transcript("mirror mirror who", "en")], llm=llm, tts=TextTTS(),
                     player=player)
    ctl.on_segment(SEG)
    assert seen["first_played"] is True  # sentence 1 was heard before sentence 2 existed
    assert played_texts(p["player"]) == ["One.", "Two."]


def test_stream_failure_before_first_sentence_uses_fallback():
    ctl, p, _ = make([Transcript("mirror mirror who", "en")],
                     llm=ScriptedLLM([LLMError("down")]), tts=TextTTS(), fillers=FakeFillers())
    ctl.on_segment(SEG)
    assert played_texts(p["player"]) == ["FILLER", FALLBACK["en"]]
    assert ctl.state == State.IDLE
    assert p["lights"].states[-1] == State.IDLE


def test_first_sentence_timeout_uses_fallback():
    release = threading.Event()
    llm = ScriptedLLM(["Too late."], before={0: lambda: release.wait(5.0)})
    tts = TextTTS()
    player = FakePlayer()
    orig = player.play

    def play(samples, sr, on_level):
        orig(samples, sr, on_level)
        if getattr(samples, "text", None) == FALLBACK["en"]:
            release.set()

    player.play = play
    ctl, p, _ = make([Transcript("mirror mirror who", "en")], llm=llm, tts=tts,
                     player=player, first_sentence_timeout_s=0.2)
    ctl.on_segment(SEG)
    assert played_texts(p["player"]) == [FALLBACK["en"]]
    assert ctl.state == State.IDLE
    assert llm.finished.is_set()  # the stream worker was stopped before on_segment returned
    assert "Too late." not in [t for t, _ in tts.calls]


def test_history_gets_joined_reply():
    ctl, p, _ = make([Transcript("mirror mirror who", "en"), Transcript("and then?", "en")],
                     llm=ScriptedLLM(["Stars align.", "You will dance."]))
    ctl.on_segment(SEG)
    ctl.on_segment(SEG)
    assert p["llm"].calls[1][0] == [{"role": "user", "content": "who"},
                                    {"role": "assistant",
                                     "content": "Stars align. You will dance."}]


def test_workers_stopped_when_playback_interrupted():
    class Interrupt(BaseException):
        pass

    llm = ScriptedLLM(["One.", "Two.", "Three."])
    player = FakePlayer()

    def play(samples, sr, on_level):
        raise Interrupt()

    player.play = play
    ctl, p, _ = make([Transcript("mirror mirror who", "en")], llm=llm, tts=TextTTS(),
                     player=player)
    before = set(threading.enumerate())
    try:
        ctl.on_segment(SEG)
    except Interrupt:
        pass
    else:
        raise AssertionError("BaseException should propagate")
    assert set(threading.enumerate()) - before == set()  # no worker left running
    assert llm.finished.is_set()


def test_segments_during_filler_dropped():
    holder = {}
    player = FakePlayer(during=lambda: holder["ctl"].on_segment(SEG))
    ctl, p, _ = make([Transcript("mirror mirror who", "en")], llm=ScriptedLLM(["Yes."]),
                     player=player, fillers=FakeFillers())
    holder["ctl"] = ctl
    ctl.on_segment(SEG)
    assert p["stt"].calls == 1
    assert p["player"].plays == 2
    assert ctl.state == State.LISTENING
