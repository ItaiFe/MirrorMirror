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


# --- mid-reply stalls, partial replies, re-warming ---------------------------------

def test_mid_reply_stall_shows_thinking_until_next_clip():
    llm = ScriptedLLM(["One.", "Two."], before={1: lambda: threading.Event().wait(0.9)})
    ctl, p, _ = make([Transcript("mirror mirror who", "en")], llm=llm, tts=TextTTS())
    ctl.on_segment(SEG)
    assert played_texts(p["player"]) == ["One.", "Two."]
    assert p["lights"].states == [State.THINKING, State.SPEAKING, State.THINKING,
                                  State.SPEAKING, State.LISTENING]
    assert (State.THINKING, 0.0, "X") in p["face"].msgs[2:]


def test_no_thinking_when_next_clip_is_quick():
    llm = ScriptedLLM(["One.", "Two."], before={1: lambda: threading.Event().wait(0.2)})
    ctl, p, _ = make([Transcript("mirror mirror who", "en")], llm=llm, tts=TextTTS())
    ctl.on_segment(SEG)
    assert p["lights"].states == [State.THINKING, State.SPEAKING, State.LISTENING]


def test_spoken_sentences_kept_when_tts_fails_mid_reply():
    class BrokenOnSecond(TextTTS):
        def synth_visemes(self, text, lang):
            if text == "Two.":
                raise RuntimeError("no voice")
            return super().synth_visemes(text, lang)

    llm = ScriptedLLM(["One.", "Two.", "Three."])
    ctl, p, _ = make([Transcript("mirror mirror who", "en"), Transcript("and then?", "en")],
                     llm=llm, tts=BrokenOnSecond())
    ctl.on_segment(SEG)
    assert played_texts(p["player"]) == ["One."]
    assert ctl.state == State.LISTENING
    ctl.on_segment(SEG)
    assert llm.calls[1][0] == [{"role": "user", "content": "who"},
                               {"role": "assistant", "content": "One."}]


def test_spoken_sentences_kept_when_playback_fails_mid_reply():
    player = FakePlayer()
    orig = player.play

    def play(samples, sr, on_level):
        if getattr(samples, "text", None) == "Two.":
            raise OSError("speaker gone")
        orig(samples, sr, on_level)

    player.play = play
    llm = ScriptedLLM(["One.", "Two.", "Three."])
    ctl, p, _ = make([Transcript("mirror mirror who", "en"), Transcript("and then?", "en")],
                     llm=llm, tts=TextTTS(), player=player)
    before = set(threading.enumerate())
    ctl.on_segment(SEG)
    assert set(threading.enumerate()) - before == set()
    assert ctl.state == State.LISTENING
    ctl.on_segment(SEG)
    assert llm.calls[1][0][-1] == {"role": "assistant", "content": "One."}


class WarmLLM(FakeLLM):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.warmed = []
        self.done = threading.Event()

    def warm(self, lang):
        self.warmed.append(lang)
        self.done.set()
        return 1.0


def _wait_rewarm(ctl):
    t = ctl._rewarm_thread
    if t is not None:
        t.join(2.0)


def test_tick_rewarms_idle_language_slot():
    llm = WarmLLM()
    ctl, p, clock = make([Transcript("mirror mirror who", "en")], llm=llm)
    clock.t += 599
    ctl.tick()
    _wait_rewarm(ctl)
    assert llm.warmed == []
    ctl.on_segment(SEG)  # an English request at t0 + 599
    clock.t += 2
    ctl.tick()
    _wait_rewarm(ctl)
    assert llm.warmed == ["he"]  # en was used 2 s ago
    clock.t += 1
    ctl.tick()
    _wait_rewarm(ctl)
    assert llm.warmed == ["he"]  # he was just re-warmed
    clock.t += 600  # both stale now: one language per re-warm, the stalest first
    ctl.tick()
    _wait_rewarm(ctl)
    assert llm.warmed == ["he", "en"]
    ctl.tick()
    _wait_rewarm(ctl)
    assert llm.warmed == ["he", "en", "he"]


def test_no_rewarm_while_busy():
    llm = WarmLLM()
    holder = {}

    def during():
        holder["clock"].t += 700
        holder["ctl"].tick()

    ctl, p, clock = make([Transcript("mirror mirror who", "en")], llm=llm,
                         player=FakePlayer(during=during))
    holder.update(ctl=ctl, clock=clock)
    ctl.on_segment(SEG)
    _wait_rewarm(ctl)
    assert llm.warmed == []


def test_rewarm_failure_is_logged_not_raised(caplog):
    class BadWarm(FakeLLM):
        def warm(self, lang):
            raise LLMError("down")

    llm = BadWarm()
    llm.tries = []
    llm.warm = lambda lang: (llm.tries.append(lang), BadWarm.warm(llm, lang))
    ctl, p, clock = make([], llm=llm)
    ctl.mark_unwarmed("he")  # e.g. llama-server was still loading at start-up
    ctl.tick()
    _wait_rewarm(ctl)
    assert "down" in caplog.text
    assert llm.tries == ["he"]
    clock.t += 29
    ctl.tick()
    _wait_rewarm(ctl)
    assert llm.tries == ["he"]
    clock.t += 2  # retried REWARM_RETRY_S after the failure
    ctl.tick()
    _wait_rewarm(ctl)
    assert llm.tries == ["he", "he"]


def test_unwarmed_language_is_warmed_at_first_idle_tick():
    llm = WarmLLM()
    ctl, p, clock = make([], llm=llm)
    ctl.mark_unwarmed("he")
    ctl.tick()
    _wait_rewarm(ctl)
    assert llm.warmed == ["he"]
    ctl.tick()
    _wait_rewarm(ctl)
    assert llm.warmed == ["he"]


class SlowWarmLLM(FakeLLM):
    def __init__(self, warm_s, *a, **kw):
        super().__init__(*a, **kw)
        self.warm_s = warm_s
        self.events = []

    def warm(self, lang):
        self.events.append(("warm start", lang))
        threading.Event().wait(self.warm_s)
        self.events.append(("warm end", lang))

    def reply(self, history, user_text, lang):
        self.events.append(("reply", lang))
        return super().reply(history, user_text, lang)


def test_answer_waits_for_inflight_rewarm():
    llm = SlowWarmLLM(0.2)
    ctl, p, clock = make([Transcript("mirror mirror who", "en")], llm=llm)
    ctl.mark_unwarmed("en")
    ctl.tick()  # re-warm of en starts in the background
    ctl.on_segment(SEG)
    assert llm.events == [("warm start", "en"), ("warm end", "en"), ("reply", "en")]


def test_answer_waits_for_rewarm_only_briefly(caplog):
    import logging
    import time

    llm = SlowWarmLLM(1.5)
    ctl, p, clock = make([Transcript("mirror mirror who", "en")], llm=llm)
    ctl.mark_unwarmed("en")
    ctl.tick()
    t = time.monotonic()
    with caplog.at_level(logging.INFO, logger="mirror.conversation"):
        ctl.on_segment(SEG)
    assert time.monotonic() - t < 1.2
    assert llm.events[:2] == [("warm start", "en"), ("reply", "en")]
    assert "still running" in caplog.text
    ctl.close(timeout_s=2.0)


def test_close_joins_rewarm_and_stops_new_ones():
    llm = SlowWarmLLM(0.3)
    ctl, p, clock = make([], llm=llm)
    ctl.mark_unwarmed("en")
    ctl.tick()
    thread = ctl._rewarm_thread
    ctl.close(timeout_s=2.0)
    assert not thread.is_alive()
    assert llm.events == [("warm start", "en"), ("warm end", "en")]
    clock.t += 10_000
    ctl.tick()
    assert ctl._rewarm_thread is thread  # no new re-warm after close


def _connection_error(msg):
    import requests

    try:
        raise LLMError(msg) from requests.exceptions.ConnectionError("refused")
    except LLMError as e:
        return e


def test_connection_error_rewarms_both_slots():
    llm = WarmLLM([_connection_error("down")])
    ctl, p, clock = make([Transcript("mirror mirror who", "en")], llm=llm)
    ctl.on_segment(SEG)  # llama-server restarted: the request fails
    assert ctl.state == State.IDLE
    ctl.tick()
    _wait_rewarm(ctl)
    ctl.tick()
    _wait_rewarm(ctl)
    assert sorted(llm.warmed) == ["en", "he"]


def test_other_llm_errors_do_not_trigger_rewarm():
    llm = WarmLLM([LLMError("empty reply")])
    ctl, p, clock = make([Transcript("mirror mirror who", "en")], llm=llm)
    ctl.on_segment(SEG)
    ctl.tick()
    _wait_rewarm(ctl)
    assert llm.warmed == []


def test_next_sentence_synthesized_while_current_one_plays():
    """Synth-ahead: sentence 2 is synthesized during sentence 1's playback, not after."""
    synth_two = threading.Event()
    seen = {}

    class AheadTTS(TextTTS):
        def synth_visemes(self, text, lang):
            if text == "Two.":
                synth_two.set()
            return super().synth_visemes(text, lang)

    player = FakePlayer()
    orig = player.play

    def play(samples, sr, on_level):
        if getattr(samples, "text", None) == "One.":
            seen["overlap"] = synth_two.wait(2.0)
        orig(samples, sr, on_level)

    player.play = play
    ctl, p, _ = make([Transcript("mirror mirror who", "en")],
                     llm=ScriptedLLM(["One.", "Two."]), tts=AheadTTS(), player=player)
    ctl.on_segment(SEG)
    assert seen["overlap"] is True
    assert played_texts(p["player"]) == ["One.", "Two."]


# --- idle lures ----------------------------------------------------------------

import datetime as _dt  # noqa: E402
import random as _random  # noqa: E402

import pytest  # noqa: E402

from mirror.conversation import in_quiet_hours, parse_quiet_hours  # noqa: E402

LURE_CLIP = np.full(441, 0.75, dtype=np.float32)


class FakeLures:
    def __init__(self, langs=("en", "he"), track=None):
        self.langs = list(langs)
        self.track = track
        self.picks = []

    def languages(self):
        return list(self.langs)

    def pick(self, lang):
        self.picks.append(lang)
        return LURE_CLIP, 22050, self.track


class WallClock:
    def __init__(self, hhmm="14:00"):
        self.set(hhmm)

    def set(self, hhmm):
        h, m = map(int, hhmm.split(":"))
        self.now = _dt.datetime(2026, 8, 30, h, m)

    def __call__(self):
        return self.now


LURE_CFG = dict(lures_enabled=True, lure_min_idle_s=120, lure_interval_s=[180, 180])


def make_lured(script=(), lures=None, wall=None, cfg=None, **kw):
    lures = lures if lures is not None else FakeLures()
    wall = wall or WallClock()
    parts = dict(stt=FakeSTT(list(script)), llm=kw.pop("llm", None) or FakeLLM(),
                 tts=kw.pop("tts", None) or TextTTS(),
                 player=kw.pop("player", None) or FakePlayer(),
                 lights=FakeLights(), face=FakeFace())
    clock = Clock()
    ctl = Controller(cfg=cfg or ConversationConfig(**LURE_CFG), clock=clock, lures=lures,
                     wall_clock=wall, rng=_random.Random(7), **parts, **kw)
    return ctl, parts, clock, lures, wall


def lure_plays(player):
    return sum(1 for s in player.played if s is LURE_CLIP)


def test_lure_config_defaults():
    c = ConversationConfig()
    assert c.lures_enabled is True and c.lure_min_idle_s == 120
    assert list(c.lure_interval_s) == [180, 420] and c.lure_quiet_hours == []


def test_no_lure_before_min_idle_and_interval():
    ctl, p, clock, lures, _ = make_lured()
    clock.t += 179
    ctl.tick()
    assert lures.picks == [] and p["player"].plays == 0


def test_lure_after_min_idle_and_interval():
    ctl, p, clock, lures, _ = make_lured()
    clock.t += 180
    ctl.tick()
    assert lure_plays(p["player"]) == 1 and len(lures.picks) == 1
    assert p["lights"].states == [State.SPEAKING, State.IDLE]
    assert ctl.state == State.IDLE
    # the next one waits for a fresh interval from this lure
    clock.t += 179
    ctl.tick()
    assert lure_plays(p["player"]) == 1
    clock.t += 1
    ctl.tick()
    assert lure_plays(p["player"]) == 2


def test_min_idle_gates_a_short_interval():
    cfg = ConversationConfig(lures_enabled=True, lure_min_idle_s=120,
                             lure_interval_s=[10, 10])
    ctl, p, clock, lures, _ = make_lured(cfg=cfg)
    clock.t += 119
    ctl.tick()
    assert lures.picks == []
    clock.t += 1
    ctl.tick()
    assert len(lures.picks) == 1


def test_interval_is_randomized_within_range():
    cfg = ConversationConfig(lures_enabled=True, lure_min_idle_s=0,
                             lure_interval_s=[100, 200])
    ctl, p, clock, lures, _ = make_lured(cfg=cfg)
    gaps, last = [], clock.t
    for _ in range(4000):
        clock.t += 1
        ctl.tick()
        if len(lures.picks) > len(gaps):
            gaps.append(clock.t - last)
            last = clock.t
    assert len(gaps) >= 20 and all(100 <= g <= 201 for g in gaps)
    assert len(set(gaps)) > 3


def test_lure_face_gets_visemes():
    track = VisemeTrack(np.array([0.0]), ["B"])
    ctl, p, clock, _, _ = make_lured(lures=FakeLures(track=track),
                                     player=FakePlayer(levels=(0.4,)))
    clock.t += 180
    ctl.tick()
    assert (State.SPEAKING, 0.4, "B") in p["face"].msgs
    assert p["face"].msgs[-1] == (State.IDLE, 0.0, "X")


def test_lure_languages_random_both_used():
    ctl, p, clock, lures, _ = make_lured()
    for _ in range(30):
        clock.t += 180
        ctl.tick()
    assert set(lures.picks) == {"en", "he"}
    assert len(lures.picks) == 30


def test_lures_disabled_or_absent():
    ctl, p, clock, lures, _ = make_lured(cfg=ConversationConfig(
        lures_enabled=False, lure_min_idle_s=0, lure_interval_s=[1, 1]))
    clock.t += 1000
    ctl.tick()
    assert lures.picks == []
    ctl, p, clock = make([])  # no lure bank at all
    clock.t += 10_000
    ctl.tick()
    assert p["player"].plays == 0


def test_no_lure_when_bank_is_empty():
    ctl, p, clock, lures, _ = make_lured(lures=FakeLures(langs=()))
    clock.t += 1000
    ctl.tick()
    assert p["player"].plays == 0 and ctl.state == State.IDLE


@pytest.mark.parametrize("window,now,quiet", [
    ("02:00-08:00", "01:59", False),
    ("02:00-08:00", "02:00", True),
    ("02:00-08:00", "07:59", True),
    ("02:00-08:00", "08:00", False),
    ("22:30-06:00", "23:00", True),   # crosses midnight
    ("22:30-06:00", "00:10", True),
    ("22:30-06:00", "05:59", True),
    ("22:30-06:00", "06:00", False),
    ("22:30-06:00", "12:00", False),
    ("22:30-06:00", "22:29", False),
])
def test_in_quiet_hours(window, now, quiet):
    h, m = map(int, now.split(":"))
    assert in_quiet_hours(parse_quiet_hours([window]), _dt.time(h, m)) is quiet


def test_parse_quiet_hours_rejects_bad_windows():
    assert parse_quiet_hours([]) == []
    for bad in (["2-8"], ["02:00"], ["25:00-08:00"], ["02:00-08:60"], ["03:00-03:00"]):
        with pytest.raises(ValueError):
            parse_quiet_hours(bad)


def test_bad_quiet_hours_fail_at_startup():
    with pytest.raises(ValueError):
        make_lured(cfg=ConversationConfig(lure_quiet_hours=["nonsense"]))


def test_no_lure_during_quiet_hours_across_midnight():
    cfg = ConversationConfig(lures_enabled=True, lure_min_idle_s=120,
                             lure_interval_s=[180, 180], lure_quiet_hours=["22:00-06:30"])
    wall = WallClock("23:30")
    ctl, p, clock, lures, _ = make_lured(cfg=cfg, wall=wall)
    clock.t += 1000
    ctl.tick()
    wall.set("03:00")
    ctl.tick()
    assert lures.picks == []
    wall.set("06:30")  # quiet hours over: the overdue lure plays
    ctl.tick()
    assert len(lures.picks) == 1


def test_no_lure_while_not_idle():
    ctl, p, clock, lures, _ = make_lured([Transcript("mirror mirror", "en")])
    ctl.on_segment(SEG)
    assert ctl.state == State.LISTENING
    clock.t += 5  # within listen_timeout_s
    ctl.tick()
    assert lures.picks == []


def test_no_lure_while_busy():
    holder = {}

    def tick_mid_answer():
        holder["clock"].t += 1000
        holder["ctl"].tick()

    ctl, p, clock, lures, _ = make_lured(
        [Transcript("mirror mirror who am I", "en")],
        player=FakePlayer(levels=(0.5,), during=tick_mid_answer))
    holder.update(ctl=ctl, clock=clock)
    ctl.on_segment(SEG)
    assert lures.picks == [] and p["player"].plays == 1


def test_interaction_resets_the_idle_timer():
    ctl, p, clock, lures, _ = make_lured([Transcript("mirror mirror", "en")])
    clock.t += 170
    ctl.on_segment(SEG)  # wakes: LISTENING
    clock.t += ConversationConfig().listen_timeout_s
    ctl.tick()  # listening times out -> IDLE, the idle timer starts again
    assert ctl.state == State.IDLE
    clock.t += 179
    ctl.tick()
    assert lures.picks == []
    clock.t += 1
    ctl.tick()
    assert len(lures.picks) == 1


def test_non_wake_chatter_does_not_reset_the_idle_timer():
    ctl, p, clock, lures, _ = make_lured([Transcript("nice art car", "en")])
    clock.t += 170
    ctl.on_segment(SEG)
    clock.t += 10
    ctl.tick()
    assert len(lures.picks) == 1


def test_segment_during_lure_dropped_and_after_lure_handled():
    holder = {}

    def mid_lure():
        holder["ctl"].on_segment(SEG)  # dropped: busy

    ctl, p, clock, lures, _ = make_lured(
        [Transcript("mirror mirror who is fairest", "en")],
        player=FakePlayer(levels=(0.5,), during=mid_lure), llm=FakeLLM())
    holder["ctl"] = ctl
    clock.t += 180
    ctl.tick()
    assert p["stt"].calls == 0 and lure_plays(p["player"]) == 1
    p["player"].during = None
    ctl.on_segment(SEG)
    assert p["stt"].calls == 1
    assert p["llm"].calls == [([], "who is fairest", "en")]
    assert ctl.state == State.LISTENING


def test_lure_leaves_history_untouched():
    ctl, p, clock, lures, _ = make_lured(
        [Transcript("mirror mirror who is fairest", "en")])
    ctl.on_segment(SEG)
    history = [dict(m) for m in ctl._history]
    assert history
    ctl._state = State.IDLE  # as if idle again, history kept (white-box)
    clock.t += 1000
    ctl.tick()
    assert lures.picks and ctl._history == history
    assert len(p["llm"].calls) == 1  # the lure never reached the LLM


def test_lure_failure_never_escapes_tick(caplog):
    class BadPlayer(FakePlayer):
        def play(self, samples, sr, on_level):
            raise OSError("speaker gone")

    ctl, p, clock, lures, _ = make_lured(player=BadPlayer())
    clock.t += 180
    with caplog.at_level("ERROR", logger="mirror.conversation"):
        ctl.tick()
    assert ctl.state == State.IDLE and not ctl._busy
    assert "lure failed" in caplog.text
    # it does not retry on every tick
    clock.t += 1
    ctl.tick()
    assert len(lures.picks) == 1


def test_lure_with_failing_sinks_still_returns_to_idle():
    class BadLights(FakeLights):
        def set_state(self, state):
            raise OSError("serial gone")

    ctl, p, clock, lures, _ = make_lured()
    ctl._lights = BadLights()
    clock.t += 180
    ctl.tick()
    assert ctl.state == State.IDLE and lure_plays(p["player"]) == 1
