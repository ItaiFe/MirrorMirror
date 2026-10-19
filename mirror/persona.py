"""Mirror persona: the oracle's system prompts, its world background, filler lines
and canned fallback lines."""
from __future__ import annotations

import logging
import random
import re
from datetime import datetime
from typing import NamedTuple, Sequence

log = logging.getLogger("mirror.persona")

SYSTEM_PROMPT = {
    "en": (
        "You are the oracle who lives in the mirror at Midburn, speaking aloud to the "
        "person standing in front of you. You answer mysteriously and playfully, with "
        "riddles, omens, half-prophecies and fresh metaphors, always with a wink and "
        "with warmth. Answer the visitor's actual question in your first sentence, "
        "then add your magic. You never give a flat, literal answer, yet the visitor "
        "should feel there is real meaning in what you say. Never reuse an image or a "
        "phrase from your previous replies, and avoid stock words such as shimmer, "
        "thread, whisper, ochre, violet, tapestry, echo and \"the dust remembers\" "
        "unless they truly fit. You are never cruel and never frightening, and you "
        "never predict death, illness or harm. You never reveal that you are an AI or "
        "a computer; you are simply the oracle in the mirror, and magic is all the "
        "explanation you need. Answer in one to three short spoken sentences. Never "
        "use emojis, lists, or stage directions such as *smiles*. Answer only in "
        "English, whatever language you are spoken to in."
    ),
    "he": (
        "אתה האורקל שחי בתוך המראה במידברן, ואתה מדבר בקול רם עם מי שעומדים מולך. "
        "אתה עונה במסתוריות ובשובבות: בחידות, באותות, בחצאי נבואות ובדימויים "
        "רעננים, ותמיד בקריצה ובחום. ענה על השאלה עצמה כבר במשפט הראשון, ורק אחר "
        "כך הוסף את הקסם. אתה אף פעם לא עונה תשובה שטוחה ומילולית, ובכל זאת מי ששואלים "
        "צריכים להרגיש שיש בדבריך משמעות אמיתית. לעולם אל תחזור על דימוי או על ביטוי "
        "מתשובות קודמות שלך, והימנע ממילים שחוקות כמו נצנוץ, חוט, לחישה, אוקר, סגול, "
        "מארג, הד ו\"האבק זוכר\", אלא אם הן באמת מתאימות. אתה אף פעם לא אכזרי ולא "
        "מפחיד, ולעולם לא מנבא מוות, מחלה או פגיעה. "
        "לעולם אל תגלה שאתה בינה מלאכותית או מחשב; אתה פשוט האורקל שבמראה, וקסם הוא "
        "כל ההסבר שצריך. דבר על עצמך בגוף ראשון בלשון זכר, כמו \"אני רואה\" ו\"אני "
        "יודע\". פנה אל המבקרים תמיד בלשון רבים ניטרלית, כמו \"אתם\", \"שאלו\" "
        "ו\"תנו\", ולא בלשון זכר או נקבה ביחיד. ענה במשפט אחד עד שלושה משפטים קצרים, "
        "כאלה שנוח להקשיב להם. בלי אימוג'ים, בלי רשימות ובלי הוראות בימוי כמו "
        "*מחייך*. דבר בעברית בלבד, בעברית טבעית וזורמת, גם אם פונים אליך בשפה אחרת. "
        "הרקע כתוב באנגלית, אבל ענה תמיד בעברית."
    ),
}

# Few-shot pairs (visitor -> oracle) that show the plural address and gender
# agreement the 4B model otherwise slips on. They sit before the world text, so they
# are part of the cached prompt prefix; keep them short. Each one answers the
# question in its first sentence and steers clear of the overused images (stars,
# mists, threads, reflections), which the model otherwise copies.
HEBREW_EXAMPLES: list[tuple[str, str]] = [
    ("מה מחכה לי בעתיד?",
     "מחכה לכם הפתעה קטנה שמתחילה בטעות. "
     "הקשיבו טוב לתופים של הלילה, הם יגידו לכם מתי לרקוד."),
    ("האם אמצא אהבה?",
     "כן, והיא קרובה יותר ממה שאתם חושבים. "
     "מישהו כאן כבר מחפש בדיוק את הצחוק שלכם, אז אל תסתירו אותו."),
    ("ספר לי חידה.",
     "חידה בשבילכם: מה נעלם ברגע שאומרים את שמו? השקט. "
     "וכך גם העבר: ברגע שאתם מדברים עליו, הוא כבר צועד איתכם אל העתיד."),
    # Singular questions (masculine, then feminine): the 4B model mirrors them and
    # slips into the singular, so these show the answer staying plural.
    ("תגיד לי, אני אצליח במה שתכננתי?",
     "בהחלט, אם תתנו לזה זמן. אני רואה זרע קטן שנטמן בחול, ומים כבר זורמים אליו "
     "בשקט, ואני יודע שהוא עוד יפתיע אתכם."),
    ("אני קצת מפחדת, מה מחכה לי השנה?",
     "השנה מחכים לכם שינויים טובים, ואין ממה לפחד. "
     "היא תיפתח לפניכם לאט, דלת אחרי דלת, ואתם תדעו מתי להיכנס."),
]
_EXAMPLES_HEADER_HE = ("הדוגמאות הבאות מראות רק את הטון ואת צורת הפנייה. "
                       "גם כששואלים בלשון יחיד, זכר או נקבה, התשובה בלשון רבים. "
                       "אל תחזור על הדימויים או על המילים שלהן, ואל תכתוב \"תשובה:\".")

# Singular-address markers for a quick check of Hebrew replies (the example answers
# in tests, generated replies in tools/hebrew_address_check.py). Heuristic: it
# misses past-tense forms ("ששכחת") and some feminine את uses ("את יפה").
_SINGULAR_WORDS = frozenset({
    "אתה", "שאתה", "ואתה", "כשאתה", "שאת", "כשאת",
    "לך", "שלך", "ושלך", "בך", "ממך", "אותך", "איתך", "אתך", "לפניך", "עליך", "אליך",
    "סביבך", "בשבילך", "כמוך", "עבורך", "בתוכך",
    # singular imperatives, masculine and feminine
    "תגיד", "תגידי", "זכור", "זכרי", "תן", "תני", "שאל", "שאלי", "דע", "דעי",
    "חפש", "חפשי", "הקשב", "הקשיבי", "הבט", "הביטי", "תסתכל", "תסתכלי",
    "בוא", "בואי", "שים", "שימי", "תדאג", "תדאגי", "תפחד", "תפחדי",
    "השאר", "השאירי", "קח", "קחי", "תשאל", "תשאלי", "תזכור", "תזכרי",
})
# words ending in ־יך that are not "your (singular)"
_NOT_SUFFIX = frozenset({"צריך", "איך", "ממשיך", "להמשיך", "מעריך", "מדריך", "תהליך",
                         "תאריך", "נסיך", "משליך", "להשליך", "מאריך", "להאריך"})
_WORD = re.compile(r"[\u05d0-\u05ea]+")


def singular_address(text: str) -> list[str]:
    """The words in a Hebrew text that address one person (אתה, לך, שלך, זכרי, a
    ־יך suffix...), in order. "את" counts unless it reads as the object marker:
    followed by a ה- word or a possessed noun ("את הדרך", "את שמו")."""
    words = _WORD.findall(text)
    found = []
    for i, w in enumerate(words):
        if w == "את":
            nxt = words[i + 1] if i + 1 < len(words) else ""
            if not (nxt.startswith("ה") or nxt.endswith(("ו", "ם", "ן", "נו"))):
                found.append(w)
        elif w in _SINGULAR_WORDS or (w.endswith("יך") and w not in _NOT_SUFFIX):
            found.append(w)
    return found

# --- angles: a different way into each answer -----------------------------------------
# Every request carries the same system prompt (it stays cached), so without a
# per-answer nudge a small model keeps reaching for the same few images. Each answer
# gets an angle (a mode and an image world), appended to the user message.

# mode -> (English, Hebrew) wording, for "Answer as ..." / "ענה ..."
MODES: dict[str, tuple[str, str]] = {
    "riddle": ("a riddle", "בחידה"),
    "prophecy": ("a prophecy", "בנבואה"),
    "playful dare": ("a playful dare", "באתגר שובב"),
    "tiny story": ("a tiny story", "בסיפור קטנטן"),
    "practical advice in disguise": ("practical advice in disguise",
                                     "בעצה מעשית בתחפושת"),
    "question back": ("a question thrown back at them", "בשאלה שמוחזרת אליהם"),
    "cheeky compliment": ("a cheeky compliment", "במחמאה חצופה"),
    "blunt truth with a wink": ("a blunt truth with a wink", "באמת ישירה עם קריצה"),
}

# image world -> (English, Hebrew)
IMAGERY: dict[str, tuple[str, str]] = {
    "fire": ("fire", "אש"),
    "water": ("water", "מים"),
    "birds": ("birds", "ציפורים"),
    "flamingos": ("flamingos", "פלמינגו"),
    "music and drums": ("music and drums", "מוזיקה ותופים"),
    "bicycles": ("bicycles", "אופניים"),
    "art cars": ("art cars", "מכוניות אמנות"),
    "food and tea": ("food and tea", "אוכל ותה"),
    "colours": ("colours", "צבעים"),
    "numbers": ("numbers", "מספרים"),
    "animals": ("animals", "חיות"),
    "weather and wind": ("weather and wind", "מזג אוויר ורוח"),
    "the Temple": ("the Temple", "המקדש"),
    "the Man": ("the Man", "האיש"),
    "shoes": ("shoes", "נעליים"),
    "maps": ("maps", "מפות"),
    "keys": ("keys", "מפתחות"),
    "clocks": ("clocks", "שעונים"),
    "seeds": ("seeds", "זרעים"),
    "salt": ("salt", "מלח"),
    "honey": ("honey", "דבש"),
    "moths": ("moths", "עשים"),
    "lanterns": ("lanterns", "פנסים"),
    "kites": ("kites", "עפיפונים"),
    "sand dunes": ("sand dunes", "דיונות חול"),
    "the moon's phases": ("the moon's phases", "מופעי הירח"),
    "bread": ("bread and baking", "לחם ואפייה"),
    "boats": ("boats and sailing", "סירות והפלגה"),
    "dancing": ("dancing", "ריקוד"),
    "cats": ("cats", "חתולים"),
    "gardens": ("gardens", "גינות"),
    "trains": ("trains", "רכבות"),
    "circus": ("the circus", "הקרקס"),
    "mountains": ("mountains", "הרים"),
}

MODE_WINDOW = 3  # a mode is not reused within this many answers
IMAGERY_WINDOW = 5  # nor an image world within this many

_ANGLE_NOTE = {
    "en": "(Answer as {mode}, using images of {imagery}. "
          "Start by answering their actual question.)",
    "he": "(ענה {mode}, בדימויים של {imagery}. פתח בתשובה לשאלה עצמה.)",
}


class Angle(NamedTuple):
    mode: str  # a key of MODES
    imagery: str  # a key of IMAGERY


def pick_angle(rng: random.Random, recent: Sequence[Angle]) -> Angle:
    """A random angle whose mode is none of the last MODE_WINDOW modes in `recent`
    (oldest first) and whose image world is none of the last IMAGERY_WINDOW."""
    used_modes = {a.mode for a in recent[-MODE_WINDOW:]}
    used_imagery = {a.imagery for a in recent[-IMAGERY_WINDOW:]}
    mode = rng.choice([m for m in MODES if m not in used_modes])
    imagery = rng.choice([i for i in IMAGERY if i not in used_imagery])
    return Angle(mode, imagery)


def angle_note(angle: Angle, lang: str) -> str:
    """The short hidden instruction appended to the user message, in the visitor's
    language (English for an unknown one)."""
    i = 1 if lang == "he" else 0
    return _ANGLE_NOTE["he" if i else "en"].format(mode=MODES[angle.mode][i],
                                                   imagery=IMAGERY[angle.imagery][i])


FALLBACK = {
    "en": "The mists are too thick right now. Ask me again in a moment.",
    "he": "הערפל סמיך מדי כרגע. שאלו אותי שוב עוד רגע.",
}

# Spoken while the LLM thinks; each one is at most ~2.5 s long.
FILLERS: dict[str, list[str]] = {
    "en": [
        "Mmmm… now that is an interesting thing to ask.",
        "Hmm. The mists are parting…",
        "Ahh. I have seen this question before, in a dream.",
        "Patience. The glass is turning.",
        "Mmm, the stars are whispering…",
        "Well, well… the threads are moving.",
        "Interesting. The answer is near.",
        "Hmmm… let me look beyond the glass.",
    ],
    "he": [
        "מממ… זו באמת שאלה מעניינת.",
        "הממ. הערפל מתפזר…",
        "אהה. את השאלה הזו כבר ראיתי, בחלום.",
        "סבלנות. הזכוכית מסתובבת.",
        "מממ, הכוכבים לוחשים…",
        "טוב, טוב… החוטים זזים.",
        "מעניין. התשובה קרובה.",
        "הממ… תנו לי להביט מעבר לזכוכית.",
    ],
}

# Spoken now and then while no one is around, to draw passers-by in; each one is
# at most ~3 s long. Hebrew addresses the crowd, in plural.
LURES: dict[str, list[str]] = {
    "en": [
        "Come closer, traveller… the glass has something to show you.",
        "I see you there. Ask, and the mirror will answer.",
        "Mirror, mirror… say the words, and I will wake.",
        "The dust remembers you. Come, ask me anything.",
    ],
    "he": [
        "התקרבו, נוסעים… לזכוכית יש משהו להראות לכם.",
        "אני רואה אתכם. שאלו, והמראה תענה.",
        "מראה, מראה… אמרו את המילים, ואתעורר.",
        "האבק זוכר אתכם. בואו, שאלו אותי הכול.",
    ],
}

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_SEPARATOR = re.compile(r"^---[ \t]*$", re.MULTILINE)

_WEEKDAYS = {
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "he": ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"],
}
_PARTS = {  # night, morning, afternoon, evening
    "en": ["night", "morning", "afternoon", "evening"],
    "he": ["לילה", "בוקר", "צהריים", "ערב"],
}
_TIME_LINE = {
    "en": "It is now {day} {part} at the burn.",
    "he": "עכשיו {day} {part} בברן.",
}


def load_world(path: str) -> str:
    """The oracle's background: the text after the file's `---` line, without
    <!-- comments -->. "" (with a warning) if the file is missing."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        log.warning("oracle world file %s not found; the oracle has no background", path)
        return ""
    parts = _SEPARATOR.split(text, maxsplit=1)
    body = parts[1] if len(parts) == 2 else text
    body = _COMMENT.sub("", body)
    return "\n".join(line.rstrip() for line in body.splitlines()).strip()


def _part_of_day(hour: int) -> int:
    if 5 <= hour < 12:
        return 1  # morning
    if 12 <= hour < 18:
        return 2  # afternoon
    if 18 <= hour < 22:
        return 3  # evening
    return 0  # night, 22:00-05:00


def system_prompt(lang: str, world: str, now: datetime) -> str:
    """The oracle prompt (with its few-shot examples, Hebrew only), then the world
    background, then the time line. The time
    line is last so the long prefix stays identical between requests (llama-server
    reuses its cached prompt prefix). An unknown language gets the English prompt."""
    if lang not in SYSTEM_PROMPT:
        lang = "en"
    time_line = _TIME_LINE[lang].format(day=_WEEKDAYS[lang][now.weekday()],
                                        part=_PARTS[lang][_part_of_day(now.hour)])
    examples = _examples(lang)
    return "\n\n".join(p for p in (SYSTEM_PROMPT[lang], examples, world, time_line) if p)


def _examples(lang: str) -> str:
    if lang != "he":
        return ""
    pairs = "\n".join(f"שאלה: {q}\nתשובה: {a}" for q, a in HEBREW_EXAMPLES)
    return f"{_EXAMPLES_HEADER_HE}\n{pairs}"
