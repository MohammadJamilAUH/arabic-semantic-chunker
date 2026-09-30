"""Arabic text normalization, tokenization and light stemming.

Everything here is used for *matching and embedding only*. Chunk text returned
to the caller is always the original text, sliced by character offsets, so
normalization can be as aggressive as retrieval needs without corrupting output.
"""

from __future__ import annotations

import re
import unicodedata

# Harakat, Quranic annotation marks, superscript alef, and tatweel.
_DIACRITICS_RE = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭـ]"
)
# Invisible direction/format controls that PDF extraction leaves behind.
_CONTROLS_RE = re.compile("[​-‏‪-‮⁦-⁩﻿]")

_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
_LETTER_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي",
    "ؤ": "و",
    "ة": "ه",
    "ک": "ك", "ی": "ي", "ڤ": "ف", "گ": "ك",
})
_SCRIPT_BOUNDARY_RE = re.compile(
    r"(?<=[ء-ي])(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])(?=[ء-ي])"
)

ARABIC_LETTER = "ء-ي"
WORD_RE = re.compile(rf"[{ARABIC_LETTER}٠-٩۰-۹A-Za-z0-9_]+")
_HAS_ARABIC_RE = re.compile(f"[{ARABIC_LETTER}]")


def strip_diacritics(text: str) -> str:
    """Remove harakat, Quranic marks and tatweel."""
    return _DIACRITICS_RE.sub("", text)


def normalize_arabic(
    text: str,
    *,
    fold_letters: bool = True,
    latin_digits: bool = True,
    lowercase: bool = True,
) -> str:
    """Normalize Arabic text for comparison.

    - NFKC, which also resolves presentation forms (U+FB50–U+FEFF) that
      PDF extractors emit instead of base letters;
    - drops diacritics, tatweel and bidi control characters;
    - folds alef/hamza variants, alef maqsura, taa marbuta and Persian letters;
    - converts Arabic-Indic and Persian digits to ASCII;
    - separates glued Arabic/Latin runs ("قانونGDPR" -> "قانون GDPR").
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _CONTROLS_RE.sub("", t)
    t = strip_diacritics(t)
    if fold_letters:
        t = t.translate(_LETTER_FOLD)
    if latin_digits:
        t = t.translate(_DIGITS)
    t = _SCRIPT_BOUNDARY_RE.sub(" ", t)
    if lowercase:
        t = t.lower()
    return re.sub(r"\s+", " ", t).strip()


def is_arabic(text: str, threshold: float = 0.3) -> bool:
    """True if at least ``threshold`` of the letters are Arabic."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ar = sum(1 for c in letters if _HAS_ARABIC_RE.match(c))
    return ar / len(letters) >= threshold


def words(text: str) -> list[str]:
    """Surface word tokens (no normalization)."""
    return WORD_RE.findall(text)


def count_words(text: str) -> int:
    """Default length function: number of word tokens."""
    return len(WORD_RE.findall(text))


# --------------------------------------------------------------------------- #
# Light stemming
# --------------------------------------------------------------------------- #
# Longest affixes first. Stemming is deliberately conservative: a stem must keep
# at least three letters, since most Arabic roots are triliteral.
_PREFIXES = (
    "وبال", "وكال", "فبال", "وال", "فال", "بال", "كال", "لل",
    "ال", "وس", "فس", "و", "ف", "ب", "ل", "ك", "س",
)
_SUFFIXES = (
    "تهما", "كما", "هما", "تما", "ونها", "ات", "ان", "ون", "ين", "ها", "هم",
    "هن", "كم", "كن", "نا", "يه", "ته", "تي", "ه", "ي", "ت",
)
_MIN_STEM = 3


def light_stem(word: str) -> str:
    """Strip one clitic prefix and one suffix from a *normalized* Arabic word.

    Latin/numeric tokens are returned unchanged.
    """
    if not _HAS_ARABIC_RE.search(word):
        return word
    w = word
    for p in _PREFIXES:
        if w.startswith(p) and len(w) - len(p) >= _MIN_STEM:
            w = w[len(p):]
            break
    for s in _SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= _MIN_STEM:
            w = w[: -len(s)]
            break
    return w


_TANWEEN_ALEF_RE = re.compile("(?:\u0627\u064b|\u064b\u0627)(?![\u0621-\u064a])")


def stems(text: str) -> list[str]:
    """Normalized, stemmed word tokens with stopwords removed.

    Accusative tanween alef is dropped first so that "طقساً" and "الطقس"
    share the stem "طقس".
    """
    text = _TANWEEN_ALEF_RE.sub("", text)
    return [
        light_stem(w)
        for w in WORD_RE.findall(normalize_arabic(text))
        if w not in STOPWORDS
    ]


STOPWORDS = frozenset(normalize_arabic(w) for w in """
في من الى إلى على عن مع هذا هذه ذلك تلك هؤلاء أولئك التي الذي الذين اللذان
اللتان اللواتي اللاتي ما ماذا لماذا كيف أين متى هل لا لم لن ليس ليست إن أن
كان كانت يكون تكون قد لقد ثم أو أم بل لكن حتى إذا اذا إذ كل بعض غير أي عند
عندما بين حيث كما هو هي هم هن أنا نحن أنت أنتم له لها لهم به بها بهم فيه فيها
منه منها عليه عليها و ف ب ل ك يا وقد وهو وهي وفي ومن وعلى وإن وأن ولا ولم
the a an of to in on for and or is are was were be by with as at from this that
""".split())
