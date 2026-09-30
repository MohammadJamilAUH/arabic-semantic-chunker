"""Arabic discourse cues used to adjust semantic boundary scores.

Embedding distance alone misses signals that Arabic writers make explicit:
"أما ..." or "من ناحية أخرى" announce a new topic, while a sentence opening
with "لذلك" or an anaphoric "هذا/هذه" leans on the previous one and should
rarely start a chunk.

The returned value is added to the boundary score before the segmenter runs:
positive favours a split *before* the sentence, negative discourages it.
"""

from __future__ import annotations

import re

from .normalize import normalize_arabic

_SHIFT = [
    "اما", "من ناحيه اخرى", "من جهه اخرى", "في المقابل", "على صعيد اخر",
    "على الصعيد", "بالانتقال الى", "ننتقل الى", "فيما يتعلق ب", "فيما يخص",
    "بالنسبه ل", "اولا", "ثانيا", "ثالثا", "رابعا", "خامسا", "اخيرا", "ختاما",
    "في الختام", "خلاصه القول", "في البدايه", "بدايه", "المحور", "الموضوع التالي",
    "however", "on the other hand", "in conclusion", "finally", "firstly",
    "secondly", "next",
]
_CONTINUE = [
    "لذلك", "لذا", "ولذلك", "ولذا", "وبالتالي", "بالتالي", "وعليه", "عليه فان",
    "كما ان", "كما", "وكذلك", "كذلك", "ايضا", "علاوه على ذلك", "بالاضافه الى ذلك",
    "اضافه الى ذلك", "فضلا عن ذلك", "مما", "ما يعني", "اي ان", "بمعنى اخر",
    "وهذا", "وهذه", "هذا", "هذه", "ذلك", "تلك", "وذلك", "حيث", "اذ", "لان",
    "لكن", "ولكن", "الا ان", "غير ان", "ومع ذلك", "رغم ذلك", "مثلا", "على سبيل المثال",
    "therefore", "thus", "also", "moreover", "furthermore", "this", "these",
    "because", "for example", "but",
]


def _compile(phrases: list[str]) -> re.Pattern:
    alts = sorted({normalize_arabic(p) for p in phrases}, key=len, reverse=True)
    return re.compile(r"^(?:" + "|".join(re.escape(a) for a in alts) + r")(?:\s|$|،|,)")


_SHIFT_RE = _compile(_SHIFT)
_CONTINUE_RE = _compile(_CONTINUE)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-–•●▪*]|\(?\d{1,3}[.)\-]|\(?[ء-ي][)\-])")


def discourse_bias(sentence: str, weight: float = 0.15) -> float:
    """Boundary bias for placing a chunk break immediately before ``sentence``."""
    if _LIST_ITEM_RE.match(sentence):
        # A list item continues the list that precedes it.
        return -0.5 * weight
    head = normalize_arabic(sentence)[:40]
    if _SHIFT_RE.match(head):
        return weight
    if _CONTINUE_RE.match(head):
        return -weight
    return 0.0
