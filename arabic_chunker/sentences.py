"""Offset-preserving Arabic sentence and clause segmentation.

Arabic punctuation is used inconsistently in real documents: the comma often
does the work of a full stop, the question mark may be Latin or Arabic, list
items rarely end in a period, and PDF extraction hard-wraps lines mid-sentence.
The splitter therefore works in three passes:

1. paragraphs, on blank lines;
2. lines, where a single newline counts as a break only when the next line
   starts a list item or the previous line already ended a sentence;
3. sentence terminators, with guards for abbreviations, initials, decimals and
   numbered-list markers.

Every span keeps its character offsets into the original text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import ARABIC_LETTER, count_words


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    paragraph_start: bool = False


_PARAGRAPH_RE = re.compile(r"\n[ \t ]*\n\s*")
_LIST_MARKER_RE = re.compile(
    r"^\s*(?:[-–—•●▪■◦*]"                                   # bullets
    r"|\(?\d{1,3}[.)\-–]"                                  # 1.  1)  (1)  1-
    r"|\(?[ء-ي][)\-–]"                           # أ)  (ب)  ج-
    r"|(?:أولاً|ثانياً|ثالثاً|رابعاً|خامساً|سادساً|سابعاً|ثامناً|تاسعاً|عاشراً)"
    r")"
)
_LINE_END_RE = re.compile(r"[.!?؟:…»”\"')\]]\s*$")

_TERMINATOR_RE = re.compile(
    rf"(?:\.{{3}}|…|[.!?؟]+)[\"'»”’)\]]*(?=\s|$|[{ARABIC_LETTER}])"
)
_PREV_TOKEN_RE = re.compile(r"(\S+?)\s*$")

_ABBREVIATIONS = frozenset({
    # Arabic
    "هـ", "ه", "م", "ق.م", "ص", "ج", "د", "أ", "أد", "أ.د", "ت", "ط", "س", "ش",
    "إلخ", "الخ", "ا", "ب", "ع", "ر", "ل",
    # Latin
    "dr", "mr", "mrs", "ms", "prof", "st", "no", "vs", "etc", "e.g", "i.e",
    "inc", "ltd", "co", "jr", "sr", "fig", "vol", "pp", "p", "art", "sec",
})
# Abbreviations that also end sentences often enough that we only protect them
# when the next word is lowercase Latin or another abbreviation-looking token.
_SENTENCE_FINAL_ABBR = frozenset({"إلخ", "الخ", "etc"})
_ERA_MARKERS = frozenset({"هـ", "ه", "م", "ق.م"})


def _is_abbreviation(text: str, dot_pos: int) -> bool:
    """Whether the '.' at ``dot_pos`` belongs to an abbreviation or list marker."""
    # Only look at a bounded window: scanning from the start of the text on
    # every period makes splitting quadratic in document length.
    head = text[max(0, dot_pos - 80):dot_pos]
    m = _PREV_TOKEN_RE.search(head)
    if not m:
        return True  # a lone dot at the start of a segment is not a sentence
    token = m.group(1).lstrip("(«\"'[").rstrip("ـ")
    low = token.lower()
    if low in _SENTENCE_FINAL_ABBR:
        nxt = text[dot_pos + 1:dot_pos + 16].lstrip()[:1]
        return bool(nxt) and nxt.islower()
    if low in _ERA_MARKERS:
        # "1445 هـ." / "2024 م." — after a year the dot doubles as a full stop.
        before = head[: m.start(1)].split()
        return not (before and before[-1].isdigit())
    if low in _ABBREVIATIONS:
        return True
    if len(token) == 1 and token.isalpha():  # initials: "M. Ali", "د. محمد"
        return True
    if re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]?", token):  # U.S.A
        return True
    # "1." / "12." at the very start of a line is a list marker, not an end.
    if token.isdigit():
        line_start = text.rfind("\n", 0, dot_pos) + 1
        if dot_pos - line_start <= len(token) + 8 and \
                text[line_start:dot_pos].strip() == token:
            return True
    return False


def _split_on_terminators(text: str, base: int, first_is_para: bool) -> list[Span]:
    spans: list[Span] = []
    cursor = 0
    for m in _TERMINATOR_RE.finditer(text):
        mark = m.group(0)
        if mark.startswith(".") and not mark.startswith("...") and \
                _is_abbreviation(text, m.start()):
            continue
        _append(spans, text, cursor, m.end(), base, first_is_para and not spans)
        cursor = m.end()
    _append(spans, text, cursor, len(text), base, first_is_para and not spans)
    return spans


def _append(out: list[Span], text: str, s: int, e: int, base: int, para: bool) -> None:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    if s < e:
        out.append(Span(base + s, base + e, text[s:e], para))


def _line_groups(para: str) -> list[tuple[int, int]]:
    """Merge hard-wrapped lines; break before list items and after finished lines."""
    groups: list[tuple[int, int]] = []
    pos = 0
    start = 0
    lines = para.split("\n")
    for i, line in enumerate(lines):
        end = pos + len(line)
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            if _LIST_MARKER_RE.match(nxt) or _LINE_END_RE.search(line) or not line.strip():
                groups.append((start, end))
                start = end + 1
        pos = end + 1
    groups.append((start, len(para)))
    return groups


def split_sentences(text: str) -> list[Span]:
    """Split ``text`` into sentence spans with original character offsets."""
    spans: list[Span] = []
    bounds = list(_PARAGRAPH_RE.finditer(text))
    edges = [(0, bounds[0].start() if bounds else len(text))]
    for i, b in enumerate(bounds):
        edges.append((b.end(), bounds[i + 1].start() if i + 1 < len(bounds) else len(text)))
    for para_start, para_end in edges:
        para = text[para_start:para_end]
        first = True
        for gs, ge in _line_groups(para):
            found = _split_on_terminators(para[gs:ge], para_start + gs, first)
            spans.extend(found)
            first = first and not found
    return spans


# --------------------------------------------------------------------------- #
# Clause splitting for sentences that exceed the chunk budget
# --------------------------------------------------------------------------- #
_CLAUSE_LEVELS = (
    re.compile(r"[؛;]"),
    re.compile(r"[،,:]"),
    # Discourse connectives that introduce a new clause. Split *before* them.
    re.compile(
        r"\s(?=(?:و|ف)?(?:لكن|بينما|إلا\s+أن|غير\s+أن|حيث|بحيث|كما\s+أن|"
        r"إضافة\s+إلى|بالإضافة\s+إلى|علاوة\s+على|ثم|أما|لذلك|لذا|وبالتالي)\s)"
    ),
)


def split_long(span: Span, max_len: int, length_fn=count_words) -> list[Span]:
    """Recursively split a span longer than ``max_len`` at clause boundaries,
    falling back to fixed word windows. Offsets are preserved."""
    if length_fn(span.text) <= max_len:
        return [span]
    for pattern in _CLAUSE_LEVELS:
        pieces = _split_keep(span, pattern)
        if len(pieces) > 1:
            merged = _greedy_merge(span, pieces, max_len, length_fn)
            if len(merged) > 1:
                out: list[Span] = []
                for piece in merged:
                    out.extend(split_long(piece, max_len, length_fn))
                return out
    return _word_windows(span, max_len, length_fn)


def _split_keep(span: Span, pattern: re.Pattern) -> list[tuple[int, int]]:
    cuts = [m.end() for m in pattern.finditer(span.text)]
    edges = [0] + [c for c in cuts if 0 < c < len(span.text)] + [len(span.text)]
    return [(a, b) for a, b in zip(edges, edges[1:]) if span.text[a:b].strip()]


def _greedy_merge(span: Span, pieces, max_len, length_fn) -> list[Span]:
    out: list[Span] = []
    cur_s, cur_e = pieces[0]
    for s, e in pieces[1:]:
        if length_fn(span.text[cur_s:e]) <= max_len:
            cur_e = e
        else:
            out.append(_sub(span, cur_s, cur_e, not out))
            cur_s, cur_e = s, e
    out.append(_sub(span, cur_s, cur_e, not out))
    return out


def _sub(span: Span, s: int, e: int, keep_para: bool) -> Span:
    raw = span.text[s:e]
    lead = len(raw) - len(raw.lstrip())
    body = raw.strip()
    start = span.start + s + lead
    return Span(start, start + len(body), body, span.paragraph_start and keep_para)


def _word_windows(span: Span, max_len: int, length_fn) -> list[Span]:
    ends = [m.end() for m in re.finditer(r"\S+", span.text)]
    starts = [m.start() for m in re.finditer(r"\S+", span.text)]
    out: list[Span] = []
    i = 0
    while i < len(starts):
        j = i + 1
        while j < len(starts) and length_fn(span.text[starts[i]:ends[j]]) <= max_len:
            j += 1
        out.append(_sub(span, starts[i], ends[j - 1], not out))
        i = j
    return out
