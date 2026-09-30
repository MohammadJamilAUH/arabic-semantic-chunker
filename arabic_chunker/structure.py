"""Document structure detection: headings and a section tree.

Recognises Markdown headings, Arabic legal/regulatory structure
(الكتاب / الباب / الفصل / الفرع / المادة, including the reversed "(31) المادة"
order that PDF extraction produces), decimal-numbered headings ("2.3 النطاق"),
bold-only lines, and common standalone section titles (مقدمة، الخاتمة ...).

Sections are hard boundaries for the chunker: no chunk ever spans two sections,
and every chunk carries the heading path of the section it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ORDINAL_WORDS = (
    r"الأول|الاول|الأولى|الاولى|الثاني|الثانية|الثالث|الثالثة|الرابع|الرابعة|"
    r"الخامس|الخامسة|السادس|السادسة|السابع|السابعة|الثامن|الثامنة|التاسع|"
    r"التاسعة|العاشر|العاشرة|الحادي\s+عشر|الحادية\s+عشرة|الثاني\s+عشر|"
    r"الثانية\s+عشرة|[ء-ي]+\s+(?:عشر|عشرة|والعشرون|والعشرين|والثلاثون)"
)
_NUM = rf"(?:\(?\s*[0-9٠-٩]+\s*\)?|{_ORDINAL_WORDS})"

# (keyword pattern, level)
_LEGAL_LEVELS = (
    (r"الكتاب", 1),
    (r"القسم", 1),
    (r"الباب", 1),
    (r"الفصل", 2),
    (r"الفرع", 3),
    (r"المبحث", 3),
    (r"المطلب", 4),
    (r"(?:ال)?مادة|المادّة", 4),
)
_LEGAL_RES = tuple(
    (re.compile(rf"^\s*(?:{kw})\s*{_NUM}(?:\s*[:：\-–—.]\s*(?P<title>.*))?\s*$"), lvl)
    for kw, lvl in _LEGAL_LEVELS
) + (
    # Reversed order from visual-order PDF extraction: "(31) المادة"
    (re.compile(r"^\s*\(\s*[0-9٠-٩]+\s*\)\s*(?:ال)?مادة\s*$"), 4),
)
_MD_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_BOLD_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*:?\s*$")
_DECIMAL_RE = re.compile(r"^\s*((?:[0-9٠-٩]+\.)+[0-9٠-٩]*)\s+(\S.*)$")
_STANDALONE = re.compile(
    r"^\s*(?:مقدمة|المقدمة|تمهيد|التمهيد|الخاتمة|خاتمة|الخلاصة|الملخص|"
    r"الملخص\s+التنفيذي|ملخص|المراجع|المصادر|التوصيات|النتائج|الملاحق|"
    r"التعريفات|تعريفات|نطاق\s+التطبيق|الأحكام\s+العامة|الأحكام\s+الختامية|"
    r"abstract|introduction|conclusions?|references|summary)\s*:?\s*$",
    re.IGNORECASE,
)
_MAX_HEADING_CHARS = 120
_MAX_NUMBERED_WORDS = 10


@dataclass
class Section:
    start: int                     # body start (after the heading line)
    end: int                       # body end
    headings: tuple[str, ...] = ()
    level: int = 0
    heading_start: int | None = None


def detect_heading(line: str) -> tuple[str, int] | None:
    """Return ``(heading_text, level)`` if ``line`` is a heading, else ``None``."""
    text = line.strip()
    if not text:
        return None
    md = _MD_RE.match(text)
    if md:
        return md.group(2).strip("* "), len(md.group(1))
    if len(text) > _MAX_HEADING_CHARS:
        return None
    for pattern, level in _LEGAL_RES:
        if pattern.match(text):
            return text.rstrip(":：-–— "), level
    bold = _BOLD_RE.match(text)
    if bold:
        return bold.group(1).strip(), 2
    if _STANDALONE.match(text):
        return text.rstrip(": "), 2
    dec = _DECIMAL_RE.match(text)
    if dec and len(dec.group(2).split()) <= _MAX_NUMBERED_WORDS \
            and not re.search(r"[.!?؟،,؛]\s*$", text):
        depth = len([p for p in dec.group(1).split(".") if p])
        return text, min(depth, 6)
    return None


def parse_sections(text: str) -> list[Section]:
    """Split ``text`` into a flat, ordered list of sections.

    Each section's ``headings`` is the full path from the outermost heading
    down to its own. Text before the first heading forms a section with an
    empty path.
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    body_start = 0
    heading_start: int | None = None
    level = 0
    pos = 0
    for line in text.splitlines(keepends=True):
        found = detect_heading(line)
        if found:
            sections.append(Section(body_start, pos, tuple(h for _, h in stack),
                                    level, heading_start))
            head, level = found
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, head))
            heading_start = pos
            body_start = pos + len(line)
        pos += len(line)
    sections.append(Section(body_start, len(text), tuple(h for _, h in stack),
                            level, heading_start))
    return [s for s in sections if text[s.start:s.end].strip()]

