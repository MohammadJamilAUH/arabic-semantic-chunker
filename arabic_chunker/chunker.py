"""Structure-aware semantic chunking for Arabic.

Pipeline
--------
1. **Structure** — the document is split into sections by heading
   (:mod:`arabic_chunker.structure`). Sections are hard boundaries.
2. **Sentences** — each section is split into sentences with original offsets;
   sentences longer than ``max_tokens`` are split at clause boundaries.
3. **Boundary scores** — for every gap between two sentences, the cosine
   distance between the mean embedding of the ``window`` sentences on each
   side, plus Arabic discourse cues and a paragraph-break bonus.
4. **Segmentation** — with ``strategy="optimal"`` a dynamic program picks the
   set of cuts that maximises total boundary strength subject to the
   ``min_tokens``/``max_tokens`` budget. Unlike greedy thresholding it never
   produces a runt chunk because a strong boundary happened to appear early,
   and never cuts at a weak gap just because the buffer was full when a
   stronger gap was two sentences back.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Literal

import numpy as np

from .discourse import discourse_bias
from .embeddings import Embedder, HashingEmbedder
from .normalize import count_words
from .sentences import Span, split_long, split_sentences
from .structure import Section, parse_sections

ThresholdType = Literal["percentile", "std", "iqr"]


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    index: int = 0
    headings: tuple[str, ...] = ()
    token_count: int = 0
    sentence_count: int = 0
    boundary_score: float | None = None
    coherence: float | None = None
    parent_id: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def contextual_text(self) -> str:
        """Text prefixed with its heading path — what you should embed/index."""
        if not self.headings:
            return self.text
        return " › ".join(self.headings) + "\n\n" + self.text

    def to_dict(self) -> dict:
        d = asdict(self)
        d["headings"] = list(self.headings)
        d["contextual_text"] = self.contextual_text
        return d


@dataclass
class HierarchicalResult:
    parents: list[Chunk]
    children: list[Chunk]

    def children_of(self, parent: Chunk) -> list[Chunk]:
        return [c for c in self.children if c.parent_id == parent.index]


@dataclass
class _Unit:
    span: Span
    section: int
    tokens: int


class ArabicSemanticChunker:
    """Semantic chunker for Arabic (and mixed Arabic/English) documents.

    Parameters
    ----------
    embedder:
        Any object with ``embed(list[str]) -> np.ndarray``. Defaults to the
        dependency-free :class:`HashingEmbedder`.
    max_tokens, min_tokens:
        Chunk size budget, measured by ``length_function``. ``max_tokens`` is
        hard; ``min_tokens`` is enforced through ``min_size_penalty`` (large by
        default, so effectively hard) and only violated when a section is
        shorter than that or no valid partition exists.
    length_function:
        Defaults to word count. Pass a tokenizer-based counter to budget in
        model tokens, e.g. ``lambda s: len(enc.encode(s))``.
    window:
        Sentences on each side of a gap averaged into its context embedding.
    threshold_type, threshold_amount:
        How the "neutral" boundary strength is derived from the document's own
        score distribution: a percentile (default 70), ``mean + k·std`` or
        ``Q3 + k·IQR``. Gaps above it are rewarded as cut points, gaps below
        are penalised.
    strategy:
        ``"optimal"`` (dynamic programming) or ``"greedy"`` (classic
        threshold-and-accumulate).
    discourse_weight, paragraph_bonus:
        Weight of Arabic discourse markers and of paragraph breaks.
    overlap_sentences:
        Sentences repeated from the previous chunk of the same section.
    respect_structure:
        Treat headings as hard boundaries and attach heading paths.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        max_tokens: int = 256,
        min_tokens: int = 48,
        length_function: Callable[[str], int] = count_words,
        window: int = 3,
        threshold_type: ThresholdType = "percentile",
        threshold_amount: float | None = None,
        strategy: Literal["optimal", "greedy"] = "optimal",
        discourse_weight: float = 0.05,
        paragraph_bonus: float = 0.15,
        min_size_penalty: float = 100.0,
        overlap_sentences: int = 0,
        respect_structure: bool = True,
    ) -> None:
        if min_tokens >= max_tokens:
            raise ValueError("min_tokens must be smaller than max_tokens")
        if strategy not in ("optimal", "greedy"):
            raise ValueError(f"unknown strategy {strategy!r}")
        if threshold_type not in ("percentile", "std", "iqr"):
            raise ValueError(f"unknown threshold_type {threshold_type!r}")
        self.embedder = embedder or HashingEmbedder()
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.length_function = length_function
        self.window = max(1, window)
        self.threshold_type = threshold_type
        self.threshold_amount = threshold_amount
        self.strategy = strategy
        self.discourse_weight = discourse_weight
        self.paragraph_bonus = paragraph_bonus
        self.min_size_penalty = min_size_penalty
        self.overlap_sentences = max(0, overlap_sentences)
        self.respect_structure = respect_structure

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chunk(self, text: str) -> list[Chunk]:
        """Split ``text`` into semantically coherent chunks."""
        units, sections = self._units(text, self.max_tokens)
        if not units:
            return []
        emb = self.embedder.embed([u.span.text for u in units])
        scores = self._boundary_scores(units, emb)
        cuts = self._segment_all(units, scores, self.min_tokens, self.max_tokens)
        return self._build(text, units, sections, emb, scores, cuts,
                           self.overlap_sentences)

    def split_text(self, text: str) -> list[str]:
        """LangChain-style convenience: chunk texts only."""
        return [c.text for c in self.chunk(text)]

    def chunk_hierarchical(self, text: str, parent_max_tokens: int = 1024,
                           parent_min_tokens: int | None = None) -> HierarchicalResult:
        """Two-level chunking for small-to-big retrieval.

        Parents are section-bounded chunks up to ``parent_max_tokens``;
        children are regular chunks nested strictly inside one parent, with
        ``parent_id`` set to the parent's ``index``.
        """
        if parent_max_tokens <= self.max_tokens:
            raise ValueError("parent_max_tokens must exceed max_tokens")
        units, sections = self._units(text, self.max_tokens)
        if not units:
            return HierarchicalResult([], [])
        emb = self.embedder.embed([u.span.text for u in units])
        scores = self._boundary_scores(units, emb)
        pmin = parent_min_tokens if parent_min_tokens is not None else parent_max_tokens // 4
        pcuts = self._segment_all(units, scores, pmin, parent_max_tokens)
        parents = self._build(text, units, sections, emb, scores, pcuts, 0)

        # Children: segment each parent independently.
        ccuts: list[int] = []
        edges = [0] + pcuts + [len(units)]
        for a, b in zip(edges, edges[1:]):
            local = self._segment_range(units, scores, a, b, self.min_tokens, self.max_tokens)
            ccuts.extend([a] if a else [])
            ccuts.extend(local)
        children = self._build(text, units, sections, emb, scores, sorted(set(ccuts)),
                               self.overlap_sentences)
        for child in children:
            for p in parents:
                if p.start <= child.metadata["core_start"] and child.end <= p.end:
                    child.parent_id = p.index
                    break
        return HierarchicalResult(parents, children)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _units(self, text: str, max_tokens: int) -> tuple[list[_Unit], list[Section]]:
        sections = parse_sections(text) if self.respect_structure else \
            [Section(0, len(text))]
        units: list[_Unit] = []
        for si, sec in enumerate(sections):
            body = text[sec.start:sec.end]
            for sp in split_sentences(body):
                shifted = Span(sp.start + sec.start, sp.end + sec.start, sp.text,
                               sp.paragraph_start)
                for piece in split_long(shifted, max_tokens, self.length_function):
                    units.append(_Unit(piece, si, self.length_function(piece.text)))
        return units, sections

    def _boundary_scores(self, units: list[_Unit], emb: np.ndarray) -> np.ndarray:
        """Score for the gap *after* unit i; ``inf`` marks a section boundary."""
        n = len(units)
        scores = np.zeros(max(n - 1, 0), dtype=np.float64)
        w = self.window
        for i in range(n - 1):
            if units[i].section != units[i + 1].section:
                scores[i] = np.inf
                continue
            sec = units[i].section
            lo = i
            while lo > i - w + 1 and lo > 0 and units[lo - 1].section == sec:
                lo -= 1
            hi = i + 1
            while hi < i + w and hi + 1 < n and units[hi + 1].section == sec:
                hi += 1
            left = emb[lo:i + 1].mean(axis=0)
            right = emb[i + 1:hi + 1].mean(axis=0)
            denom = float(np.linalg.norm(left) * np.linalg.norm(right)) or 1.0
            dist = 1.0 - float(left @ right) / denom
            nxt = units[i + 1].span
            bias = discourse_bias(nxt.text, self.discourse_weight)
            if nxt.paragraph_start:
                bias += self.paragraph_bonus
            scores[i] = dist + bias
        return scores

    def _threshold(self, finite: np.ndarray) -> tuple[float, float]:
        if finite.size == 0:
            return 0.0, 1.0
        scale = float(finite.std()) or 1.0
        if self.threshold_type == "percentile":
            amt = 70.0 if self.threshold_amount is None else self.threshold_amount
            return float(np.percentile(finite, amt)), scale
        if self.threshold_type == "std":
            amt = 0.5 if self.threshold_amount is None else self.threshold_amount
            return float(finite.mean() + amt * finite.std()), scale
        q1, q3 = np.percentile(finite, [25, 75])
        amt = 0.0 if self.threshold_amount is None else self.threshold_amount
        return float(q3 + amt * (q3 - q1)), scale

    def _segment_all(self, units, scores, min_t, max_t) -> list[int]:
        """Cut indices (a cut at k means a chunk starts at unit k)."""
        self._theta, self._scale = self._threshold(scores[np.isfinite(scores)])
        cuts: list[int] = []
        start = 0
        for i in range(len(units) - 1):
            if not np.isfinite(scores[i]):
                cuts.extend(self._segment_range(units, scores, start, i + 1, min_t, max_t))
                cuts.append(i + 1)
                start = i + 1
        cuts.extend(self._segment_range(units, scores, start, len(units), min_t, max_t))
        return sorted(set(cuts))

    def _segment_range(self, units, scores, a, b, min_t, max_t) -> list[int]:
        """Internal cut points for units[a:b] (a single section)."""
        if b - a <= 1:
            return []
        tok = np.array([u.tokens for u in units[a:b]], dtype=np.int64)
        prefix = np.concatenate([[0], np.cumsum(tok)])
        if prefix[-1] <= max_t:
            return []
        if self.strategy == "greedy":
            return [a + k for k in self._greedy(scores[a:b - 1], tok, min_t, max_t)]
        return [a + k for k in self._optimal(scores[a:b - 1], prefix, min_t, max_t)]

    def _reward(self, s: float) -> float:
        return (s - self._theta) / self._scale

    def _optimal(self, gaps, prefix, min_t, max_t) -> list[int]:
        n = len(prefix) - 1
        best = np.full(n + 1, -np.inf)
        back = np.zeros(n + 1, dtype=np.int64)
        best[0] = 0.0
        for j in range(1, n + 1):
            for i in range(j - 1, -1, -1):
                size = prefix[j] - prefix[i]
                if size > max_t and i < j - 1:
                    break
                if best[i] == -np.inf:
                    continue
                val = best[i]
                if i > 0:
                    val += self._reward(gaps[i - 1])
                if size < min_t:
                    val -= self.min_size_penalty * (min_t - size) / min_t
                if val > best[j]:
                    best[j], back[j] = val, i
        cuts, j = [], n
        while j > 0:
            i = int(back[j])
            if i > 0:
                cuts.append(i)
            j = i
        return sorted(cuts)

    def _greedy(self, gaps, tok, min_t, max_t) -> list[int]:
        cuts, size = [], int(tok[0])
        for k in range(1, len(tok)):
            nxt = int(tok[k])
            if size + nxt > max_t or (size >= min_t and gaps[k - 1] > self._theta):
                cuts.append(k)
                size = nxt
            else:
                size += nxt
        return cuts

    def _build(self, text, units, sections, emb, scores, cuts, overlap) -> list[Chunk]:
        edges = [0] + list(cuts) + [len(units)]
        chunks: list[Chunk] = []
        for a, b in zip(edges, edges[1:]):
            if a >= b:
                continue
            first = a
            if overlap and a > 0 and units[a - 1].section == units[a].section:
                first = max(a - overlap, 0)
                while units[first].section != units[a].section:
                    first += 1
            start, end = units[first].span.start, units[b - 1].span.end
            body = text[start:end]
            vecs = emb[a:b]
            centroid = vecs.mean(axis=0)
            cn = float(np.linalg.norm(centroid)) or 1.0
            coherence = float(np.mean(vecs @ centroid) / cn)
            bscore = None
            if a > 0 and np.isfinite(scores[a - 1]):
                bscore = round(float(scores[a - 1]), 4)
            sec = sections[units[a].section]
            chunks.append(Chunk(
                text=body, start=start, end=end, index=len(chunks),
                headings=sec.headings,
                token_count=self.length_function(body),
                sentence_count=b - first,
                boundary_score=bscore,
                coherence=round(coherence, 4),
                metadata={"core_start": units[a].span.start,
                          "overlap_sentences": a - first},
            ))
        return chunks
