"""Chunk-quality metrics.

- :func:`pk` and :func:`window_diff` compare predicted boundaries with gold
  ones (Beeferman et al. 1999; Pevzner & Hearst 2002). Lower is better.
- :func:`chunk_quality` needs no gold labels: it reports intra-chunk
  coherence, similarity between adjacent chunks, and size statistics.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .embeddings import Embedder, HashingEmbedder


def _to_mask(boundaries: Iterable[int], n: int) -> np.ndarray:
    """Boundary positions (unit k starts a new segment) -> gap mask of size n-1."""
    mask = np.zeros(max(n - 1, 0), dtype=bool)
    for b in boundaries:
        if 0 < b < n:
            mask[b - 1] = True
    return mask


def _default_k(gold: np.ndarray, n: int) -> int:
    segments = int(gold.sum()) + 1
    return max(2, round(n / segments / 2))


def pk(gold: Sequence[int], pred: Sequence[int], n: int, k: int | None = None) -> float:
    """Pk: probability that two units ``k`` apart are wrongly judged to be in
    the same/different segment. ``gold``/``pred`` are start indices of
    segments over ``n`` units."""
    g, p = _to_mask(gold, n), _to_mask(pred, n)
    k = k or _default_k(g, n)
    if n <= k:
        return 0.0
    errs = sum(g[i:i + k].any() != p[i:i + k].any() for i in range(n - k))
    return errs / (n - k)


def window_diff(gold: Sequence[int], pred: Sequence[int], n: int,
                k: int | None = None) -> float:
    """WindowDiff: like Pk but also penalises a wrong *number* of boundaries."""
    g, p = _to_mask(gold, n), _to_mask(pred, n)
    k = k or _default_k(g, n)
    if n <= k:
        return 0.0
    errs = sum(int(g[i:i + k].sum()) != int(p[i:i + k].sum()) for i in range(n - k))
    return errs / (n - k)


def chunk_quality(chunks: Sequence, embedder: Embedder | None = None) -> dict:
    """Label-free quality report for a list of chunks (``Chunk`` or ``str``)."""
    texts = [c if isinstance(c, str) else c.text for c in chunks]
    if not texts:
        return {"chunks": 0}
    embedder = embedder or HashingEmbedder()
    emb = embedder.embed(texts)
    adjacent = [float(emb[i] @ emb[i + 1]) for i in range(len(texts) - 1)]
    coh = [c.coherence for c in chunks
           if not isinstance(c, str) and c.coherence is not None]
    sizes = [len(t.split()) for t in texts]
    return {
        "chunks": len(texts),
        "mean_words": round(float(np.mean(sizes)), 1),
        "min_words": int(np.min(sizes)),
        "max_words": int(np.max(sizes)),
        "size_cv": round(float(np.std(sizes) / (np.mean(sizes) or 1)), 3),
        "mean_coherence": round(float(np.mean(coh)), 4) if coh else None,
        "mean_adjacent_similarity": round(float(np.mean(adjacent)), 4) if adjacent else None,
    }
