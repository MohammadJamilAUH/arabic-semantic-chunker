"""Embedding backends.

An embedder is anything with ``embed(texts: list[str]) -> np.ndarray`` that
returns one L2-normalised row per text.

- :class:`HashingEmbedder` needs only NumPy. It combines light-stemmed content
  words with character n-grams of those stems (robust to Arabic morphology and
  broken PDF spacing), weighted by in-document IDF so that words shared by
  every sentence do not mask topic shifts. It is lexical, not semantic, but
  deterministic, fast and a solid baseline for boundary detection.
- :class:`SentenceTransformerEmbedder` wraps any sentence-transformers model
  (multilingual E5, BGE-M3, LaBSE, Arabic-specific models ...).
- :class:`CallableEmbedder` adapts any function, e.g. a hosted embedding API.
"""

from __future__ import annotations

import zlib
from collections import Counter
from typing import Callable, Protocol, Sequence

import numpy as np

from .normalize import normalize_arabic, stems


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder:
    """Dependency-free lexical embedder tuned for Arabic."""

    def __init__(self, dim: int = 2048, ngram_range: tuple[int, int] = (2, 4),
                 word_weight: float = 2.0, use_idf: bool = True) -> None:
        self.dim = dim
        self.ngram_range = ngram_range
        self.word_weight = word_weight
        self.use_idf = use_idf

    def _features(self, text: str) -> Counter:
        # Content words only: function words and clitics carry no topic signal
        # but would dominate raw character n-grams in Arabic.
        feats: Counter = Counter()
        lo, hi = self.ngram_range
        for w in stems(text):
            feats["w:" + w] += self.word_weight
            padded = f" {w} "
            for n in range(lo, hi + 1):
                for i in range(len(padded) - n + 1):
                    feats["c:" + padded[i:i + n]] += 1.0
        if not feats:  # only stopwords: fall back to the raw words
            for w in normalize_arabic(text).split():
                feats["w:" + w] += 1.0
        return feats

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = [self._features(t) for t in texts]
        idf: dict[str, float] = {}
        if self.use_idf and len(rows) > 1:
            df = Counter(f for r in rows for f in r)
            n = len(rows)
            idf = {f: float(np.log((1 + n) / (1 + c))) + 1.0 for f, c in df.items()}
        out = np.zeros((len(rows), self.dim), dtype=np.float32)
        for i, feats in enumerate(rows):
            for f, tf in feats.items():
                h = zlib.crc32(f.encode("utf-8"))
                sign = 1.0 if (h >> 31) & 1 else -1.0
                out[i, h % self.dim] += sign * (1.0 + np.log(tf)) * idf.get(f, 1.0)
        return _l2(out)


class SentenceTransformerEmbedder:
    """Wraps a ``sentence-transformers`` model (install the ``[transformers]`` extra).

    E5-family models expect a ``"passage: "`` prefix; it is added
    automatically when the model name contains ``e5``.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base",
                 prefix: str | None = None, batch_size: int = 32,
                 device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "pip install 'arabic-semantic-chunker[transformers]'") from e
        self.model = SentenceTransformer(model_name, device=device)
        self.prefix = prefix if prefix is not None else (
            "passage: " if "e5" in model_name.lower() else "")
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode([self.prefix + t for t in texts],
                                 batch_size=self.batch_size,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class CallableEmbedder:
    """Adapts ``fn(list[str]) -> array-like`` into an embedder."""

    def __init__(self, fn: Callable[[list[str]], Sequence[Sequence[float]]],
                 batch_size: int = 64) -> None:
        self.fn = fn
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        parts = [np.asarray(self.fn(list(texts[i:i + self.batch_size])), dtype=np.float32)
                 for i in range(0, len(texts), self.batch_size)]
        if not parts:
            return np.zeros((0, 1), dtype=np.float32)
        return _l2(np.vstack(parts))
