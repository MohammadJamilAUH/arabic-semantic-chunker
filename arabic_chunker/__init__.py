"""Advanced structure-aware semantic chunking for Arabic text."""

from .chunker import ArabicSemanticChunker, Chunk, HierarchicalResult
from .discourse import discourse_bias
from .embeddings import (CallableEmbedder, Embedder, HashingEmbedder,
                         SentenceTransformerEmbedder)
from .metrics import chunk_quality, pk, window_diff
from .normalize import count_words, light_stem, normalize_arabic, strip_diacritics
from .sentences import Span, split_long, split_sentences
from .structure import Section, detect_heading, parse_sections

__version__ = "0.1.0"

__all__ = [
    "ArabicSemanticChunker", "Chunk", "HierarchicalResult",
    "Embedder", "HashingEmbedder", "SentenceTransformerEmbedder", "CallableEmbedder",
    "split_sentences", "split_long", "Span",
    "parse_sections", "detect_heading", "Section",
    "normalize_arabic", "strip_diacritics", "light_stem", "count_words",
    "discourse_bias", "chunk_quality", "pk", "window_diff",
]
