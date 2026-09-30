"""Topic-segmentation benchmark for Arabic chunkers.

Builds Choi-style synthetic documents: each is a concatenation of 4–7
segments, and each segment is the first 3–7 sentences of one topic in
``data/topics.json`` (16 hand-written MSA topics), as in Choi (2000).
``--random-start`` takes the run from a random position instead, so a segment
may open mid-topic with "كما…" or "وهذا…"; that is a harder and less
realistic setting, and it works against the discourse cues by construction. Adjacent segments never
share a topic. A good chunker should place its cuts exactly at the segment
seams, which gives exact gold boundaries to score against with Pk and
WindowDiff (lower is better).

Segments are joined with a single space by default, so there is no paragraph
or heading cue: only the content says where one topic ends. Pass
``--paragraphs`` to join segments with blank lines instead.

    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --model intfloat/multilingual-e5-base
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arabic_chunker import (ArabicSemanticChunker, HashingEmbedder, count_words,  # noqa: E402
                            pk, split_sentences, window_diff)

DATA = Path(__file__).parent / "data" / "topics.json"


def build_docs(n_docs: int, seed: int, paragraphs: bool,
               random_start: bool = False) -> list[tuple[str, list[int]]]:
    topics = json.loads(DATA.read_text(encoding="utf-8"))
    names = sorted(topics)
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        segs, prev = [], None
        for _ in range(rng.randint(4, 7)):
            name = rng.choice([n for n in names if n != prev])
            sents = topics[name]
            k = rng.randint(3, 7)
            start = rng.randint(0, len(sents) - k) if random_start else 0
            segs.append(sents[start:start + k])
            prev = name
        gold, n = [], 0
        for seg in segs[:-1]:
            n += len(seg)
            gold.append(n)
        sep = "\n\n" if paragraphs else " "
        docs.append((sep.join(" ".join(s) for s in segs), gold))
    return docs


class CachingEmbedder:
    """Memoises sentence embeddings so every method sees identical vectors."""

    def __init__(self, inner):
        self.inner = inner
        self.cache: dict[str, np.ndarray] = {}

    def embed(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            for t, v in zip(missing, self.inner.embed(missing)):
                self.cache[t] = v
        return np.vstack([self.cache[t] for t in texts])


def sentence_starts(text: str) -> list[int]:
    return [s.start for s in split_sentences(text)]


def to_units(starts: list[int], positions: list[int]) -> list[int]:
    return sorted({bisect.bisect_left(starts, p) for p in positions if p > 0})


# ---------------------------------------------------------------- methods
def fixed_size(text: str, budget: int) -> list[int]:
    """Sentence-aware fixed-size baseline: pack sentences up to ``budget`` words."""
    cuts, size = [], 0
    for s in split_sentences(text):
        n = count_words(s.text)
        if size and size + n > budget:
            cuts.append(s.start)
            size = 0
        size += n
    return cuts


def semantic(chunker: ArabicSemanticChunker):
    def run(text: str) -> list[int]:
        return [c.metadata["core_start"] for c in chunker.chunk(text)]
    return run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--model", default=None, help="sentence-transformers model name")
    ap.add_argument("--paragraphs", action="store_true")
    ap.add_argument("--random-start", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=110)
    ap.add_argument("--min-tokens", type=int, default=20)
    ap.add_argument("--json", action="store_true", help="print raw results as JSON")
    args = ap.parse_args(argv)

    if args.model:
        from arabic_chunker import SentenceTransformerEmbedder
        base = SentenceTransformerEmbedder(args.model)
        emb_name = args.model
    else:
        base = HashingEmbedder()
        emb_name = "HashingEmbedder"
    emb = CachingEmbedder(base)

    docs = build_docs(args.docs, args.seed, args.paragraphs, args.random_start)
    seg_words = statistics.mean(
        count_words(t) / (len(g) + 1) for t, g in docs)

    def make(**kw):
        return semantic(ArabicSemanticChunker(
            emb, max_tokens=args.max_tokens, min_tokens=args.min_tokens,
            respect_structure=False, **kw))

    methods = {
        "fixed-size (sentence-aware)": lambda t: fixed_size(t, round(seg_words)),
        "semantic greedy": make(strategy="greedy"),
        "semantic optimal, no discourse cues": make(discourse_weight=0.0),
        "semantic optimal (default)": make(),
    }

    results = {}
    for name, fn in methods.items():
        pks, wds, counts, t0 = [], [], [], time.time()
        for text, gold in docs:
            starts = sentence_starts(text)
            pred = to_units(starts, fn(text))
            pks.append(pk(gold, pred, len(starts)))
            wds.append(window_diff(gold, pred, len(starts)))
            counts.append(len(pred) + 1)
        results[name] = {
            "pk": statistics.mean(pks), "window_diff": statistics.mean(wds),
            "chunks_per_doc": statistics.mean(counts), "seconds": time.time() - t0,
        }

    gold_chunks = statistics.mean(len(g) + 1 for _, g in docs)
    if args.json:
        print(json.dumps({"embedder": emb_name, "docs": args.docs,
                          "gold_chunks_per_doc": gold_chunks, "results": results},
                         ensure_ascii=False, indent=2))
        return 0
    print(f"Embedder: `{emb_name}` · {args.docs} documents · seed {args.seed} · "
          f"{'paragraph breaks' if args.paragraphs else 'no layout cues'} · "
          f"{'random' if args.random_start else 'topic-initial'} segments · "
          f"gold segments/doc {gold_chunks:.2f}\n")
    print("| Method | Pk ↓ | WindowDiff ↓ | Chunks/doc |")
    print("|---|---:|---:|---:|")
    for name, r in results.items():
        print(f"| {name} | {r['pk']:.3f} | {r['window_diff']:.3f} | "
              f"{r['chunks_per_doc']:.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
