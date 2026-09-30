"""Command-line interface: ``arabic-chunker FILE [options]``."""

from __future__ import annotations

import argparse
import json
import sys

from .chunker import ArabicSemanticChunker
from .metrics import chunk_quality


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="arabic-chunker",
                                 description="Semantic chunking for Arabic text.")
    ap.add_argument("file", help="UTF-8 text/Markdown file, or - for stdin")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--min-tokens", type=int, default=48)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--strategy", choices=["optimal", "greedy"], default="optimal")
    ap.add_argument("--threshold-type", choices=["percentile", "std", "iqr"],
                    default="percentile")
    ap.add_argument("--threshold-amount", type=float, default=None)
    ap.add_argument("--overlap", type=int, default=0, help="overlap in sentences")
    ap.add_argument("--no-structure", action="store_true",
                    help="ignore headings/sections")
    ap.add_argument("--model", default=None,
                    help="sentence-transformers model (default: built-in hashing embedder)")
    ap.add_argument("--hierarchical", type=int, metavar="PARENT_MAX", default=None,
                    help="emit parent/child chunks with this parent budget")
    ap.add_argument("--format", choices=["json", "jsonl", "text"], default="text")
    ap.add_argument("--stats", action="store_true", help="print quality metrics to stderr")
    args = ap.parse_args(argv)

    text = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    embedder = None
    if args.model:
        from .embeddings import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder(args.model)
    chunker = ArabicSemanticChunker(
        embedder, max_tokens=args.max_tokens, min_tokens=args.min_tokens,
        window=args.window, strategy=args.strategy,
        threshold_type=args.threshold_type, threshold_amount=args.threshold_amount,
        overlap_sentences=args.overlap, respect_structure=not args.no_structure,
    )

    if args.hierarchical:
        res = chunker.chunk_hierarchical(text, parent_max_tokens=args.hierarchical)
        payload = {"parents": [c.to_dict() for c in res.parents],
                   "children": [c.to_dict() for c in res.children]}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        chunks = res.children
    else:
        chunks = chunker.chunk(text)
        if args.format == "json":
            print(json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2))
        elif args.format == "jsonl":
            for c in chunks:
                print(json.dumps(c.to_dict(), ensure_ascii=False))
        else:
            for c in chunks:
                path = " › ".join(c.headings)
                print(f"── chunk {c.index} · {c.token_count} tokens"
                      f"{' · ' + path if path else ''} ──")
                print(c.text)
                print()
    if args.stats:
        print(json.dumps(chunk_quality(chunks), ensure_ascii=False), file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
