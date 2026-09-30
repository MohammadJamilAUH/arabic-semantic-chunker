"""Run: python examples/quickstart.py"""

from pathlib import Path

from arabic_chunker import ArabicSemanticChunker, chunk_quality

text = (Path(__file__).parent / "sample.md").read_text(encoding="utf-8")

chunker = ArabicSemanticChunker(max_tokens=60, min_tokens=15)
chunks = chunker.chunk(text)
for c in chunks:
    print(f"[{c.index}] {' › '.join(c.headings)}  ({c.token_count} tokens, "
          f"coherence={c.coherence})")
    print(c.text, "\n")
print(chunk_quality(chunks))

# Small-to-big retrieval: index children, return their parent to the LLM.
res = chunker.chunk_hierarchical(text, parent_max_tokens=150)
print(f"{len(res.parents)} parents / {len(res.children)} children")

# Swap in a neural embedder (pip install 'arabic-semantic-chunker[transformers]'):
# from arabic_chunker import SentenceTransformerEmbedder
# chunker = ArabicSemanticChunker(SentenceTransformerEmbedder("intfloat/multilingual-e5-base"))
