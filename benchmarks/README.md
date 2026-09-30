# Benchmark

`run_benchmark.py` measures how well a chunker finds topic boundaries in Arabic.

## Setup

Documents are built Choi-style (Choi, 2000) from `data/topics.json`: 16
hand-written Modern Standard Arabic topics (weather, football, oil economy,
diabetes, AI, date palms, space, cooking, traffic, Abbasid history, desert
ecology, banking, education, labour law, archaeology, tourism), 9 sentences
each. Each document concatenates 4–7 segments; each segment is the first 3–7
sentences of a topic, and adjacent segments never share a topic. The seams
between segments are the gold boundaries.

Scores are **Pk** and **WindowDiff** (lower is better; 0 is perfect, and
≈0.5 is what boundaries unrelated to the content score). All semantic methods
use `max_tokens=110`, `min_tokens=20` and `respect_structure=False`. The
fixed-size baseline packs whole sentences up to the mean gold segment length,
so it gets the right chunk size for free but ignores content.

Two layouts:

- **no layout cues**: segments joined by a single space. Only the content
  says where a topic ends. This is the hard case.
- **paragraph breaks** (`--paragraphs`): segments joined by blank lines,
  as a well-formatted document would be. Here paragraphs line up exactly
  with topic changes, which is more favourable than real documents, where
  a topic usually spans several paragraphs.

## Results (built-in `HashingEmbedder`, 200 documents, seed 13)

| Method | No layout cues: Pk ↓ | WD ↓ | Paragraph breaks: Pk ↓ | WD ↓ |
|---|---:|---:|---:|---:|
| fixed-size (sentence-aware) | 0.476 | 0.476 | 0.476 | 0.476 |
| semantic, greedy | 0.364 | 0.373 | 0.171 | 0.180 |
| semantic, optimal, no discourse cues | 0.319 | 0.328 | 0.061 | 0.068 |
| **semantic, optimal (default)** | **0.304** | **0.313** | **0.062** | **0.069** |

Neural embedders are run by the `benchmark` GitHub Actions workflow (results
in its job summary), because they need a model download.

## What this shows, and what it doesn't

- The optimal (dynamic-programming) segmentation beats greedy thresholding
  in every setting, on both metrics, using the same embeddings.
- The dependency-free embedder is lexical. With no layout cues it finds
  topic changes clearly better than a size-only split, but far from
  perfectly; use a neural embedder for plain text without paragraphs.
- Arabic discourse cues give a small gain without layout cues and none when
  paragraphs exist. The defaults (`window=3`, `discourse_weight=0.05`,
  `paragraph_bonus=0.15`) were chosen with a sweep on seed 99 and confirmed
  on seed 13; with only 16 topics this is a small, synthetic corpus, so
  treat the defaults as a starting point and re-tune on your own documents.
- With `--random-start`, segments may open mid-topic with "كما…" or
  "وهذا…", which real topic changes rarely do. That setting is harder and
  penalises the discourse cues by construction.

## Running

```bash
python benchmarks/run_benchmark.py                   # no layout cues
python benchmarks/run_benchmark.py --paragraphs
python benchmarks/run_benchmark.py --model intfloat/multilingual-e5-base
python benchmarks/run_benchmark.py --json            # machine-readable
```
