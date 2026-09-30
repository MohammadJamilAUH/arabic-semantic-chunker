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

## Results (200 documents, seed 13)

Pk, lower is better. WindowDiff tracks Pk within 0.01 everywhere and is in
the job summaries of the `benchmark` workflow. The fixed-size baseline scores
0.476 in every setting.

| Embedder | Greedy, no layout | **Optimal, no layout** | Greedy, paragraphs | **Optimal, paragraphs** |
|---|---:|---:|---:|---:|
| built-in `HashingEmbedder` | 0.368 | **0.314** | 0.186 | **0.076** |
| `intfloat/multilingual-e5-small` | 0.226 | **0.079** | 0.130 | **0.012** |
| `intfloat/multilingual-e5-base` | 0.183 | **0.056** | 0.138 | **0.021** |
| `paraphrase-multilingual-MiniLM-L12-v2` | 0.194 | **0.064** | 0.149 | **0.009** |

"Optimal" is the default configuration. Neural rows come from the
`benchmark` GitHub Actions workflow, which reruns them on every change to
the library.

## What this shows, and what it doesn't

- The optimal (dynamic-programming) segmentation beats greedy thresholding
  in every setting, on both metrics, using the same embeddings.
- The dependency-free embedder is lexical. With no layout cues it finds
  topic changes clearly better than a size-only split, but far from
  perfectly; use a neural embedder for plain text without paragraphs.
- With a neural embedder the optimal segmentation finds topic changes
  almost perfectly once paragraphs exist, and misses about 6% of
  sentence pairs with no layout at all. Greedy thresholding is 2–15× worse
  with the same embeddings.
- Arabic discourse cues give a small gain with the lexical embedder and are
  neutral (±0.002 Pk) with the neural ones. An earlier version added them
  as a fixed amount and doubled the E5 models' error, because neural
  distances vary over a much narrower range; bonuses are now measured in
  standard deviations of each document's own distances.
- The defaults (`window=3`, `discourse_weight=0.25`, `paragraph_bonus=2.0`)
  were chosen with a sweep on seed 99 and confirmed on seed 13. With 16
  topics this is a small, synthetic corpus, so treat them as a starting
  point and re-tune on your own documents.
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
