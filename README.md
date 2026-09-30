# arabic-semantic-chunker

Structure-aware **semantic chunking for Arabic** documents, built for RAG.

Generic chunkers treat Arabic as "text with spaces". That breaks in practice:
the comma often does the work of a full stop, `د.` and `هـ.` look like sentence
ends, PDF extraction hard-wraps lines and emits presentation-form glyphs, legal
documents are organised by `الباب / الفصل / المادة`, and clitics (`وبالكتاب`)
hide word overlap from lexical similarity. This library handles each of those,
then places chunk boundaries where the topic actually changes.

```python
from arabic_chunker import ArabicSemanticChunker

chunker = ArabicSemanticChunker(max_tokens=256, min_tokens=48)
for chunk in chunker.chunk(text):
    print(chunk.headings, chunk.token_count)
    index(chunk.contextual_text)   # heading path + text: embed this
```

## How it works

```
text ─► sections (headings) ─► sentences ─► clause split ─► embeddings
                                                         │
     chunks ◄── optimal segmentation (DP) ◄── boundary scores + discourse cues
```

1. **Structure first.** Headings are detected (Markdown, `الكتاب/الباب/الفصل/الفرع/المبحث/المادة`
   with digits or ordinal words, the reversed `(31) المادة` produced by PDF
   extraction, `2.3 النطاق`, bold lines, `مقدمة/الخاتمة/التعريفات…`). Sections are
   **hard boundaries**: no chunk ever mixes two articles, and every chunk carries
   its full heading path.
2. **Arabic sentence segmentation** with exact character offsets. Handles
   `؟ ! . …`, closing `» ” )`, abbreviations and initials (`د. محمد`, `أ.د`,
   `U.S.A`), decimals (`3.5`), era markers (`1445 هـ.` ends a sentence),
   numbered-list markers, hard-wrapped PDF lines and bullet lists.
3. **Clause splitting** for over-long sentences: `؛` first, then `،`, then
   discourse connectives (`بينما، إلا أن، حيث، علاوة على…`), then word windows.
4. **Boundary scores.** For each gap between sentences: cosine distance between
   the mean embedding of `window` sentences on each side, **plus Arabic discourse
   cues** — topic-shift openers (`أما، من ناحية أخرى، في المقابل، ختاماً`) push
   towards a cut; continuation openers and anaphora (`لذلك، كما، وبالتالي، هذا،
   هذه`) and list items pull away from one; paragraph breaks add a small bonus.
5. **Optimal segmentation.** A dynamic program chooses the set of cuts that
   maximises total boundary strength (relative to a threshold learned from the
   document's own score distribution) under the `min_tokens`/`max_tokens` budget.
   Greedy threshold chunking — also available as `strategy="greedy"` — commits
   to the first strong gap it sees and so produces runts and mis-placed cuts;
   the DP looks at the whole section.

Chunk text is always an exact slice of the input (`text[c.start:c.end] == c.text`),
so citations and highlighting are trivial. Normalization is used for
matching only.

## Features

| | |
|---|---|
| Normalization | NFKC (fixes PDF presentation forms), diacritics/tatweel removal, alef/yaa/taa-marbuta folding, Arabic-Indic digits, bidi controls, Arabic/Latin run separation |
| Light stemmer | clitic prefixes (`وبال، فال، لل…`), suffixes (`ون، ات، ها، هم…`), accusative tanween alef |
| Embedders | dependency-free `HashingEmbedder` (stems + char n-grams + in-document IDF); `SentenceTransformerEmbedder` (E5, BGE-M3, LaBSE, Arabic models; E5 prefix handled); `CallableEmbedder` for any API |
| Budget | any `length_function` — words by default, or your tokenizer |
| Overlap | `overlap_sentences=n`, never across section boundaries |
| Hierarchical | `chunk_hierarchical()` → parents + children with `parent_id` for small-to-big retrieval |
| Metrics | `pk`, `window_diff` against gold boundaries; label-free `chunk_quality()` |
| CLI | `arabic-chunker file.md --format jsonl --stats` |

## Benchmark

Topic-boundary detection on 200 synthetic Arabic documents (Pk, lower is
better; the fixed-size baseline scores 0.476). Details, caveats and
WindowDiff are in [`benchmarks/`](benchmarks/README.md).

| Embedder | Greedy, no layout | **Optimal, no layout** | **Optimal, paragraphs** |
|---|---:|---:|---:|
| built-in (NumPy only) | 0.368 | **0.314** | **0.076** |
| `multilingual-e5-base` | 0.183 | **0.056** | **0.021** |
| `paraphrase-multilingual-MiniLM-L12-v2` | 0.194 | **0.064** | **0.009** |

Use a neural embedder for plain text without paragraph breaks; the built-in
one is lexical and does well mainly when the document has paragraphs.

## Install

```bash
pip install git+https://github.com/MohammadJamilAUH/arabic-semantic-chunker
# neural embeddings:
pip install "arabic-semantic-chunker[transformers] @ git+https://github.com/MohammadJamilAUH/arabic-semantic-chunker"
```

Only NumPy is required.

## Usage

### Neural embeddings

```python
from arabic_chunker import ArabicSemanticChunker, SentenceTransformerEmbedder

chunker = ArabicSemanticChunker(
    SentenceTransformerEmbedder("intfloat/multilingual-e5-base"),
    max_tokens=300, min_tokens=60,
)
```

Any hosted API works through `CallableEmbedder(lambda texts: client.embed(texts))`.

### Budget in model tokens

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
chunker = ArabicSemanticChunker(length_function=lambda s: len(enc.encode(s)), max_tokens=512)
```

### Parent/child chunks

```python
res = chunker.chunk_hierarchical(text, parent_max_tokens=1024)
for child in res.children:
    store(child.contextual_text, parent=res.parents[child.parent_id].text)
```

### Tuning

| Parameter | Default | Effect |
|---|---|---|
| `max_tokens` / `min_tokens` | 256 / 48 | hard ceiling / effectively-hard floor |
| `threshold_type` | `"percentile"` | also `"std"` (mean + k·σ) and `"iqr"` (Q3 + k·IQR) |
| `threshold_amount` | 70 / 0.5 / 0 | higher → fewer, larger chunks |
| `window` | 3 | sentences averaged on each side of a gap |
| `discourse_weight` | 0.25 | Arabic discourse markers, in std-devs of the document's boundary distances (0 disables) |
| `paragraph_bonus` | 2.0 | preference for cutting at blank lines, same units |
| `strategy` | `"optimal"` | or `"greedy"` |
| `respect_structure` | `True` | heading detection and hard section boundaries |

### CLI

```bash
arabic-chunker examples/sample.md --max-tokens 60 --min-tokens 15 --stats
arabic-chunker report.md --format jsonl --overlap 1 > chunks.jsonl
arabic-chunker law.txt --hierarchical 1024 --model intfloat/multilingual-e5-base
```

### Evaluation

```python
from arabic_chunker import pk, window_diff, chunk_quality
pk(gold_starts, predicted_starts, n_sentences)          # lower is better
chunk_quality(chunks)  # coherence, adjacent similarity, size stats
```

## Development

```bash
pip install -e ".[dev]"
pytest
python benchmarks/run_benchmark.py
```

Releases: bump `version` in `pyproject.toml`, then push a matching tag
(`git tag v0.1.0 && git push origin v0.1.0`). The `release` workflow tests,
builds and publishes to PyPI through trusted publishing (see the one-time
setup at the top of `.github/workflows/release.yml`).

---

## بالعربية

مكتبة لتقطيع النصوص العربية تقطيعاً دلالياً يراعي بنية الوثيقة، مخصّصة لأنظمة
الاسترجاع المعزّز (RAG):

- تكتشف العناوين والبنية القانونية (الباب، الفصل، المادة…) وتجعلها حدوداً لا
  يعبرها أي مقطع، وتُلحق بكل مقطع مسار العناوين الخاص به.
- تقسّم الجمل العربية بدقّة مع مراعاة الاختصارات (د.، أ.د، هـ.) والأرقام العشرية
  والقوائم والأسطر المقطوعة من ملفات PDF، وتحافظ على مواضع الأحرف الأصلية.
- تحسب قوّة الحدّ بين كل جملتين من المسافة الدلالية مضافاً إليها مؤشّرات الخطاب
  العربي (أمّا، من ناحية أخرى ← بداية موضوع؛ لذلك، كما، هذا ← استمرار).
- تختار مواضع القطع المثلى بالبرمجة الديناميكية ضمن حدود الحجم الدنيا والعليا،
  بدلاً من القطع الجشع.
- تعمل دون أي اعتماديات غير NumPy، وتدعم نماذج التضمين العصبية عند الحاجة.

## License

MIT
