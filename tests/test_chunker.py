from pathlib import Path

import numpy as np
import pytest

from arabic_chunker import (ArabicSemanticChunker, CallableEmbedder, HashingEmbedder,
                            chunk_quality, count_words, detect_heading, discourse_bias,
                            light_stem, normalize_arabic, parse_sections, pk,
                            split_long, split_sentences, window_diff)

SAMPLE = (Path(__file__).parent.parent / "examples" / "sample.md").read_text(encoding="utf-8")

WEATHER = [
    "تشهد الدولة اليوم طقساً حاراً ورطباً في المناطق الساحلية.",
    "وتتوقع الأرصاد الجوية ارتفاع درجات الحرارة إلى خمس وأربعين درجة.",
    "كما تنشط الرياح المثيرة للغبار والأتربة في المناطق الداخلية.",
    "وتنصح الأرصاد بتجنب التعرض لأشعة الشمس وقت الظهيرة.",
    "ويُتوقع أن يستمر الطقس الحار والرطب حتى نهاية الأسبوع.",
]
FOOTBALL = [
    "فاز المنتخب الوطني لكرة القدم على نظيره بثلاثة أهداف نظيفة.",
    "وسجّل مهاجم المنتخب هدفين في الشوط الأول من المباراة.",
    "وأشاد مدرب المنتخب بأداء اللاعبين والتزامهم بالخطة.",
    "ويستعد المنتخب لمواجهة قوية في الدور القادم من البطولة.",
    "وتذاكر المباراة القادمة للمنتخب نفدت خلال ساعات من طرحها.",
]


# ---------------------------------------------------------------- normalize
def test_normalize_folds_variants_and_digits():
    assert normalize_arabic("إِنَّ الأُمَّةَ") == "ان الامه"
    assert normalize_arabic("عام ٢٠٢٤") == "عام 2024"
    assert normalize_arabic("قانونGDPR") == "قانون gdpr"
    assert normalize_arabic("كتـــاب") == "كتاب"
    # presentation forms from PDF extraction
    assert normalize_arabic("ﻣﺮﺣﺒﺎ") == "مرحبا"


def test_light_stem_keeps_triliteral_core():
    assert light_stem("والكتاب") == "كتاب"
    assert light_stem("المعلمون") == "معلم"
    assert light_stem("علم") == "علم"


# ---------------------------------------------------------------- sentences
def test_sentence_split_offsets_are_exact():
    text = "هذه جملة أولى. وهذه جملة ثانية؟ وهذه ثالثة!"
    spans = split_sentences(text)
    assert [s.text for s in spans] == ["هذه جملة أولى.", "وهذه جملة ثانية؟", "وهذه ثالثة!"]
    for s in spans:
        assert text[s.start:s.end] == s.text


def test_sentence_split_guards_abbreviations_and_decimals():
    text = "قال د. محمد إن النسبة بلغت 3.5 بالمئة عام 1445 هـ. ثم غادر."
    spans = split_sentences(text)
    assert len(spans) == 2
    assert spans[0].text.endswith("هـ.")


def test_sentence_split_joins_wrapped_lines_but_breaks_lists():
    text = "هذه جملة طويلة انقطعت\nفي منتصفها بسبب الاستخراج.\n- بند أول\n- بند ثان"
    spans = [s.text for s in split_sentences(text)]
    assert spans[0] == "هذه جملة طويلة انقطعت\nفي منتصفها بسبب الاستخراج."
    assert spans[1:] == ["- بند أول", "- بند ثان"]


def test_paragraph_start_flag():
    spans = split_sentences("جملة أولى. جملة ثانية.\n\nفقرة جديدة.")
    assert [s.paragraph_start for s in spans] == [True, False, True]


def test_split_long_prefers_clause_boundaries():
    text = ("ينص القانون على حماية البيانات الشخصية للمواطنين والمقيمين؛ "
            "ويلزم الجهات الحكومية والخاصة بتعيين مسؤول لحماية البيانات، "
            "ويحدد العقوبات المترتبة على المخالفين")
    span = split_sentences(text)[0]
    parts = split_long(span, 10)
    assert all(count_words(p.text) <= 10 for p in parts)
    assert parts[0].text.endswith("؛")
    for p in parts:
        assert text[p.start:p.end] == p.text


# ---------------------------------------------------------------- structure
@pytest.mark.parametrize("line,level", [
    ("# عنوان", 1), ("### فرعي", 3), ("الباب الأول: أحكام عامة", 1),
    ("الفصل الثاني", 2), ("المادة (12)", 4), ("مادة 5 - التعريفات", 4),
    ("(31) المادة", 4), ("2.3 نطاق التطبيق", 2), ("**الأهداف**", 2), ("مقدمة", 2),
])
def test_detect_heading(line, level):
    found = detect_heading(line)
    assert found is not None and found[1] == level


@pytest.mark.parametrize("line", [
    "هذه جملة عادية وليست عنواناً.",
    "1. نص طويل ينتهي بنقطة.",
    "",
])
def test_non_headings(line):
    assert detect_heading(line) is None


def test_parse_sections_builds_heading_paths():
    secs = parse_sections(SAMPLE)
    paths = [s.headings for s in secs]
    assert ("سياسة حماية البيانات الشخصية", "مقدمة") in paths
    assert ("سياسة حماية البيانات الشخصية", "العقوبات", "المادة (2)") in paths


# ---------------------------------------------------------------- discourse
def test_discourse_bias_signs():
    assert discourse_bias("أما من الناحية المالية فالوضع مستقر.") > 0
    assert discourse_bias("لذلك قررت الإدارة تأجيل المشروع.") < 0
    assert discourse_bias("الشركة تعمل في قطاع الطاقة.") == 0


# ---------------------------------------------------------------- chunker
def test_chunks_are_exact_slices_and_respect_budget():
    ch = ArabicSemanticChunker(max_tokens=40, min_tokens=10)
    chunks = ch.chunk(SAMPLE)
    assert chunks
    for c in chunks:
        assert SAMPLE[c.start:c.end] == c.text
        assert c.token_count <= 40
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunks_never_cross_sections():
    ch = ArabicSemanticChunker(max_tokens=500, min_tokens=10)
    chunks = ch.chunk(SAMPLE)
    articles = [c for c in chunks if c.headings and c.headings[-1].startswith("المادة")]
    assert len(articles) == 2
    assert all("المادة" not in c.text for c in chunks)


def test_finds_topic_shift_without_structure():
    text = " ".join(WEATHER + FOOTBALL)
    ch = ArabicSemanticChunker(max_tokens=70, min_tokens=20, respect_structure=False)
    chunks = ch.chunk(text)
    assert len(chunks) == 2
    assert chunks[1].text.startswith("فاز المنتخب")


def test_greedy_strategy_respects_budget():
    text = " ".join(WEATHER + FOOTBALL)
    chunks = ArabicSemanticChunker(max_tokens=40, min_tokens=10, strategy="greedy",
                                   respect_structure=False).chunk(text)
    assert len(chunks) >= 3
    assert all(c.token_count <= 40 for c in chunks)


def test_optimal_honours_min_tokens_even_with_low_threshold():
    # A low threshold rewards almost every gap; the size constraint must still hold.
    text = " ".join(WEATHER + FOOTBALL)
    chunks = ArabicSemanticChunker(max_tokens=70, min_tokens=25, respect_structure=False,
                                   threshold_amount=10).chunk(text)
    assert len(chunks) >= 2
    assert all(25 <= c.token_count <= 70 for c in chunks)


def test_overlap_repeats_previous_sentence():
    text = " ".join(WEATHER + FOOTBALL)
    ch = ArabicSemanticChunker(max_tokens=70, min_tokens=20, overlap_sentences=1,
                               respect_structure=False)
    a, b = ch.chunk(text)
    assert b.metadata["overlap_sentences"] == 1
    assert b.text.startswith(WEATHER[-1])


def test_contextual_text_prefixes_headings():
    c = ArabicSemanticChunker(max_tokens=200, min_tokens=5).chunk(SAMPLE)[0]
    assert c.contextual_text.startswith("سياسة حماية البيانات الشخصية › مقدمة\n\n")


def test_hierarchical_children_nest_in_parents():
    ch = ArabicSemanticChunker(max_tokens=30, min_tokens=8)
    res = ch.chunk_hierarchical(SAMPLE, parent_max_tokens=100)
    assert res.parents and len(res.children) >= len(res.parents)
    for child in res.children:
        parent = res.parents[child.parent_id]
        assert parent.start <= child.start and child.end <= parent.end


def test_custom_embedder_and_length_function():
    calls = []

    def fake(texts):
        calls.append(len(texts))
        return np.eye(len(texts), 8)[:, :8] + 0.01

    ch = ArabicSemanticChunker(CallableEmbedder(fake), max_tokens=400, min_tokens=5,
                               length_function=len)
    assert ch.chunk(SAMPLE)
    assert calls


def test_empty_and_tiny_inputs():
    ch = ArabicSemanticChunker()
    assert ch.chunk("") == []
    assert ch.chunk("   \n\n ") == []
    [only] = ch.chunk("جملة واحدة فقط.")
    assert only.text == "جملة واحدة فقط."


def test_invalid_config():
    with pytest.raises(ValueError):
        ArabicSemanticChunker(max_tokens=10, min_tokens=20)
    with pytest.raises(ValueError):
        ArabicSemanticChunker(strategy="magic")


# ---------------------------------------------------------------- embeddings & metrics
def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    a = e.embed(WEATHER[:2])
    b = e.embed(WEATHER[:2])
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_hashing_embedder_separates_topics():
    emb = HashingEmbedder().embed(WEATHER + FOOTBALL)
    within = float(emb[0] @ emb[1])
    across = float(emb[0] @ emb[5])
    assert within > across


def test_pk_and_window_diff():
    assert pk([5], [5], 10) == 0.0
    assert window_diff([5], [5], 10) == 0.0
    assert pk([5], [2], 10) > 0
    assert window_diff([5], [2, 5, 8], 10) > 0


def test_chunk_quality_report():
    chunks = ArabicSemanticChunker(max_tokens=60, min_tokens=10).chunk(SAMPLE)
    q = chunk_quality(chunks)
    assert q["chunks"] == len(chunks)
    assert 0 < q["mean_coherence"] <= 1
