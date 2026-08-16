"""Uji metrik evaluasi, termasuk diagnostik reduplikasi."""

from disfluency_id.baseline import NaiveDetector, RuleBasedDetector
from disfluency_id.evaluate import (
    error_samples,
    evaluate,
    format_comparison,
    format_report,
    merge,
    reduplication_token_indices,
)
from disfluency_id.schema import FP, O, REP


def test_perfect_predictions_score_one(lex, corpus):
    subset = corpus[:30]
    res = evaluate(subset, [u.labels for u in subset], lex, system="sempurna")
    assert res.accuracy == 1.0
    assert res.micro[2] == 1.0
    assert res.span.prf[2] == 1.0
    assert res.redup_false_cut_rate == 0.0
    assert res.fluent_wrongly_cut == 0


def test_counts_add_up(lex, corpus):
    subset = corpus[:40]
    detector = RuleBasedDetector(lex)
    res = evaluate(subset, [detector.predict(u) for u in subset], lex)
    assert res.n_tokens == sum(len(u) for u in subset)
    assert sum(sum(row.values()) for row in res.confusion.values()) == res.n_tokens


def test_known_confusion_is_recorded(lex, make_utt):
    utt = make_utt([("eee", FP, 0.20), ("saya", O, 0.10), ("mau", O, 0.05)])
    res = evaluate([utt], [[O, O, O]], lex)
    assert res.confusion[FP][O] == 1
    assert res.per_label[FP].fn == 1
    assert res.per_label[FP].tp == 0
    assert res.binary.fn == 1


def test_reduplication_indices_cover_both_spellings(lex, make_utt):
    utt = make_utt(
        [("anak-anak", O, 0.05), ("dan", O, 0.05), ("orang", O, 0.05), ("orang", O, 0.02)]
    )
    idx = reduplication_token_indices(utt, lex)
    assert 0 in idx           # bertanda hubung
    assert {2, 3} <= idx      # ditulis terpisah


def test_reduplication_metric_catches_the_damaging_error(lex, make_utt):
    """Memotong reduplikasi harus tercatat terpisah dari kesalahan lain."""
    utt = make_utt([("anak", O, 0.05), ("anak", O, 0.02), ("mandiri", O, 0.05)])
    res = evaluate([utt], [[REP, O, O]], lex)
    assert res.redup_tokens == 2
    assert res.redup_wrongly_cut == 1
    assert res.redup_false_cut_rate == 0.5
    assert res.redup_preservation == 0.5


def test_naive_system_loses_reduplication_on_the_seed_corpus(lex, corpus):
    """Perbandingan yang menjadi alasan penelitian ini diajukan."""
    naive = evaluate(
        corpus, [NaiveDetector(lex).predict(u) for u in corpus], lex, "naif"
    )
    rule = evaluate(
        corpus, [RuleBasedDetector(lex).predict(u) for u in corpus], lex, "aturan"
    )
    assert naive.redup_preservation < rule.redup_preservation
    assert rule.redup_preservation == 1.0


def test_merge_sums_the_folds(lex, corpus):
    half = len(corpus) // 2
    a = corpus[:half]
    b = corpus[half:]
    detector = RuleBasedDetector(lex)

    ra = evaluate(a, [detector.predict(u) for u in a], lex, "a")
    rb = evaluate(b, [detector.predict(u) for u in b], lex, "b")
    whole = evaluate(corpus, [detector.predict(u) for u in corpus], lex, "utuh")
    merged = merge([ra, rb], "gabungan")

    assert merged.n_tokens == whole.n_tokens
    assert merged.accuracy == whole.accuracy
    assert merged.redup_tokens == whole.redup_tokens


def test_error_samples_only_reports_mistakes(lex, corpus):
    subset = corpus[:50]
    detector = NaiveDetector(lex)
    evaluate(subset, [detector.predict(u) for u in subset], lex)
    for sample in error_samples(subset, limit=10):
        assert sample["gold"] != sample["predicted"]


def test_error_samples_can_focus_on_a_label(lex, corpus):
    subset = corpus[:80]
    detector = NaiveDetector(lex)
    evaluate(subset, [detector.predict(u) for u in subset], lex)
    for sample in error_samples(subset, limit=10, focus=O):
        assert sample["gold"] == O


def test_reports_render(lex, corpus):
    res = evaluate(corpus, [RuleBasedDetector(lex).predict(u) for u in corpus], lex, "aturan")

    text = format_report(res)
    assert "aturan" in text
    assert "Reduplication kept rate" in text
    for lab in ("FP", "DM", "REP"):
        assert f"| {lab} |" in text

    comparison = format_comparison([res])
    assert "Reduplication kept" in comparison
    assert "aturan" in comparison
