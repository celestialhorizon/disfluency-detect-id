"""Uji pembacaan korpus, sintesis penanda waktu, dan pembagian data."""

import random

import pytest

from disfluency_id.corpus import (
    TimingProfile,
    build_corpus,
    describe,
    kfold,
    label_counts,
    neutral_timings,
    parse_annotated_line,
    read_annotated,
    simulate_timings,
    split,
)
from disfluency_id.schema import LABELS, O


# --------------------------------------------------------------------------
# Parsing anotasi
# --------------------------------------------------------------------------


def test_parse_inline_tags():
    got = parse_annotated_line("saya/REP saya mau tanya")
    assert got == [("saya", "REP"), ("saya", O), ("mau", O), ("tanya", O)]


def test_parse_untagged_line_is_all_fluent():
    got = parse_annotated_line("penelitian ini memakai pendekatan kuantitatif")
    assert {lab for _, lab in got} == {O}


def test_unknown_tag_is_treated_as_part_of_the_word():
    """Tag tak dikenal tidak boleh diam-diam menjadi label."""
    got = parse_annotated_line("jalan/XYZ terus")
    assert got[0] == ("jalan/XYZ", O)


def test_hyphenated_reduplication_survives_parsing():
    got = parse_annotated_line("anak-anak/REP anak-anak itu")
    assert got[0] == ("anak-anak", "REP")


def test_read_annotated_skips_comments_and_blanks(tmp_path):
    src = tmp_path / "mini.txt"
    src.write_text(
        "# komentar\n\nsaya/REP saya mau\n\n# lagi\nhalo dunia\n",
        encoding="utf-8",
    )
    rows = read_annotated(src)
    assert len(rows) == 2


# --------------------------------------------------------------------------
# Sintesis penanda waktu
# --------------------------------------------------------------------------


def test_timings_are_monotonic_and_non_overlapping():
    words = [("eee", "FP"), ("saya", O), ("saya", "REP"), ("mau", O)]
    tokens = simulate_timings(words, random.Random(3), TimingProfile())
    for a, b in zip(tokens, tokens[1:]):
        assert a.end <= b.start, "token tidak boleh tumpang tindih"
    assert all(t.duration > 0 for t in tokens)


def test_filled_pause_is_longer_than_a_short_word():
    """Jeda terisi memang diucapkan lebih panjang; simulator harus menirunya."""
    rng = random.Random(11)
    fp = [
        simulate_timings([("eee", "FP")], rng)[0].duration for _ in range(40)
    ]
    rng = random.Random(11)
    word = [
        simulate_timings([("ke", O)], rng)[0].duration for _ in range(40)
    ]
    assert sum(fp) / len(fp) > sum(word) / len(word)


def test_simulator_is_reproducible_given_a_seed():
    words = [("saya", "REP"), ("saya", O), ("mau", O)]
    a = simulate_timings(words, random.Random(99))
    b = simulate_timings(words, random.Random(99))
    assert [t.to_dict() for t in a] == [t.to_dict() for t in b]


def test_timing_distributions_deliberately_overlap():
    """Bukti prosodi tidak boleh memisahkan kelas secara sempurna.

    Bila sebaran jeda kedua kelas terpisah bersih, model akan belajar dari
    artefak simulator, bukan dari fenomena kebahasaan. Simulator karena itu
    sengaja menyisipkan kasus yang penanda prosodinya menyesatkan.
    """
    profile = TimingProfile()
    rng = random.Random(5)
    redup_gaps = []
    for _ in range(300):
        toks = simulate_timings([("anak", O), ("anak", O)], rng, profile)
        redup_gaps.append(toks[1].start - toks[0].end)

    misleading = sum(1 for g in redup_gaps if g >= profile.gap_repetition[0])
    assert misleading > 0, "tidak ada kasus reduplikasi yang terdengar disfluen"
    assert misleading < len(redup_gaps) / 2


def test_neutral_timings_sit_between_thresholds(lex):
    """Masukan tanpa audio harus membuat bukti prosodi bernilai nol."""
    tokens = neutral_timings([("anak", O), ("anak", O)])
    gap = tokens[1].start - tokens[0].end
    assert lex.gap_redup_max < gap < lex.gap_disfluency_min


# --------------------------------------------------------------------------
# Korpus benih
# --------------------------------------------------------------------------


def test_seed_corpus_loads(corpus):
    assert len(corpus) > 100
    assert all(len(u) > 0 for u in corpus)


def test_seed_corpus_covers_every_label(corpus):
    counts = label_counts(corpus)
    for lab in LABELS:
        assert counts[lab] > 0, f"label {lab} tidak terwakili di korpus benih"


def test_seed_corpus_is_mostly_fluent(corpus):
    """Disfluensi harus menjadi kelas minoritas, seperti pada ucapan nyata."""
    stats = describe(corpus)
    assert 0.05 < stats["rasio_disfluen"] < 0.35


def test_uids_are_unique(corpus):
    uids = [u.uid for u in corpus]
    assert len(uids) == len(set(uids))


def test_corpus_carries_synthetic_timing_warning(corpus):
    """Metadata wajib menyatakan bahwa penanda waktu bukan hasil pengukuran."""
    assert all(u.meta.get("timing") == "sintetis" for u in corpus)


def test_utterances_are_laid_out_on_one_continuous_timeline(corpus):
    """Tanpa ini, Edit Decision List lintas-ujaran akan saling tindih."""
    for a, b in zip(corpus, corpus[1:]):
        assert a.end < b.start, f"{a.uid} dan {b.uid} tumpang tindih"


def test_non_continuous_layout_restarts_each_utterance():
    utts = build_corpus(continuous=False)
    assert utts[0].start == pytest.approx(utts[1].start)


# --------------------------------------------------------------------------
# Pembagian data
# --------------------------------------------------------------------------


def test_split_is_disjoint_and_complete(corpus):
    train, test = split(corpus, test_ratio=0.3)
    assert len(train) + len(test) == len(corpus)
    assert not ({u.uid for u in train} & {u.uid for u in test})
    assert test


def test_split_is_deterministic(corpus):
    a = [u.uid for u in split(corpus)[1]]
    b = [u.uid for u in split(corpus)[1]]
    assert a == b


@pytest.mark.parametrize("k", [3, 5, 10])
def test_kfold_partitions_the_corpus(corpus, k):
    seen: list[str] = []
    for train, test in kfold(corpus, k=k):
        assert not ({u.uid for u in train} & {u.uid for u in test})
        assert len(train) + len(test) == len(corpus)
        seen.extend(u.uid for u in test)
    assert sorted(seen) == sorted(u.uid for u in corpus)
