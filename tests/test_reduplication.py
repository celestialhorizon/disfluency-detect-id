"""Uji pemisahan reduplikasi morfologis dari repetition disfluency.

Berkas ini menguji klaim inti prototipe. Bila salah satu uji di sini
gagal, klaim penelitian tentang penanganan reduplikasi Bahasa Indonesia
ikut gugur.
"""

import ast
import inspect
from pathlib import Path

import pytest

from disfluency_id import reduplication
from disfluency_id.reduplication import (
    AMBIGUOUS,
    DECISION_MARGIN,
    DISFLUENCY,
    REDUPLICATION,
    RULES,
    WEIGHTS,
    classify_adjacent_repeat,
    find_repeat_events,
    is_hyphenated_reduplication,
    is_partial_word,
    split_reduplication,
    weight_table,
)


# --------------------------------------------------------------------------
# Bentuk bertanda hubung
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "word",
    [
        "anak-anak",      # jamak produktif
        "hati-hati",      # leksikal
        "tiba-tiba",      # leksikal
        "kupu-kupu",      # inheren, tidak punya bentuk tunggal
        "sayur-mayur",    # berima
        "bolak-balik",    # berima
        "berlari-lari",   # berafiks ber-
        "mobil-mobilan",  # berafiks -an
        "tarik-menarik",  # berafiks me-
        "masing-masing",  # kata fungsi yang tetap boleh direduplikasi
    ],
)
def test_hyphenated_forms_recognised_as_reduplication(word, lex):
    assert is_hyphenated_reduplication(word, lex), word


@pytest.mark.parametrize("word", ["saya", "anak", "peneliti", "sistem"])
def test_plain_words_are_not_hyphenated_reduplication(word, lex):
    assert not is_hyphenated_reduplication(word, lex)


def test_split_reduplication():
    assert split_reduplication("anak-anak") == ("anak", "anak")
    assert split_reduplication("sayur-mayur") == ("sayur", "mayur")
    assert split_reduplication("saya") is None
    assert split_reduplication("a-b-c") is None


# --------------------------------------------------------------------------
# Putusan pasangan identik: bukti leksikal vs bukti prosodi
# --------------------------------------------------------------------------


def test_function_word_repeat_is_disfluency_even_when_spoken_fast(lex):
    """Bukti leksikal harus mengalahkan bukti prosodi.

    'saya saya' diucapkan tanpa jeda tetap repetisi disfluen: kata fungsi
    Bahasa Indonesia tidak direduplikasi secara produktif.
    """
    kind, score, reasons = classify_adjacent_repeat("saya", gap=0.01, lex=lex)
    assert kind == DISFLUENCY
    assert score < 0
    assert any("function word" in r for r in reasons)


def test_known_reduplication_survives_a_long_pause(lex):
    """'anak anak' dengan jeda panjang tetap dipertahankan.

    Penutur boleh saja ragu di tengah reduplikasi; bentuk yang terdaftar
    tidak boleh dipotong hanya karena jedanya melebar.
    """
    kind, score, _ = classify_adjacent_repeat("anak", gap=0.30, lex=lex)
    assert kind == REDUPLICATION
    assert score > 0


def test_open_class_word_decided_by_prosody(lex):
    """Untuk kata kelas terbuka tak terdaftar, jedalah yang menentukan."""
    fast, _, _ = classify_adjacent_repeat("peneliti", gap=0.01, lex=lex)
    slow, _, _ = classify_adjacent_repeat("peneliti", gap=0.35, lex=lex)
    assert fast == REDUPLICATION
    assert slow == DISFLUENCY


def test_grey_zone_yields_ambiguous(lex):
    """Jeda di antara kedua ambang tidak boleh memaksa keputusan."""
    mid = (lex.gap_redup_max + lex.gap_disfluency_min) / 2
    kind, _, _ = classify_adjacent_repeat("peneliti", gap=mid, lex=lex)
    assert kind == AMBIGUOUS


def test_asymmetric_duration_pushes_towards_disfluency(lex):
    """Salinan pertama yang terpotong adalah ciri repetisi, bukan reduplikasi."""
    _, symmetric, _ = classify_adjacent_repeat(
        "peneliti", gap=0.10, lex=lex, dur_first=0.30, dur_second=0.30
    )
    _, clipped, _ = classify_adjacent_repeat(
        "peneliti", gap=0.10, lex=lex, dur_first=0.10, dur_second=0.30
    )
    assert clipped < symmetric


def test_longer_run_pushes_towards_disfluency(lex):
    """Reduplikasi Bahasa Indonesia menggandakan sekali, tidak lebih."""
    _, twice, _ = classify_adjacent_repeat("sistem", gap=0.02, lex=lex, run_length=2)
    _, thrice, reasons = classify_adjacent_repeat(
        "sistem", gap=0.02, lex=lex, run_length=3
    )
    assert thrice < twice
    assert any("times in a row" in r for r in reasons)


def test_filler_repeat_is_disfluency(lex):
    kind, _, _ = classify_adjacent_repeat("kayak", gap=0.02, lex=lex)
    assert kind == DISFLUENCY


def test_every_decision_carries_reasons(lex):
    """Putusan tanpa alasan tidak dapat diaudit; ini syarat wajib."""
    for base, gap in [("saya", 0.3), ("anak", 0.01), ("peneliti", 0.13)]:
        _, _, reasons = classify_adjacent_repeat(base, gap=gap, lex=lex)
        assert reasons


# --------------------------------------------------------------------------
# Tabel bobot: satu sumber, dan penjaga supaya tetap satu
#
# Bobotnya pernah diketik dua kali -- sekali di kode, sekali di tabel
# spesifikasi notebook -- lalu tiga kaidah baru masuk ke kode tanpa masuk ke
# tabel. Melencengnya tidak kelihatan: tiap baris tabel tetap benar, yang
# salah adalah baris yang tidak ada. Uji di bawah ini yang menutup celah itu.
# --------------------------------------------------------------------------


def _classifier_tree() -> ast.FunctionDef:
    src = inspect.getsource(classify_adjacent_repeat)
    return ast.parse(inspect.cleandoc(src)).body[0]


def _weight_keys_used_by_classifier() -> set[str]:
    used = set()
    for node in ast.walk(_classifier_tree()):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "WEIGHTS"
            and isinstance(node.slice, ast.Constant)
        ):
            used.add(node.slice.value)
    return used


def test_score_is_never_moved_by_a_number_typed_inline():
    """Tiap sumbangan skor harus lewat `WEIGHTS`, bukan angka di tempat.

    Ini syarat yang bikin tabel spesifikasi tidak bisa lagi melenceng: bobot
    baru yang diketik langsung di sini tidak akan pernah muncul di tabel.
    """
    for node in ast.walk(_classifier_tree()):
        if isinstance(node, ast.AugAssign) and getattr(node.target, "id", "") == "score":
            assert not isinstance(node.value, ast.Constant), (
                f"baris {node.lineno} menggeser skor dengan angka telanjang; "
                "pakai WEIGHTS[...] supaya ikut tercetak di tabel"
            )


def test_every_weight_is_actually_applied():
    """Baris tabel yang tidak dipakai kodenya sama menyesatkannya."""
    assert _weight_keys_used_by_classifier() == set(WEIGHTS)


def test_every_weight_has_a_table_row():
    keyed = {r.key for r in RULES if r.interpolates is None}
    assert keyed == set(WEIGHTS)


def test_interpolated_row_points_at_real_weights():
    for rule in RULES:
        if rule.interpolates is not None:
            assert rule.key not in WEIGHTS  # tidak punya bobot sendiri
            assert all(k in WEIGHTS for k in rule.interpolates)


def test_interpolated_row_shows_both_ends():
    """Baris zona abu-abu harus terbaca sebagai rentang, bukan satu angka.

    Versi pertama mengganti titik jadi koma pada seluruh teks, sehingga
    pemisah rentangnya ikut jadi koma: '+1,5 ,, -1,5'.
    """
    (rule,) = [r for r in RULES if r.interpolates is not None]
    text = rule.weight_text()
    for key in rule.interpolates:
        assert f"{WEIGHTS[key]:+.1f}".replace(".", ",") in text
    assert ",," not in text


def test_rendered_table_carries_the_live_numbers(lex):
    table = weight_table(lex)
    for value in WEIGHTS.values():
        assert f"{value:+.1f}".replace(".", ",") in table
    # Ambang jedanya dari berkas leksikon, jadi ikut terbawa kalau berubah.
    assert "0,08" in table and "0,18" in table
    assert str(DECISION_MARGIN).replace(".", ",") in table


def test_table_is_the_source_the_classifier_reads(monkeypatch, lex):
    """Bukti bahwa tabelnya sungguh dibaca, bukan cuma kebetulan cocok."""
    _, before, _ = classify_adjacent_repeat("peneliti", gap=0.01, lex=lex)
    monkeypatch.setitem(WEIGHTS, "open_class", WEIGHTS["open_class"] + 10.0)
    _, after, _ = classify_adjacent_repeat("peneliti", gap=0.01, lex=lex)
    assert after == pytest.approx(before + 10.0)


def test_grey_zone_interpolation_meets_both_ends(lex):
    """Ujung zona abu-abu harus menyambung mulus ke kedua bobot jedanya."""
    eps = 1e-6
    _, near, _ = classify_adjacent_repeat("peneliti", gap=lex.gap_redup_max + eps, lex=lex)
    _, far, _ = classify_adjacent_repeat(
        "peneliti", gap=lex.gap_disfluency_min - eps, lex=lex
    )
    _, at_near, _ = classify_adjacent_repeat("peneliti", gap=lex.gap_redup_max, lex=lex)
    _, at_far, _ = classify_adjacent_repeat(
        "peneliti", gap=lex.gap_disfluency_min, lex=lex
    )
    assert near == pytest.approx(at_near, abs=1e-3)
    assert far == pytest.approx(at_far, abs=1e-3)


def test_grey_zone_is_monotone(lex):
    """Jeda melebar tidak boleh menaikkan skor reduplikasi."""
    gaps = [0.09, 0.11, 0.13, 0.15, 0.17]
    scores = [classify_adjacent_repeat("peneliti", gap=g, lex=lex)[1] for g in gaps]
    assert scores == sorted(scores, reverse=True)


def test_module_exposes_the_table_for_the_notebook():
    """Notebook mencetak tabelnya, tidak mengetik ulang."""
    assert callable(reduplication.weight_table)


def test_readme_table_is_the_rendered_one(lex):
    """README tidak bisa dijalankan, jadi tabelnya dikunci uji ini.

    Versi ketikan tangan di README ketinggalan tiga kaidah selama berbulan
    -- persis kesalahan yang sama dengan tabel di notebook.
    """
    readme = Path(__file__).resolve().parent.parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- TABEL-BOBOT", "<!-- /TABEL-BOBOT -->"
    assert start in text and end in text, "penanda blok tabel hilang dari README"
    block = text.split(start, 1)[1].split("\n", 1)[1].split(end, 1)[0]
    expected = weight_table(lex)
    assert block.strip() == expected.strip(), (
        "tabel di README tidak lagi sama dengan weight_table(); "
        "ganti isi blok TABEL-BOBOT dengan:\n\n" + expected
    )


# --------------------------------------------------------------------------
# Penelusuran pada ujaran
# --------------------------------------------------------------------------


def test_find_repeat_events_on_utterance(lex, make_utt):
    utt = make_utt(
        [
            ("saya", "REP", 0.05),
            ("saya", "O", 0.30),
            ("lihat", "O", 0.04),
            ("anak", "O", 0.04),
            ("anak", "O", 0.01),
            ("itu", "O", 0.04),
        ]
    )
    events = find_repeat_events(utt, lex)
    assert len(events) == 2
    by_base = {e.base: e for e in events}
    assert by_base["saya"].kind == DISFLUENCY
    assert by_base["anak"].kind == REDUPLICATION


def test_find_repeat_events_handles_triple_run(lex, make_utt):
    utt = make_utt(
        [("yang", "REP", 0.05), ("yang", "REP", 0.25), ("yang", "O", 0.25), ("jelas", "O", 0.04)]
    )
    events = find_repeat_events(utt, lex)
    assert len(events) == 2
    assert all(e.kind == DISFLUENCY for e in events)


def test_no_events_when_no_adjacent_repeats(lex, plain_utt):
    utt = plain_utt("kami membandingkan tiga model dengan konfigurasi sama")
    assert find_repeat_events(utt, lex) == []


def test_confidence_within_bounds(lex, make_utt):
    utt = make_utt([("saya", "REP", 0.05), ("saya", "O", 0.30)])
    for ev in find_repeat_events(utt, lex):
        assert 0.0 <= ev.confidence <= 1.0


# --------------------------------------------------------------------------
# Fragmen kata
# --------------------------------------------------------------------------


def test_trailing_hyphen_is_partial_word(lex):
    assert is_partial_word("peng-", "pengujiannya", gap_after=0.10, lex=lex)
    assert is_partial_word("sis-", "sistemnya", gap_after=0.10, lex=lex)


def test_prefix_without_hyphen_detected_when_spoken_closely(lex):
    assert is_partial_word("peng", "pengujiannya", gap_after=0.10, lex=lex)


def test_long_pause_rejects_bare_prefix(lex):
    """Tanpa tanda hubung, jeda panjang berarti dua kata utuh yang berbeda."""
    assert not is_partial_word("peng", "pengujiannya", gap_after=0.9, lex=lex)


def test_function_word_is_not_a_fragment(lex):
    """'ke' adalah preposisi utuh, bukan potongan dari 'kelas'."""
    assert not is_partial_word("ke", "kelas", gap_after=0.05, lex=lex)


def test_identical_words_are_not_fragments(lex):
    assert not is_partial_word("anak", "anak", gap_after=0.02, lex=lex)
