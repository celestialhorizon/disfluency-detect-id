"""Uji leksikon: normalisasi, pencarian kata tunggal, dan frasa."""

import pytest

from disfluency_id.lexicon import (
    CAT_DM,
    CAT_EDIT,
    CAT_FP,
    canonical,
    collapse_repeats,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("eeee", "ee"),
        ("eee", "ee"),
        ("ee", "ee"),
        ("hmmmm", "hmm"),
        ("aaaaa", "aa"),
        ("saya", "saya"),
        ("anak-anak", "anak-anak"),
        ("", ""),
    ],
)
def test_collapse_repeats(raw, expected):
    assert collapse_repeats(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("EEEE", "ee"),
        ("Saya,", "saya"),
        ("(anak-anak)", "anak-anak"),
        ("Bandung.", "bandung"),
        ("sa-", "sa-"),
    ],
)
def test_canonical_normalises_case_and_punctuation(raw, expected):
    assert canonical(raw) == expected


def test_canonical_keeps_trailing_hyphen(lex):
    """Tanda hubung di akhir menandai fragmen kata dan tidak boleh hilang."""
    assert canonical("peng-").endswith("-")


def test_filled_pause_variants_all_resolve(lex):
    for form in ["eee", "eeee", "EEE", "emm", "hmm", "hmmmm", "anu"]:
        assert lex.category(form) == CAT_FP, form


def test_discourse_markers_detected(lex):
    for form in ["nah", "kayak", "gitu", "pokoknya", "sih"]:
        assert lex.category(form) == CAT_DM, form


def test_editing_terms_detected(lex):
    assert lex.category("maaf") == CAT_EDIT
    assert lex.category("maksudnya") == CAT_EDIT


def test_ordinary_words_are_not_fillers(lex):
    for form in ["penelitian", "responden", "kuesioner", "regresi", "anak"]:
        assert lex.category(form) is None, form


def test_meaningful_homonyms_excluded_from_lexicon(lex):
    """Bentuk baku yang lebih sering bermakna sengaja tidak didaftar."""
    for form in ["begitu", "begini", "intinya", "salah", "atau"]:
        assert lex.category(form) is None, form


def test_multiword_match_longest_first(lex):
    words = ["apa", "ya", "saya", "bingung"]
    hit = lex.match_multiword(words, 0)
    assert hit is not None
    length, cat = hit
    assert length == 2
    assert cat == CAT_DM


def test_multiword_no_false_match(lex):
    assert lex.match_multiword(["saya", "bingung", "sekali"], 0) is None


def test_function_words(lex):
    assert lex.is_function_word("yang")
    assert lex.is_function_word("saya")
    assert not lex.is_function_word("peneliti")


def test_reduplicable_excludes_function_words(lex):
    assert not lex.is_reduplicable("yang")
    assert not lex.is_reduplicable("saya")
    assert lex.is_reduplicable("anak")
    assert lex.is_reduplicable("peneliti")


def test_reduplicable_allows_listed_exceptions(lex):
    """Sebagian kata fungsi tetap punya bentuk reduplikasi yang sah."""
    assert lex.is_reduplicable("masing-masing")
    assert lex.is_reduplicable("tiap-tiap")


def test_thresholds_loaded(lex):
    assert 0 < lex.gap_redup_max < lex.gap_disfluency_min


# ------------------------------------------------- penanda ralat kuat vs ambigu


def test_strong_editing_terms_are_a_subset_of_editing_terms(lex):
    assert lex.editing_term_strong
    assert lex.editing_term_strong <= lex.editing_term


def test_bukan_is_an_editing_term_but_not_a_strong_one(lex):
    """'bukan' memang bisa menandai koreksi, tetapi jauh lebih sering
    menjadi pengingkar biasa. Ia terdaftar agar detektor bisa memakainya
    bila konteks mendukung, tanpa keanggotaan itu sendiri memutuskan."""
    assert lex.category("bukan") == CAT_EDIT
    assert not lex.is_strong_editing_term("bukan")


def test_ralat_is_strong(lex):
    assert lex.is_strong_editing_term("ralat")
    assert lex.is_strong_editing_term("Maksudnya,")


def test_strong_lookup_is_canonicalised(lex):
    assert lex.is_strong_editing_term("MAAF")


def test_missing_strong_key_falls_back_to_previous_behaviour(tmp_path):
    """Leksikon kustom yang disusun sebelum kunci `editing_term_kuat` ada
    harus tetap berperilaku seperti semula, bukan kehilangan seluruh
    penanda kuatnya secara diam-diam."""
    import json
    import shutil

    from disfluency_id.lexicon import DATA_DIR, DEFAULT_EDITING_TERM_STRONG, Lexicon

    shutil.copytree(DATA_DIR, tmp_path / "lex")
    p = tmp_path / "lex" / "fillers_id.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    del data["editing_term_kuat"]
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    lama = Lexicon.load(tmp_path / "lex")
    assert lama.editing_term_strong == set(DEFAULT_EDITING_TERM_STRONG)
