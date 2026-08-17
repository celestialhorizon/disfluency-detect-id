"""Uji detektor berbasis aturan.

Uji pertama pada berkas ini memotret persoalan yang melatari penelitian:
pendekatan yang diwarisi dari Bahasa Inggris memotong reduplikasi
gramatikal Bahasa Indonesia. Uji kedua memastikan sistem sadar-reduplikasi
tidak melakukannya.
"""

import pytest

from disfluency_id.baseline import (
    NaiveDetector,
    RuleBasedDetector,
    find_repair_windows,
)
from disfluency_id.schema import DISFLUENT_LABELS, DM, FP, O, PW, REP, RPR


# --------------------------------------------------------------------------
# Persoalan inti
# --------------------------------------------------------------------------


def test_naive_detector_destroys_reduplication(lex, make_utt):
    """Memindahkan asumsi Bahasa Inggris merusak 'anak anak' menjadi 'anak'."""
    utt = make_utt(
        [("anak", O, 0.04), ("anak", O, 0.01), ("sekarang", O, 0.05), ("mandiri", O, 0.05)]
    )
    preds = NaiveDetector(lex).predict(utt)
    assert preds[0] == REP


def test_rule_detector_preserves_reduplication(lex, make_utt):
    utt = make_utt(
        [("anak", O, 0.04), ("anak", O, 0.01), ("sekarang", O, 0.05), ("mandiri", O, 0.05)]
    )
    preds = RuleBasedDetector(lex).predict(utt)
    assert all(p == O for p in preds), preds


def test_rule_detector_still_catches_real_repetition(lex, make_utt):
    """Melindungi reduplikasi tidak boleh berarti melewatkan repetisi asli."""
    utt = make_utt(
        [("saya", O, 0.04), ("saya", O, 0.30), ("mau", O, 0.05), ("tanya", O, 0.05)]
    )
    preds = RuleBasedDetector(lex).predict(utt)
    assert preds[0] == REP
    assert preds[1] == O


def test_repeated_reduplication_is_a_repetition(lex, make_utt):
    """'anak-anak anak-anak' pasti disfluensi: tidak ada reduplikasi ganda."""
    utt = make_utt(
        [("anak-anak", O, 0.04), ("anak-anak", O, 0.10), ("itu", O, 0.05), ("lapar", O, 0.05)]
    )
    preds = RuleBasedDetector(lex).predict(utt)
    assert preds[0] == REP
    assert preds[1] == O


# --------------------------------------------------------------------------
# Filler
# --------------------------------------------------------------------------


def test_filled_pause_detected(lex, make_utt):
    utt = make_utt([("eee", O, 0.20), ("saya", O, 0.10), ("setuju", O, 0.05)])
    preds = RuleBasedDetector(lex).predict(utt)
    assert preds[0] == FP


def test_multiword_discourse_marker_detected(lex, make_utt):
    utt = make_utt(
        [("apa", O, 0.10), ("ya", O, 0.05), ("saya", O, 0.08), ("bingung", O, 0.05)]
    )
    preds = RuleBasedDetector(lex).predict(utt)
    assert preds[0] == DM and preds[1] == DM
    assert preds[2] == O


@pytest.mark.parametrize(
    "words,protected",
    [
        (["ya", "saya", "setuju", "sekali"], 0),          # jawaban afirmatif
        (["apa", "yang", "kamu", "maksud"], 0),           # kata tanya
        (["gitu", "saja", "sudah", "cukup"], 0),          # demonstratif bermakna
    ],
)
def test_meaningful_homonyms_are_not_cut(lex, make_utt, words, protected):
    """Kata yang di tempat lain menjadi filler tidak boleh selalu dipotong."""
    utt = make_utt([(w, O, 0.05) for w in words])
    preds = RuleBasedDetector(lex).predict(utt)
    assert preds[protected] == O, preds


def test_sentence_medial_jadi_is_not_a_filler(lex, make_utt):
    """'jadi' di tengah kalimat adalah konjungsi konsekutif bermakna."""
    utt = make_utt(
        [
            ("hasilnya", O, 0.05),
            ("belum", O, 0.05),
            ("stabil", O, 0.05),
            ("jadi", O, 0.05),
            ("kami", O, 0.05),
            ("menunda", O, 0.05),
        ]
    )
    preds = RuleBasedDetector(lex).predict(utt)
    assert preds[3] == O, preds


def test_utterance_initial_jadi_is_a_filler(lex, make_utt):
    utt = make_utt([("jadi", O, 0.05), ("kami", O, 0.05), ("menunda", O, 0.05)])
    preds = RuleBasedDetector(lex).predict(utt)
    assert preds[0] == DM


# --------------------------------------------------------------------------
# Fragmen kata dan ralat
# --------------------------------------------------------------------------


def test_partial_word_detected(lex, make_utt):
    utt = make_utt(
        [("peng-", O, 0.10), ("pengujiannya", O, 0.15), ("selesai", O, 0.05)]
    )
    preds = RuleBasedDetector(lex).predict(utt)
    assert preds[0] == PW


def test_repair_marks_reparandum_and_editing_term(lex, make_utt):
    """'ke pasar eh ke toko' -> reparandum 'ke pasar' dan 'eh' sama-sama dibuang."""
    utt = make_utt(
        [
            ("saya", O, 0.05),
            ("mau", O, 0.05),
            ("ke", O, 0.05),
            ("pasar", O, 0.05),
            ("eh", O, 0.15),
            ("ke", O, 0.10),
            ("toko", O, 0.05),
        ]
    )
    preds = RuleBasedDetector(lex).predict(utt)
    assert preds[2:5] == [RPR, RPR, RPR], preds
    assert preds[5] == O and preds[6] == O

    utt.assign(preds)
    assert utt.clean_text() == "saya mau ke toko"


def test_repair_window_uses_longest_rough_copy(lex, make_utt):
    utt = make_utt(
        [
            ("filenya", O, 0.05),
            ("di", O, 0.05),
            ("folder", O, 0.05),
            ("unduhan", O, 0.05),
            ("eh", O, 0.15),
            ("di", O, 0.10),
            ("folder", O, 0.05),
            ("dokumen", O, 0.05),
        ]
    )
    windows = find_repair_windows(utt, lex)
    assert windows == [(1, 4)]


def test_editing_term_after_filler_is_not_a_repair(lex, make_utt):
    """'aa maksud saya bukan begitu' bukan ralat, melainkan ujaran bermakna."""
    utt = make_utt(
        [
            ("aa", O, 0.20),
            ("maksud", O, 0.10),
            ("saya", O, 0.05),
            ("bukan", O, 0.05),
            ("begitu", O, 0.05),
            ("arahnya", O, 0.05),
        ]
    )
    preds = RuleBasedDetector(lex).predict(utt)
    assert preds[0] == FP
    assert all(p == O for p in preds[1:]), preds


def test_predictions_have_the_right_length(lex, corpus):
    detector = RuleBasedDetector(lex)
    for utt in corpus[:40]:
        assert len(detector.predict(utt)) == len(utt)


def test_all_predicted_labels_are_valid(lex, corpus):
    detector = RuleBasedDetector(lex)
    allowed = DISFLUENT_LABELS | {O}
    for utt in corpus:
        assert set(detector.predict(utt)) <= allowed
