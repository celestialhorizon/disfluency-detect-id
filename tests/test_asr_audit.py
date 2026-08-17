"""Uji audit mutu transkrip ASR."""

import pytest

from disfluency_id.asr_audit import (
    LAJU_DISFLUENSI_LISAN_SPONTAN,
    LAJU_FILLER_LISAN_SPONTAN,
    LAJU_JEDA_TERISI_LISAN_SPONTAN,
    _neutral_gap,
    audit,
    audit_dict,
    format_audit,
)
from disfluency_id.lexicon import Lexicon
from disfluency_id.reduplication import classify_adjacent_repeat
from disfluency_id.schema import O, Token, Utterance


@pytest.fixture(scope="module")
def lex():
    return Lexicon.load()


def _utt(spec, uid="t-0001"):
    """Bangun ujaran dari (kata, mulai, selesai)."""
    return Utterance(
        uid=uid,
        tokens=[Token(text=w, start=s, end=e, label=O) for w, s, e in spec],
        source="uji",
    )


def _chain(words, dur=0.30, gap=0.12):
    """Ujaran dengan jeda seragam yang bisa dikendalikan."""
    spec, t = [], 0.0
    for w in words:
        spec.append((w, t, t + dur))
        t += dur + gap
    return _utt(spec)


# ---------------------------------------------------------------- penanda waktu


def test_zero_gaps_are_counted_and_flagged(lex):
    """Penanda waktu bersambung persis adalah artefak penjajaran."""
    utt = _utt([("satu", 0.0, 0.4), ("dua", 0.4, 0.8), ("tiga", 0.8, 1.2)])
    a = audit([utt], lex)
    assert a.gaps.total == 2
    assert a.gaps.zero == 2
    assert a.gaps.zero_share == pytest.approx(1.0)
    assert not a.gaps.prosody_is_trustworthy


def test_realistic_gaps_pass_the_check(lex):
    a = audit([_chain(["saya", "mau", "bilang", "sesuatu"], gap=0.15)], lex)
    assert a.gaps.zero == 0
    assert a.gaps.prosody_is_trustworthy


def test_gap_zones_partition_every_measured_gap(lex):
    """Tiap jeda jatuh pada tepat satu zona; totalnya harus utuh."""
    utt = _utt(
        [
            ("a", 0.00, 0.30),
            ("b", 0.32, 0.62),  # 0.02 -> zona reduplikasi
            ("c", 0.75, 1.05),  # 0.13 -> zona abu-abu
            ("d", 1.35, 1.65),  # 0.30 -> zona disfluensi
        ]
    )
    g = audit([utt], lex).gaps
    assert g.in_redup_zone + g.in_grey_zone + g.in_disfluency_zone == g.total == 3
    assert (g.in_redup_zone, g.in_grey_zone, g.in_disfluency_zone) == (1, 1, 1)


def test_median_gap_reported(lex):
    a = audit([_chain(["a", "b", "c", "d", "e"], gap=0.20)], lex)
    assert a.gaps.median == pytest.approx(0.20, abs=1e-9)


def test_negative_gaps_are_clamped_not_dropped(lex):
    """Penanda waktu ASR kadang tumpang tindih; itu tetap jeda nol,
    bukan jeda negatif yang menarik median ke bawah nol."""
    utt = _utt([("satu", 0.0, 0.50), ("dua", 0.45, 0.90)])
    g = audit([utt], lex).gaps
    assert g.total == 1
    assert g.zero == 1
    assert min(g.values) >= 0.0


# --------------------------------------------------------------------- filler


def test_absent_filled_pauses_are_flagged_as_deletion(lex):
    """Transkrip tanpa satu pun jeda terisi mencurigakan, bukan bagus."""
    a = audit([_chain(["saya", "mau", "bilang", "sesuatu", "hari", "ini"])], lex)
    assert a.fillers.filled_pause == 0
    assert a.fillers.suspected_deletion


def test_normal_filler_rate_is_not_flagged(lex):
    words = ["eee", "saya", "mau", "eee", "bilang", "sesuatu", "hari", "ini"]
    a = audit([_chain(words)], lex)
    assert a.fillers.filled_pause == 2
    assert a.fillers.fp_rate > LAJU_FILLER_LISAN_SPONTAN
    assert not a.fillers.suspected_deletion


def test_discourse_markers_alone_do_not_clear_the_deletion_flag(lex):
    """Penanda wacana kata sungguhan yang lolos ASR; jeda terisi yang
    dirapikan. Banyaknya 'kayak' tidak membuktikan 'eee' tidak dibuang."""
    words = ["kayak", "saya", "kayak", "mau", "kayak", "bilang", "gitu", "sih"]
    a = audit([_chain(words)], lex)
    assert a.fillers.discourse_marker >= 3
    assert a.fillers.filled_pause == 0
    assert a.fillers.suspected_deletion


def test_laju_jeda_terisi_bukan_laju_disfluensi(lex):
    """Dua besaran berbeda, dan tertukarnya pernah jadi ambang yang salah.

    5,97% adalah laju SELURUH disfluensi (jeda terisi + pengulangan +
    restart); 2,56% laju jeda terisi saja. Ambang penghapusan harus
    diturunkan dari yang kedua, sebab yang diukur `fp_rate` juga yang kedua.
    """
    assert LAJU_JEDA_TERISI_LISAN_SPONTAN < LAJU_DISFLUENSI_LISAN_SPONTAN
    assert LAJU_FILLER_LISAN_SPONTAN == LAJU_DISFLUENSI_LISAN_SPONTAN


def test_ambang_penghapusan_diturunkan_dari_laju_jeda_terisi(lex):
    """1,5% jeda terisi lolos; ambang lamanya (6%/3 = 2%) menjatuhkannya.

    Uji ini yang membedakan ambang baru dari yang lama: 1,5% ada persis di
    antara 1,28% dan 2%, jadi ia gagal kalau pembilangnya kembali ke laju
    disfluensi total.
    """
    words = ["eee" if i % 67 == 0 else "kata" for i in range(200)]
    a = audit([_chain(words)], lex)
    assert a.fillers.filled_pause == 3
    assert a.fillers.fp_rate == pytest.approx(0.015)
    assert not a.fillers.suspected_deletion

    # Setengah dari itu tetap tertangkap.
    words = ["eee" if i % 200 == 0 else "kata" for i in range(200)]
    b = audit([_chain(words)], lex)
    assert b.fillers.fp_rate == pytest.approx(0.005)
    assert b.fillers.suspected_deletion


def test_filler_breakdown_counts_each_form(lex):
    a = audit([_chain(["eee", "eee", "anu", "saya", "mau"])], lex)
    assert a.fillers.counts["eee"] == 2
    assert a.fillers.counts["anu"] == 1


# ------------------------------------------------- penanda ralat kuat vs ambigu


def test_bukan_as_plain_negation_is_not_counted_as_editing_term(lex):
    """'bukan' pengingkar paling lazim dalam Bahasa Indonesia; menghitung
    keanggotaan leksikonnya sebagai fungsi ralat membuat kalimat negasi
    biasa terhitung ralat. Kalimat di bawah diambil dari rekaman nyata."""
    utt = _chain(["jadi", "memang", "bukan", "setiap", "agama", "itu"])
    f = audit([utt], lex).fillers
    assert f.editing_term == 0
    assert f.editing_term_ambiguous == 1


def test_strong_editing_terms_are_counted_as_editing_terms(lex):
    f = audit([_chain(["saya", "ralat", "maksudnya", "begini"])], lex).fillers
    assert f.editing_term == 2
    assert f.editing_term_ambiguous == 0


def test_ambiguous_editing_terms_stay_out_of_the_total(lex):
    """Sejajar dengan `filled_pause_ambiguous`: bentuk yang belum diputus
    tidak boleh menggelembungkan laju filler yang dilaporkan."""
    polos = audit([_chain(["setiap", "agama", "itu"])], lex).fillers
    dengan = audit([_chain(["bukan", "setiap", "agama", "itu"])], lex).fillers
    assert dengan.editing_term_ambiguous == 1
    assert dengan.total == polos.total


def test_ambiguous_editing_term_still_appears_in_breakdown(lex):
    """Tidak dihitung bukan berarti disembunyikan -- rinciannya tetap ada
    supaya bisa diperiksa manual."""
    f = audit([_chain(["jadi", "bukan", "begitu"])], lex).fillers
    assert f.counts["bukan"] == 1


def test_report_separates_the_two_kinds_of_ambiguity(lex):
    text = format_audit(audit([_chain(["bukan", "setiap", "agama"])], lex))
    assert "Penanda ralat ambigu" in text
    assert "Jeda terisi ambigu" in text


def test_audit_dict_exposes_ambiguous_editing_terms(lex):
    d = audit_dict(audit([_chain(["jadi", "bukan", "begitu"])], lex))
    assert d["filler"]["penanda_ralat_ambigu"] == 1
    assert d["filler"]["penanda_ralat"] == 0


# ------------------------------------------------------------------ pengulangan


def test_neutral_gap_contributes_no_prosodic_evidence(lex):
    """Titik tengah zona abu-abu harus benar-benar berbobot nol.

    'xyzabc' tak dikenal leksikon, jadi satu-satunya bukti yang tersisa
    adalah kelas terbuka (+0,5). Skor yang persis 0,5 membuktikan jeda
    menyumbang nol -- dasar dipakainya jeda ini sebagai uji kontrafaktual.
    """
    _, score, _ = classify_adjacent_repeat("xyzabc", _neutral_gap(lex), lex)
    assert score == pytest.approx(0.5)


def test_prosody_dependent_decision_is_detected(lex):
    """'wabarakatuh wabarakatuh' bertahan sebagai reduplikasi hanya karena
    jeda rapat; cabut prosodinya dan putusan berbalik."""
    utt = _utt([("wabarakatuh", 0.0, 0.30), ("wabarakatuh", 0.32, 0.62)])
    r = audit([utt], lex).repeats
    assert r.events == 1
    assert r.decided_by_prosody_alone == 1
    assert r.prosody_dependent_share == pytest.approx(1.0)


def test_lexically_settled_decision_is_not_prosody_dependent(lex):
    """'saya saya' disfluensi karena kata fungsi; jeda tidak mengubahnya."""
    utt = _utt([("saya", 0.0, 0.30), ("saya", 0.31, 0.61)])
    r = audit([utt], lex).repeats
    assert r.events == 1
    assert r.disfluency == 1
    assert r.decided_by_prosody_alone == 0


def test_repeat_verdicts_sum_to_events(lex):
    words = ["saya", "saya", "mau", "anak", "anak", "pergi", "lari", "lari"]
    r = audit([_chain(words)], lex).repeats
    assert r.reduplication + r.disfluency + r.ambiguous == r.events


# ---------------------------------------------------------------------- laporan


def test_empty_corpus_does_not_divide_by_zero(lex):
    a = audit([_utt([("halo", 0.0, 0.3)])], lex)
    assert a.gaps.total == 0
    assert a.gaps.zero_share == 0.0
    assert a.gaps.median == 0.0
    format_audit(a)  # tidak boleh melempar


def test_report_names_the_remedy_when_timings_collapse(lex):
    utt = _utt([("satu", 0.0, 0.4), ("dua", 0.4, 0.8)])
    text = format_audit(audit([utt], lex))
    assert "RUNTUH" in text
    assert "forced alignment" in text


def test_report_is_quiet_when_nothing_is_wrong(lex):
    words = ["eee", "saya", "mau", "eee", "bilang", "sesuatu", "hari", "ini"]
    text = format_audit(audit([_chain(words, gap=0.15)], lex))
    assert "RUNTUH" not in text
    assert "DIDUGA DIHAPUS" not in text


def test_audit_dict_is_json_serialisable(lex):
    import json

    d = audit_dict(audit([_chain(["eee", "saya", "saya", "mau"])], lex))
    json.dumps(d, ensure_ascii=False)
    assert d["penanda_waktu"]["prosodi_layak_pakai"] in (True, False)
    assert d["filler"]["diduga_dihapus_asr"] in (True, False)


def test_duration_sums_speech_not_silence(lex):
    a = audit(
        [
            _utt([("a", 0.0, 1.0), ("b", 1.0, 2.0)], uid="u1"),
            _utt([("c", 60.0, 61.0)], uid="u2"),
        ],
        lex,
    )
    #: 2,0 s + 1,0 s bicara. Senyap 58 detik antar-ujaran tidak ikut terhitung;
    #: tanpa itu durasinya akan terbaca 61 detik.
    assert a.duration == pytest.approx(3.0)


def test_run_length_survives_the_counterfactual(lex):
    """Rentetan tiga kata dinilai dengan penalti rentetan; uji tanpa-prosodi
    harus mempertahankan penalti itu, bukan diam-diam menurunkannya ke dua.
    Kalau tidak, selisihnya bukan lagi murni akibat prosodi."""
    utt = _utt(
        [("saya", 0.00, 0.30), ("saya", 0.31, 0.61), ("saya", 0.62, 0.92)]
    )
    events = audit([utt], lex).repeats
    assert events.events == 2
    assert events.disfluency == 2
    #: Kata fungsi + rentetan tiga sudah memutuskan perkara tanpa prosodi.
    assert events.decided_by_prosody_alone == 0
