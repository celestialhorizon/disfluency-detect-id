"""Uji penyusunan Edit Decision List."""

import json

import pytest

from disfluency_id.edl import EDLConfig, build_edl, timecode, to_ffmpeg, write_outputs
from disfluency_id.schema import AGGRESSIVE_CUT, CONSERVATIVE_CUT, DM, FP, O, REP


@pytest.fixture
def tagged(make_utt):
    """Ujaran dengan disfluensi di awal, tengah, dan akhir."""
    utt = make_utt(
        [
            ("eee", FP, 0.25),
            ("jadi", DM, 0.15),
            ("saya", REP, 0.08),
            ("saya", O, 0.30),
            ("mau", O, 0.05),
            ("bertanya", O, 0.05),
        ]
    )
    utt.assign([FP, DM, REP, O, O, O])
    return utt


# --------------------------------------------------------------------------
# Timecode
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, "00:00:00.000"),
        (1.5, "00:00:01.500"),
        (61.25, "00:01:01.250"),
        (3661.001, "01:01:01.001"),
        (-5.0, "00:00:00.000"),
    ],
)
def test_timecode(seconds, expected):
    assert timecode(seconds) == expected


# --------------------------------------------------------------------------
# Penyusunan potongan
# --------------------------------------------------------------------------


def test_cuts_are_sorted_and_disjoint(tagged):
    edl = build_edl([tagged])
    for a, b in zip(edl.cuts, edl.cuts[1:]):
        assert a.end <= b.start


def test_keeps_and_cuts_tile_the_timeline(tagged):
    edl = build_edl([tagged])
    covered = sum(b - a for a, b in edl.keeps) + edl.removed
    assert covered == pytest.approx(edl.duration, abs=1e-6)


def test_cuts_never_touch_a_kept_word(tagged):
    """Memotong onset kata yang dipertahankan jauh lebih terdengar
    daripada menyisakan senyap, jadi jarak aman wajib dihormati."""
    cfg = EDLConfig()
    edl = build_edl([tagged], cfg)
    kept = [t for t in tagged.tokens if t.pred not in cfg.cut_labels]
    for cut in edl.cuts:
        for tok in kept:
            assert cut.end <= tok.start or cut.start >= tok.end


def test_conservative_cut_keeps_discourse_markers(tagged):
    edl = build_edl([tagged], EDLConfig(cut_labels=CONSERVATIVE_CUT))
    assert "jadi" in edl.transcript_after
    assert "eee" not in edl.transcript_after


def test_aggressive_cut_removes_discourse_markers(tagged):
    edl = build_edl([tagged], EDLConfig(cut_labels=AGGRESSIVE_CUT))
    assert "jadi" not in edl.transcript_after


def test_transcript_after_drops_only_the_disfluent_copy(tagged):
    edl = build_edl([tagged], EDLConfig(cut_labels=CONSERVATIVE_CUT))
    assert edl.transcript_after == "jadi saya mau bertanya"


def test_reduction_is_positive_and_bounded(tagged):
    edl = build_edl([tagged])
    assert 0 < edl.reduction < 1
    assert edl.kept + edl.removed == pytest.approx(edl.duration, abs=1e-6)


def test_nothing_cut_when_no_disfluency(make_utt):
    utt = make_utt([("kami", O, 0.05), ("menunda", O, 0.05), ("peluncuran", O, 0.05)])
    utt.assign([O, O, O])
    edl = build_edl([utt])
    assert edl.cuts == []
    assert len(edl.keeps) == 1
    assert edl.reduction == 0.0


def test_empty_input_is_handled():
    edl = build_edl([])
    assert edl.cuts == []
    assert edl.duration == 0.0


def test_tiny_cuts_are_dropped(make_utt):
    """Potongan di bawah ambang tidak sepadan dengan risiko sambungan pecah."""
    utt = make_utt([("a", O, 0.0), ("b", O, 0.0), ("c", O, 0.0)], dur=0.02)
    utt.assign([O, FP, O])
    edl = build_edl([utt], EDLConfig(min_cut=0.5))
    assert edl.cuts == []


def test_adjacent_cuts_are_merged(make_utt):
    utt = make_utt(
        [("eee", O, 0.20), ("emm", O, 0.02), ("saya", O, 0.20), ("setuju", O, 0.05)]
    )
    utt.assign([FP, FP, O, O])
    edl = build_edl([utt])
    assert len(edl.cuts) == 1
    assert "eee" in edl.cuts[0].text and "emm" in edl.cuts[0].text


def test_multiple_utterances_share_one_timeline(make_utt):
    a = make_utt([("eee", O, 0.20), ("halo", O, 0.10)], uid="u1")
    a.assign([FP, O])
    b = make_utt([("emm", O, 0.20), ("dunia", O, 0.10)], uid="u2")
    for tok in b.tokens:
        tok.start += 10.0
        tok.end += 10.0
    b.assign([FP, O])

    edl = build_edl([a, b])
    assert len(edl.cuts) == 2
    assert edl.cuts[0].end < edl.cuts[1].start


# --------------------------------------------------------------------------
# Ekspor
# --------------------------------------------------------------------------


def test_ffmpeg_command_concatenates_every_kept_segment(tagged):
    edl = build_edl([tagged])
    cmd = to_ffmpeg(edl, "masukan.mp4", "keluaran.mp4")
    assert f"concat=n={len(edl.keeps)}" in cmd
    assert cmd.count("atrim=") == len(edl.keeps)
    assert "masukan.mp4" in cmd and "keluaran.mp4" in cmd


def test_ffmpeg_handles_everything_cut(make_utt):
    utt = make_utt([("eee", O, 0.20)])
    utt.assign([FP])
    edl = build_edl([utt])
    assert to_ffmpeg(edl).startswith("#")


def test_write_outputs_creates_every_artifact(tagged, tmp_path):
    edl = build_edl([tagged])
    paths = write_outputs(edl, tmp_path, stem="uji")

    assert set(paths) == {"json", "csv", "ffmpeg", "transkrip"}
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["statistik"]["jumlah_potongan"] == len(edl.cuts)
    assert len(payload["potongan"]) == len(edl.cuts)

    csv_lines = paths["csv"].read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == len(edl.cuts) + 1


def test_json_export_records_the_reason_for_each_cut(tagged, tmp_path):
    """Keputusan potong harus dapat ditelusuri, bukan sekadar rentang waktu."""
    edl = build_edl([tagged])
    payload = json.loads(
        write_outputs(edl, tmp_path)["json"].read_text(encoding="utf-8")
    )
    for cut in payload["potongan"]:
        assert cut["alasan"]
        assert cut["label"]
