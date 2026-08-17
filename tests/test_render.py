"""Uji eksekusi EDL menjadi berkas video.

Seluruhnya berjalan tanpa ffmpeg dan tanpa berkas video: yang diuji adalah
aritmetika span dan perintah yang disusun. Itu memang batas modulnya -- yang
bisa salah diam-diam adalah titik potongnya, bukan ffmpeg-nya.
"""

import json

import pytest

from disfluency_id.render import (
    RenderConfig,
    Span,
    ffmpeg_argv,
    extend_to_media,
    filter_complex,
    keep_spans,
    load_edl,
    proof_spans,
    resolve_source,
    timeline_bounds,
)


@pytest.fixture
def payload():
    """EDL kecil: dua potongan pada garis waktu yang mulai di detik 1,0."""
    return {
        "sumber": "video_20260118_163813.mp4",
        "statistik": {"durasi_asli_detik": 29.0},
        "potongan": [
            {"mulai": 5.0, "selesai": 5.5, "teks": "eee"},
            {"mulai": 20.0, "selesai": 21.0, "teks": "saya saya"},
        ],
        "segmen_disimpan": [
            {"mulai": 1.0, "selesai": 5.0},
            {"mulai": 5.5, "selesai": 20.0},
            {"mulai": 21.0, "selesai": 30.0},
        ],
    }


# --------------------------------------------------------------------------
# Span
# --------------------------------------------------------------------------


def test_timeline_bounds_bukan_nol_sampai_durasi(payload):
    """Garis waktu mulai pada kata pertama, bukan pada detik nol."""
    assert timeline_bounds(payload) == (1.0, 30.0)


def test_timeline_bounds_edl_kosong():
    assert timeline_bounds({}) == (0.0, 0.0)


def test_keep_spans_menyalin_segmen_disimpan(payload):
    spans = keep_spans(payload)
    assert [(s.start, s.end) for s in spans] == [(1.0, 5.0), (5.5, 20.0), (21.0, 30.0)]


def test_keep_spans_membuang_segmen_nol(payload):
    payload["segmen_disimpan"].append({"mulai": 30.0, "selesai": 30.0})
    assert len(keep_spans(payload)) == 3


def test_bukti_menghasilkan_sebelum_lalu_sesudah(payload):
    spans = proof_spans(payload, pad=2.0)
    assert [s.label for s in spans] == ["SEBELUM", "SESUDAH", "SESUDAH"] * 2


def test_bukti_sebelum_memuat_potongan_dan_sesudah_tidak(payload):
    sebelum, kiri, kanan = proof_spans(payload, pad=2.0)[:3]
    assert sebelum.start <= 5.0 and sebelum.end >= 5.5
    assert kiri.end == 5.0
    assert kanan.start == 5.5


def test_bukti_dijepit_ujung_garis_waktu():
    """Jendela tidak boleh menyeberang ke luar rekaman."""
    p = {
        "potongan": [{"mulai": 1.2, "selesai": 1.5}],
        "segmen_disimpan": [{"mulai": 1.0, "selesai": 1.2}, {"mulai": 1.5, "selesai": 2.0}],
    }
    spans = proof_spans(p, pad=10.0)
    assert min(s.start for s in spans) == 1.0
    assert max(s.end for s in spans) == 2.0


def test_bukti_tidak_menyeberang_ke_potongan_tetangga():
    """Cuplikan SESUDAH yang memuat disfluensi lain akan membantah dirinya sendiri."""
    p = {
        "potongan": [{"mulai": 10.0, "selesai": 10.4}, {"mulai": 11.0, "selesai": 11.5}],
        "segmen_disimpan": [{"mulai": 0.0, "selesai": 10.0}, {"mulai": 11.5, "selesai": 30.0}],
    }
    spans = proof_spans(p, pad=5.0)
    assert max(s.end for s in spans[:3]) <= 11.0
    assert min(s.start for s in spans[3:]) >= 10.4


def test_bukti_menghormati_max_cuts(payload):
    assert len(proof_spans(payload, max_cuts=1)) == 3


def test_bukti_edl_tanpa_potongan():
    assert proof_spans({"potongan": [], "segmen_disimpan": [{"mulai": 0.0, "selesai": 5.0}]}) == []


# --------------------------------------------------------------------------
# Perintah ffmpeg
# --------------------------------------------------------------------------


def test_filter_menyambung_setiap_span(payload):
    spans = keep_spans(payload)
    filt = filter_complex(spans)
    assert f"concat=n={len(spans)}" in filt
    assert filt.count("[0:v]trim=start=") == len(spans)   # "trim=" juga cocok dengan "atrim="
    assert filt.count("atrim=start=") == len(spans)


def test_filter_menolak_span_kosong():
    with pytest.raises(ValueError):
        filter_complex([])


def test_label_hanya_muncul_bila_ada_fonta():
    span = [Span(0.0, 1.0, "SEBELUM")]
    assert "drawtext" not in filter_complex(span, RenderConfig(font=None))
    assert "drawtext" in filter_complex(span, RenderConfig(font="/x/font.ttf"))


def test_skala_dan_fps_hanya_bila_diminta():
    span = [Span(0.0, 1.0)]
    polos = filter_complex(span, RenderConfig())
    assert "scale=" not in polos and "fps=" not in polos
    kecil = filter_complex(span, RenderConfig(width=640, height=360, fps=30))
    assert "scale=640:360" in kecil and "fps=30" in kecil


def test_argv_memakai_nama_berkas_sungguhan(payload, tmp_path):
    """Regresi: naskah .sh bawaan menulis singgahan `input.mp4`."""
    src = tmp_path / "video_20260118_163813.mp4"
    argv = ffmpeg_argv(src, tmp_path / "hasil.mp4", keep_spans(payload))
    assert str(src) in argv
    assert "input.mp4" not in argv
    assert argv[argv.index("-map") + 1] == "[v]"


def test_argv_tidak_pernah_lewat_shell(payload, tmp_path):
    argv = ffmpeg_argv("a.mp4", "b.mp4", keep_spans(payload))
    assert isinstance(argv, list) and argv[0] == "ffmpeg"
    assert all(isinstance(x, str) for x in argv)


def test_resolve_source_dari_kunci_sumber(payload, tmp_path):
    assert resolve_source(payload, tmp_path).name == "video_20260118_163813.mp4"


def test_load_edl_membaca_json(payload, tmp_path):
    p = tmp_path / "edl.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert load_edl(p)["sumber"] == payload["sumber"]


def test_perentangan_mengembalikan_senyap_ujung(payload):
    """Membuang disfluensi tidak boleh sekalian membuang pembuka dan penutup."""
    spans = extend_to_media(keep_spans(payload), media_duration=31.5)
    assert spans[0].start == 0.0
    assert spans[-1].end == 31.5


def test_perentangan_tidak_memendekkan(payload):
    spans = extend_to_media(keep_spans(payload), media_duration=10.0)
    assert spans[-1].end == 30.0


def test_perentangan_tanpa_durasi_media(payload):
    spans = extend_to_media(keep_spans(payload))
    assert spans[0].start == 0.0 and spans[-1].end == 30.0


def test_perentangan_span_kosong():
    assert extend_to_media([]) == []
