"""Uji bagi tahap-tahap transkripsi yang tidak menuntut model.

Empat fungsi di tengah rangkaian ASR — `cari_loop`, `bikin_segmen`,
`profil_senyap`, `kikis_ke_bunyi` — memakai pustaka standar saja, sehingga
dapat diuji tanpa GPU maupun bobot model. Justru di situ letak logika yang
bisa salah diam-diam; pemanggilan modelnya sendiri tidak.
"""

import array
import math
import wave

import pytest

from disfluency_id.asr import (bikin_segmen, cari_loop, ke_wav,
                               kikis_ke_bunyi, profil_senyap)


def w(teks, a, b):
    return {"text": teks, "timestamp": (a, b)}


# --------------------------------------------------------------------------
# cari_loop
# --------------------------------------------------------------------------

def test_pengulangan_sah_dua_putaran_tidak_dianggap_degenerasi():
    """`lu lu` dan `tiba tiba` adalah DATA, bukan sampah bangkitan.

    Ini uji terpenting di berkas ini: kalau ambangnya turun ke 2, detektor
    degenerasi akan memakan objek penelitiannya sendiri.
    """
    kata = [w("lu", 0.0, 0.2), w("lu", 0.2, 0.4), w("ikutin", 0.4, 0.8)]
    assert cari_loop(kata) == []

    kata = [w("tiba", 0.0, 0.2), w("tiba", 0.2, 0.4), w("dipenjara", 0.4, 0.9)]
    assert cari_loop(kata) == []


def test_tiga_putaran_masih_lolos_empat_tertangkap():
    tiga = [w("k", i * 0.1, i * 0.1 + 0.1) for i in range(3)]
    assert cari_loop(tiga) == []

    empat = [w("k", i * 0.1, i * 0.1 + 0.1) for i in range(4)]
    assert len(cari_loop(empat)) == 1


def test_rentetan_panjang_dilaporkan_satu_kali_bukan_empat():
    """Tanpa peleburan, `k k k k ...` cocok untuk n=1,2,3,4 sekaligus.

    Kutu itu pernah membuat 2,9 dtk audio rusak terlaporkan sebagai 11,4 dtk.
    """
    kata = [w("k", i * 0.1, i * 0.1 + 0.1) for i in range(26)]
    hasil = cari_loop(kata)
    assert len(hasil) == 1
    assert hasil[0]["mulai_idx"] == 0
    assert hasil[0]["henti_idx"] == 26


def test_loop_dua_kata_tertangkap():
    """Detektor satu-kata pernah meloloskan `ya kan. ya kan. ya kan.`"""
    kata = []
    for i in range(6):
        kata.append(w("ya", i * 0.4, i * 0.4 + 0.2))
        kata.append(w("kan.", i * 0.4 + 0.2, i * 0.4 + 0.4))
    hasil = cari_loop(kata)
    assert len(hasil) == 1
    assert hasil[0]["n"] == 2
    assert hasil[0]["isi"] == "ya kan."


def test_perbandingannya_mengabaikan_huruf_besar_dan_spasi():
    kata = [w(" K", 0.0, 0.1), w("k ", 0.1, 0.2),
            w("K", 0.2, 0.3), w(" k ", 0.3, 0.4)]
    assert len(cari_loop(kata)) == 1


def test_daftar_kosong_aman():
    assert cari_loop([]) == []
    assert bikin_segmen([]) == []


# --------------------------------------------------------------------------
# bikin_segmen
# --------------------------------------------------------------------------

def test_segmen_dipotong_di_jeda_ambang():
    kata = [w("satu", 0.0, 0.5), w("dua", 0.5, 1.0),
            w("tiga", 1.8, 2.2)]          # jeda 0,8 dtk -- tepat di ambang
    seg = bikin_segmen(kata, jeda_pisah=0.8)
    assert len(seg) == 2
    assert seg[0]["start"] == 0.0 and seg[0]["end"] == 1.0
    assert seg[1]["start"] == 1.8 and seg[1]["end"] == 2.2


def test_jeda_di_bawah_ambang_tidak_memotong():
    kata = [w("satu", 0.0, 0.5), w("dua", 1.2, 1.6)]   # jeda 0,7 dtk
    assert len(bikin_segmen(kata, jeda_pisah=0.8)) == 1


def test_ambang_bawaan_sama_dengan_max_gap_ingest():
    """Kalau keduanya berbeda, transkrip dan korpus akan bercerita hal
    berlainan tentang berkas yang sama."""
    import inspect

    from disfluency_id.ingest import from_whisper_json
    bawaan_segmen = inspect.signature(bikin_segmen).parameters["jeda_pisah"].default
    bawaan_ingest = inspect.signature(from_whisper_json).parameters["max_gap"].default
    assert bawaan_segmen == bawaan_ingest


def test_setiap_kata_masuk_tepat_satu_segmen():
    kata = [w("a", 0.0, 0.2), w("b", 1.5, 1.7), w("c", 1.7, 1.9), w("d", 9.0, 9.2)]
    seg = bikin_segmen(kata)
    assert sum(len(s["words"]) for s in seg) == len(kata)
    assert [x["word"] for s in seg for x in s["words"]] == ["a", "b", "c", "d"]


# --------------------------------------------------------------------------
# kikis_ke_bunyi
# --------------------------------------------------------------------------

def test_kikisan_tidak_pernah_menabrak_kata_berikutnya():
    """Kutu tumpang-tindih satu bingkai.

    Versi pertama menulis akhir kata sebagai (q+1)*hop. Karena kata
    bersebelahan berbagi bingkai batas, kata ke-i menabrak awal kata ke-i+1
    sebanyak 0,01 dtk -- dan pemeriksa yang menjepit jeda negatif ke nol
    melaporkannya sebagai "jeda persis nol", persis gejala yang mau
    diperbaiki.
    """
    hop = 0.01
    senyap = [False] * 100
    kata = [w("a", 0.00, 0.30), w("b", 0.30, 0.60), w("c", 0.60, 0.90)]
    hasil, _ = kikis_ke_bunyi(kata, senyap, hop)
    for kini, lanjut in zip(hasil, hasil[1:]):
        assert kini["timestamp"][1] <= lanjut["timestamp"][0] + 1e-9


def test_senyap_di_ujung_dikikis():
    hop = 0.01
    senyap = [True] * 10 + [False] * 20 + [True] * 10   # bunyi di 0,10-0,30
    kata = [w("a", 0.0, 0.40)]
    hasil, dikikis = kikis_ke_bunyi(kata, senyap, hop)
    a, b = hasil[0]["timestamp"]
    assert a == pytest.approx(0.10, abs=0.011)
    assert b == pytest.approx(0.30, abs=0.011)
    assert dikikis == 1


def test_kata_yang_seluruhnya_senyap_dibiarkan_apa_adanya():
    """Mengikisnya sampai nol menghasilkan token tanpa durasi, dan itu lebih
    menyesatkan daripada penanda waktu yang longgar."""
    hop = 0.01
    kata = [w("a", 0.0, 0.20)]
    hasil, dikikis = kikis_ke_bunyi(kata, [True] * 50, hop)
    assert hasil[0]["timestamp"] == (0.0, 0.20)
    assert dikikis == 0


def test_profil_senyap_kosong_tidak_mengubah_apa_pun():
    kata = [w("a", 0.0, 0.2), w("b", 0.2, 0.4)]
    hasil, dikikis = kikis_ke_bunyi(kata, [], 0.01)
    assert [x["timestamp"] for x in hasil] == [(0.0, 0.2), (0.2, 0.4)]
    assert dikikis == 0


# --------------------------------------------------------------------------
# profil_senyap
# --------------------------------------------------------------------------

def tulis_wav(path, sr, contoh):
    with wave.open(str(path), "wb") as w_:
        w_.setnchannels(1)
        w_.setsampwidth(2)
        w_.setframerate(sr)
        w_.writeframes(array.array("h", contoh).tobytes())


def test_senyap_dan_bunyi_terbedakan(tmp_path):
    sr = 16000
    diam = [0] * sr                                    # 1 dtk senyap
    nada = [int(12000 * math.sin(2 * math.pi * 220 * i / sr)) for i in range(sr)]
    p = tmp_path / "uji.wav"
    tulis_wav(p, sr, diam + nada)

    senyap, hop = profil_senyap(p)
    assert hop == 0.01
    n = len(senyap)
    # Paruh pertama senyap, paruh kedua tidak. Tepi di tengah dilonggarkan.
    assert all(senyap[: n // 2 - 5])
    assert not any(senyap[n // 2 + 5:])


def test_wav_stereo_ditolak_dengan_pesan_yang_jelas(tmp_path):
    p = tmp_path / "stereo.wav"
    with wave.open(str(p), "wb") as w_:
        w_.setnchannels(2)
        w_.setsampwidth(2)
        w_.setframerate(16000)
        w_.writeframes(array.array("h", [0] * 3200).tobytes())
    with pytest.raises(ValueError, match="16-bit mono"):
        profil_senyap(p)


def test_wav_terlalu_pendek_tidak_meledak(tmp_path):
    p = tmp_path / "pendek.wav"
    tulis_wav(p, 16000, [0] * 10)
    senyap, _ = profil_senyap(p)
    assert senyap == []


# --------------------------------------------------------------------------
# ke_wav
# --------------------------------------------------------------------------

def test_ke_wav_melewati_ffmpeg_bila_hasilnya_sudah_ada(tmp_path, monkeypatch):
    """Transkripsi diulang berkali-kali; ekstraksi ulang audio yang sama itu
    pemborosan yang nyata pada berkas panjang."""
    import subprocess as sp

    sumber = tmp_path / "rekaman.mp4"
    sumber.write_bytes(b"bukan mp4 sungguhan")
    (tmp_path / "rekaman.16k.wav").write_bytes(b"")

    def jangan_dipanggil(*a, **k):
        raise AssertionError("ffmpeg dipanggil padahal WAV-nya sudah ada")

    monkeypatch.setattr(sp, "run", jangan_dipanggil)
    assert ke_wav(sumber, tmp_path).name == "rekaman.16k.wav"
