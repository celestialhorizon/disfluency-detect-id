"""Uji registri media: pembacaan provenans dan verifikasi panjang unduhan.

Seluruh uji di sini berjalan tanpa jaringan. `unduh` menerima `pembuka` yang
dapat diganti persis supaya jalur verifikasinya — bagian yang bisa salah
diam-diam — teruji tanpa mengunduh apa pun.
"""

import pytest

from disfluency_id.media import (
    BELUM,
    KOLOM_WAJIB,
    Sumber,
    _ganti_nama,
    baca_provenans,
    perlu_unduh,
    ringkas_etika,
    total_durasi,
    unduh,
    unduh_semua,
)

KEPALA = ",".join(KOLOM_WAJIB)


def _tulis(tmp_path, *baris, nama="provenans.csv"):
    path = tmp_path / nama
    path.write_text("\n".join([KEPALA, *baris]) + "\n", encoding="utf-8")
    return path


def _baris(berkas="a.mp4", penutur="Andi", tautan="https://asal/a",
           unduh="https://x/a.mp4", lisensi="CC-BY", terbit="2026-01-02",
           tanggal="2026-08-16", durasi="57.8", asli="245", jendela="30",
           izin="ya", catatan="-"):
    return (f"{berkas},Judul,{penutur},{tautan},{unduh},{lisensi},{terbit},"
            f"{tanggal},{durasi},{asli},{jendela},{izin},{catatan}")


# --------------------------------------------------------------------------
# Pembacaan provenans
# --------------------------------------------------------------------------


def test_membaca_baris_lengkap(tmp_path):
    daftar = baca_provenans(_tulis(tmp_path, _baris()))
    assert len(daftar) == 1
    s = daftar[0]
    assert s.berkas == "a.mp4"
    assert s.durasi_detik == pytest.approx(57.8)
    assert s.siap_dikutip


def test_baris_kosong_di_ujung_diabaikan(tmp_path):
    """CSV yang disunting tangan hampir selalu berakhir dengan baris kosong."""
    path = tmp_path / "p.csv"
    path.write_text(KEPALA + "\n" + _baris() + "\n\n\n", encoding="utf-8")
    assert len(baca_provenans(path)) == 1


def test_durasi_belum_dicatat_jadi_none_bukan_gagal(tmp_path):
    daftar = baca_provenans(_tulis(tmp_path, _baris(durasi=BELUM)))
    assert daftar[0].durasi_detik is None


def test_durasi_bukan_angka_menyebut_nomor_barisnya(tmp_path):
    with pytest.raises(ValueError, match="baris 2"):
        baca_provenans(_tulis(tmp_path, _baris(durasi="satu menit")))


def test_kolom_kurang_disebut_namanya(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("berkas,judul\na.mp4,Judul\n", encoding="utf-8")
    with pytest.raises(ValueError, match="izin"):
        baca_provenans(path)


def test_berkas_ganda_ditolak(tmp_path):
    """Dua baris untuk satu rekaman berarti salah satunya akan terabaikan diam-diam."""
    path = _tulis(tmp_path, _baris(berkas="a.mp4"), _baris(berkas="a.mp4"))
    with pytest.raises(ValueError, match="ganda"):
        baca_provenans(path)


def test_provenans_hilang_menyebut_lintasannya(tmp_path):
    with pytest.raises(FileNotFoundError, match="tidak ada.csv"):
        baca_provenans(tmp_path / "tidak ada.csv")


# --------------------------------------------------------------------------
# Medan etika
# --------------------------------------------------------------------------


def test_medan_etika_kosong_terdaftar_bukan_menggagalkan(tmp_path):
    """Provenans belum lengkap harus terlihat, tetapi tidak menghentikan kerja.

    Menghentikannya menggoda orang mengisi asal supaya jalan, dan provenans
    yang diisi asal lebih berbahaya daripada yang jujur kosong.
    """
    daftar = baca_provenans(
        _tulis(tmp_path, _baris(penutur=BELUM, izin=BELUM))
    )
    s = daftar[0]
    assert s.belum_dicatat == ["penutur", "izin"]
    assert not s.siap_dikutip
    assert "penutur, izin" in ringkas_etika(daftar)[0]


def test_medan_etika_kosong_melompong_juga_terhitung(tmp_path):
    daftar = baca_provenans(_tulis(tmp_path, _baris(izin="")))
    assert daftar[0].belum_dicatat == ["izin"]


def test_total_durasi_menghitung_yang_belum_diisi(tmp_path):
    daftar = baca_provenans(
        _tulis(tmp_path, _baris(berkas="a.mp4", durasi="10.5"),
               _baris(berkas="b.mp4", durasi=BELUM))
    )
    total, kosong = total_durasi(daftar)
    assert total == pytest.approx(10.5)
    assert kosong == 1


# --------------------------------------------------------------------------
# Verifikasi panjang unduhan
# --------------------------------------------------------------------------


class _Resp:
    """Tiruan respons HTTP; `kirim` boleh lebih pendek dari Content-Length."""

    def __init__(self, isi, panjang=None, kirim=None):
        self._sisa = kirim if kirim is not None else isi
        self.headers = {} if panjang is None else {"Content-Length": str(panjang)}

    def read(self, n):
        potongan, self._sisa = self._sisa[:n], self._sisa[n:]
        return potongan

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _pembuka(isi, panjang=None, kirim=None):
    def buka(url, timeout=None):
        return _Resp(isi, panjang if panjang is not None else len(isi), kirim)
    return buka


def test_unduh_menulis_berkas_dan_melaporkan_terunduh(tmp_path):
    isi = b"x" * 5000
    path, status = unduh("https://x/a.mp4", tmp_path / "a.mp4",
                         pembuka=_pembuka(isi), blok=1024)
    assert status == "terunduh"
    assert path.read_bytes() == isi


def test_berkas_pendek_ditolak_walau_ia_ada(tmp_path):
    """Inti modul ini: unduhan putus meninggalkan berkas yang ADA tapi pendek.

    Pemeriksaan `exists()` akan menerimanya sebagai lengkap, lalu ASR
    mentranskripsikan potongan tanpa memberi tahu bahwa rekamannya terpotong.
    """
    tujuan = tmp_path / "a.mp4"
    with pytest.raises(IOError, match="terpotong"):
        unduh("https://x/a.mp4", tujuan,
              pembuka=_pembuka(b"x" * 5000, panjang=5000, kirim=b"x" * 1200),
              blok=1024)
    assert not tujuan.exists(), "berkas utuh tidak boleh muncul dari unduhan gagal"


def test_sisa_unduhan_gagal_disimpan_sebagai_part(tmp_path):
    tujuan = tmp_path / "a.mp4"
    with pytest.raises(IOError):
        unduh("https://x/a.mp4", tujuan,
              pembuka=_pembuka(b"x" * 5000, panjang=5000, kirim=b"x" * 1200),
              blok=1024)
    assert (tmp_path / "a.mp4.part").stat().st_size == 1200


def test_berkas_lengkap_tidak_diunduh_ulang(tmp_path):
    tujuan = tmp_path / "a.mp4"
    tujuan.write_bytes(b"x" * 5000)
    _, status = unduh("https://x/a.mp4", tujuan, pembuka=_pembuka(b"y" * 5000))
    assert status == "ada"
    assert tujuan.read_bytes() == b"x" * 5000, "berkas lengkap tidak boleh ditimpa"


def test_berkas_lokal_pendek_diunduh_ulang(tmp_path):
    """Panjang yang tidak cocok berarti sisa unduhan lama, bukan berkas sah."""
    tujuan = tmp_path / "a.mp4"
    tujuan.write_bytes(b"x" * 1200)
    _, status = unduh("https://x/a.mp4", tujuan,
                      pembuka=_pembuka(b"y" * 5000), blok=1024)
    assert status == "terunduh"
    assert tujuan.read_bytes() == b"y" * 5000


def test_tanpa_content_length_dilaporkan_bukan_disembunyikan(tmp_path):
    def buka(url, timeout=None):
        return _Resp(b"x" * 300, panjang=None)

    _, status = unduh("https://x/a.mp4", tmp_path / "a.mp4", pembuka=buka)
    assert status == "terunduh-tanpa-verifikasi"


def test_kunci_sesaat_windows_diulang_bukan_digagalkan(tmp_path, monkeypatch):
    """Kutu nyata, ketahuan saat mengunduh disfluency_5.mp4 di Windows.

    Pemindai virus membuka berkas yang baru selesai ditulis, dan selama itu
    `os.replace` gagal WinError 32 walau program ini sudah menutupnya. Uji
    ini memalsukan dua kegagalan beruntun lalu memastikan namanya tetap
    berganti.
    """
    from pathlib import Path as P

    asli = P.replace
    sisa_gagal = {"n": 2}

    def replace_bergoyang(self, target):
        if sisa_gagal["n"] > 0:
            sisa_gagal["n"] -= 1
            raise PermissionError(32, "sedang dipakai proses lain")
        return asli(self, target)

    monkeypatch.setattr(P, "replace", replace_bergoyang)

    sementara = tmp_path / "a.mp4.part"
    sementara.write_bytes(b"x" * 10)
    _ganti_nama(sementara, tmp_path / "a.mp4", percobaan=5, jeda=0.0)

    assert (tmp_path / "a.mp4").read_bytes() == b"x" * 10
    assert sisa_gagal["n"] == 0


def test_kunci_yang_tak_kunjung_lepas_tetap_dilaporkan(tmp_path, monkeypatch):
    """Mengulang selamanya akan menyembunyikan kunci yang sesungguhnya macet."""
    from pathlib import Path as P

    def selalu_gagal(self, target):
        raise PermissionError(32, "sedang dipakai proses lain")

    monkeypatch.setattr(P, "replace", selalu_gagal)

    sementara = tmp_path / "a.mp4.part"
    sementara.write_bytes(b"x" * 10)
    with pytest.raises(PermissionError):
        _ganti_nama(sementara, tmp_path / "a.mp4", percobaan=3, jeda=0.0)
    assert sementara.exists(), ".part harus utuh supaya jalankan ulang menemukannya"


def test_perlu_unduh_tanpa_panjang_menerima_berkas_yang_ada(tmp_path):
    tujuan = tmp_path / "a.mp4"
    tujuan.write_bytes(b"x")
    assert perlu_unduh(tujuan, None) is False
    assert perlu_unduh(tmp_path / "belum ada.mp4", None) is True


# --------------------------------------------------------------------------
# unduh_semua
# --------------------------------------------------------------------------


def test_hanya_menerima_berkas_yang_punya_baris_provenans(tmp_path):
    """Syarat inti: rekaman tanpa provenans tidak bisa diunduh sama sekali."""
    path = _tulis(tmp_path, _baris(berkas="a.mp4"))
    with pytest.raises(ValueError, match="b.mp4"):
        unduh_semua(path, tmp_path / "media", hanya=["b.mp4"],
                    pembuka=_pembuka(b"x" * 10), lapor=None)


def test_baris_tanpa_tautan_dilewati_bukan_digagalkan(tmp_path):
    """Rekaman yang disalin manual sah tidak punya tautan sama sekali."""
    path = _tulis(tmp_path, _baris(berkas="a.mp4", tautan="", unduh=""))
    hasil = unduh_semua(path, tmp_path / "media", pembuka=_pembuka(b"x"), lapor=None)
    assert [status for _, _, status in hasil] == ["tanpa-tautan"]


def test_unduhan_memakai_cermin_bukan_asal_kanonis(tmp_path):
    """Asal kanonis adalah halaman yang dikutip, bukan berkas yang bisa diambil.

    Halaman YouTube tidak mengembalikan mp4 bila diambil langsung, jadi
    memakai `tautan_sumber` sebagai alamat unduhan akan selalu keliru.
    """
    diminta = []

    def buka(url, timeout=None):
        diminta.append(url)
        return _Resp(b"x" * 20, panjang=20)

    path = _tulis(tmp_path, _baris(tautan="https://youtube.com/watch?v=abc",
                                   unduh="https://cdn/a.mp4"))
    unduh_semua(path, tmp_path / "media", pembuka=buka, lapor=None)
    assert diminta == ["https://cdn/a.mp4"]


def test_tanpa_cermin_jatuh_ke_asal_kanonis(tmp_path):
    """Berkas yang memang dapat diunduh langsung tidak butuh kolom cermin."""
    diminta = []

    def buka(url, timeout=None):
        diminta.append(url)
        return _Resp(b"x" * 20, panjang=20)

    path = _tulis(tmp_path, _baris(tautan="https://asal/a.mp4", unduh=""))
    unduh_semua(path, tmp_path / "media", pembuka=buka, lapor=None)
    assert diminta == ["https://asal/a.mp4"]


def test_unduh_semua_mengembalikan_status_tiap_baris(tmp_path):
    path = _tulis(tmp_path, _baris(berkas="a.mp4"), _baris(berkas="b.mp4"))
    hasil = unduh_semua(path, tmp_path / "media",
                        pembuka=_pembuka(b"x" * 40), lapor=None)
    assert [s.berkas for s, _, _ in hasil] == ["a.mp4", "b.mp4"]
    assert all(status == "terunduh" for _, _, status in hasil)
    assert all(p.exists() for _, p, _ in hasil)


# --------------------------------------------------------------------------
# Provenans yang sesungguhnya dipakai proyek ini
# --------------------------------------------------------------------------


def test_provenans_proyek_terbaca_dan_tiap_tautan_unik():
    from pathlib import Path

    akar = Path(__file__).resolve().parent.parent
    daftar = baca_provenans(akar / "data" / "media" / "provenans.csv")
    assert daftar, "provenans proyek tidak boleh kosong"

    tautan = [s.alamat_unduh for s in daftar if s.alamat_unduh]
    assert len(tautan) == len(set(tautan)), (
        "dua baris menunjuk tautan yang sama; salah satu berkas akan tertimpa"
    )
    for s in daftar:
        assert s.berkas.strip() == s.berkas
        assert not s.tautan_unduh or s.tautan_unduh.startswith("http")


def test_porsi_kutipan_dihitung_bukan_dinyatakan(tmp_path):
    daftar = baca_provenans(_tulis(tmp_path, _baris(durasi="60", asli="240")))
    assert daftar[0].porsi_kutipan == pytest.approx(0.25)


def test_porsi_kutipan_none_bila_durasi_asli_belum_diukur(tmp_path):
    daftar = baca_provenans(_tulis(tmp_path, _baris(asli=BELUM)))
    assert daftar[0].porsi_kutipan is None


def test_tiap_sumber_proyek_punya_url_kanonis_dan_porsi_terukur():
    """Bahan pihak ketiga menuntut sitasi yang dapat ditelusuri.

    Tanpa URL asli, lisensinya tidak dapat diperiksa siapa pun — termasuk
    penguji. Tanpa kedua durasi, porsi kutipan tidak dapat dihitung, padahal
    itulah ukuran yang menimbang penggunaan wajar.
    """
    from pathlib import Path

    akar = Path(__file__).resolve().parent.parent
    daftar = baca_provenans(akar / "data" / "media" / "provenans.csv")
    assert len(daftar) == 6

    for s in daftar:
        assert s.tautan_sumber.startswith("https://www.youtube.com/watch?v="), (
            f"{s.berkas}: tautan_sumber harus URL YouTube aslinya"
        )
        assert s.porsi_kutipan is not None, f"{s.berkas}: porsi kutipan tak terhitung"
        assert 0 < s.porsi_kutipan < 1, (
            f"{s.berkas}: potongan {s.porsi_kutipan:.1%} dari karya asli — "
            "porsi >= 100% berarti salinan utuh, bukan kutipan"
        )
        assert s.siap_dikutip, f"{s.berkas}: {s.belum_dicatat}"


def test_bahan_pihak_ketiga_tidak_boleh_mengaku_milik_peneliti():
    """Seluruh bahan proyek ini podcast YouTube, bukan rekaman peneliti.

    Satu baris pernah tercatat `lisensi = milik peneliti`, dan keliru di kolom
    ini bukan salah ketik: ia menghapus kewajiban lisensi dan izin penutur
    sekaligus.
    """
    from pathlib import Path

    akar = Path(__file__).resolve().parent.parent
    for s in baca_provenans(akar / "data" / "media" / "provenans.csv"):
        assert "milik peneliti" not in s.lisensi.lower(), (
            f"{s.berkas}: bahan pihak ketiga tidak boleh diberi lisensi "
            "'milik peneliti'"
        )
