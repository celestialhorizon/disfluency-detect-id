"""Registri media: `provenans.csv` sebagai satu-satunya daftar sumber.

Rekaman tidak pernah masuk repositori (lihat `.gitignore` dan
`data/media/CATATAN.md`). Yang masuk repositori adalah **daftarnya**, dan
daftar itu `data/media/provenans.csv`. Modul ini yang membacanya lalu
mengunduh berkas yang belum ada.

    provenans.csv  ->  baca_provenans  ->  unduh_semua  ->  data/media/*.mp4
                                                                |
                                                  ingest --audio, edl, render

Akibat rancangan ini yang paling penting bukan kemudahannya, melainkan
urutannya: **berkas yang tidak punya baris provenans tidak akan terunduh.**
`data/media/CATATAN.md` mensyaratkan provenans diisi sebelum berkas diproses,
dan sebelumnya syarat itu cuma aturan yang harus diingat — daftar sumber
sesungguhnya ada di dalam sel notebook, terpisah dari provenansnya, sehingga
menambah rekaman tanpa mencatat asalnya sama sekali tidak tertahan. Sekarang
keduanya satu berkas, jadi syarat itu berlaku dengan sendirinya.

Tiga medan — `penutur`, `tanggal_ambil`, `izin` — sengaja hanya diperingatkan,
tidak menghentikan pengunduhan. Menghentikannya akan menggoda orang untuk
mengisi asal supaya jalan, dan provenans yang diisi asal lebih berbahaya
daripada provenans yang jujur kosong: yang pertama tidak terlihat, yang kedua
tercetak tiap kali perintah dijalankan.

Pengunduhannya memeriksa `Content-Length`, bukan sekadar keberadaan berkas.
Unduhan yang putus di tengah meninggalkan berkas yang *ada* tetapi pendek, dan
pemeriksaan keberadaan akan menerimanya sebagai lengkap — lalu ASR
mentranskripsikan potongan itu tanpa memberi tahu siapa pun bahwa rekamannya
terpotong. Karena itu berkas diunduh ke `.part` lebih dulu dan baru diberi nama
aslinya setelah jumlah baitnya cocok.

Hanya pustaka standar yang dipakai, sesuai sifat prototipe ini.

CATATAN ETIKA. Tautan pada kolom `tautan_sumber` boleh jadi terbuka untuk umum
tanpa autentikasi. Itu sah untuk rekaman peneliti sendiri, tetapi rekaman yang
memuat suara pihak ketiga tidak boleh ditaruh di sana; pakai salinan manual
atau penyimpanan yang menuntut kredensial. Ketentuan selengkapnya ada pada
docstring `disfluency_id/ingest.py`.
"""

from __future__ import annotations

import csv
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

BELUM = "BELUM DICATAT"

#: Kolom yang wajib ada pada berkas provenans.
#:
#: `tautan_sumber` dan `tautan_unduh` sengaja dipisah. Yang pertama adalah asal
#: kanonis karya itu — halaman YouTube, laman podcast — dan itulah yang dikutip
#: serta yang menentukan lisensinya. Yang kedua sekadar tempat rangkaian ini
#: mengambil berkasnya, lazimnya cermin di CDN. Menyatukan keduanya membuat
#: cermin tampak seperti asal, dan lisensi karya jadi tidak terlacak.
KOLOM_WAJIB = (
    "berkas",
    "judul",
    "penutur",
    "tautan_sumber",
    "tautan_unduh",
    "lisensi",
    "tanggal_terbit",
    "tanggal_ambil",
    "durasi_detik",
    "durasi_asli_detik",
    "jendela",
    "izin",
    "catatan",
)

#: Medan yang menentukan boleh-tidaknya rekaman dipakai dalam publikasi.
KOLOM_ETIKA = ("penutur", "tautan_sumber", "lisensi", "tanggal_ambil", "izin")

BLOK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class Sumber:
    """Satu baris provenans."""

    berkas: str
    judul: str
    penutur: str
    tautan_sumber: str
    tautan_unduh: str
    lisensi: str
    tanggal_terbit: str
    tanggal_ambil: str
    durasi_detik: float | None
    durasi_asli_detik: float | None
    jendela: int | None
    izin: str
    catatan: str

    @property
    def alamat_unduh(self) -> str:
        """Tempat berkas diambil: cermin bila ada, kalau tidak asal kanonisnya."""
        return self.tautan_unduh or self.tautan_sumber

    @property
    def porsi_kutipan(self) -> float | None:
        """Bagian karya asli yang dipakai, 0..1; None bila belum terukur.

        Ini ukuran yang dipakai untuk menimbang penggunaan wajar: kutipan
        pendek dari karya panjang untuk keperluan penelitian berbeda
        kedudukannya dengan penyalinan karya secara utuh. Angkanya harus dapat
        dihitung, bukan dinyatakan, karena itulah kedua durasi disimpan
        terpisah alih-alih hanya rasionya.
        """
        if not self.durasi_asli_detik or self.durasi_detik is None:
            return None
        return self.durasi_detik / self.durasi_asli_detik

    @property
    def belum_dicatat(self) -> list[str]:
        """Medan etika yang masih kosong, sesuai urutan `KOLOM_ETIKA`."""
        return [
            k
            for k in KOLOM_ETIKA
            if not str(getattr(self, k)).strip()
            or str(getattr(self, k)).strip().upper() == BELUM
        ]

    @property
    def siap_dikutip(self) -> bool:
        """Benar bila seluruh medan etika sudah terisi.

        Namanya sengaja bukan `valid`: berkasnya tetap boleh ditranskripsikan
        untuk keperluan uji coba, yang tidak boleh adalah mengutip angka
        hasilnya di dokumen penelitian sebelum medan ini lengkap.
        """
        return not self.belum_dicatat


def baca_provenans(path: Path | str) -> list[Sumber]:
    """Baca `provenans.csv` menjadi daftar `Sumber`.

    Berkas provenans ditulis tangan, jadi kesalahannya pun kesalahan manusia:
    kolom salah eja, baris kosong di ujung, durasi yang belum diisi. Semuanya
    ditangani dengan pesan yang menyebut apa yang kurang, bukan `KeyError`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"berkas provenans tidak ditemukan: {path}")

    with path.open(encoding="utf-8", newline="") as fh:
        baris = list(csv.DictReader(fh))

    if not baris:
        raise ValueError(f"{path} tidak memuat satu baris pun selain kepala kolom")

    kurang = [k for k in KOLOM_WAJIB if k not in (baris[0].keys())]
    if kurang:
        raise ValueError(
            f"{path} kekurangan kolom: {', '.join(kurang)}\n"
            f"Kolom yang diharapkan: {', '.join(KOLOM_WAJIB)}"
        )

    hasil: list[Sumber] = []
    for i, b in enumerate(baris, start=2):  # baris 1 adalah kepala kolom
        nama = (b.get("berkas") or "").strip()
        if not nama:
            continue  # baris kosong di ujung berkas, lazim pada CSV sunting tangan

        def angka(kolom: str) -> float | None:
            mentah = (b.get(kolom) or "").strip()
            if not mentah or mentah.upper() == BELUM:
                return None
            try:
                return float(mentah)
            except ValueError as exc:
                raise ValueError(
                    f"{path} baris {i}: {kolom} '{mentah}' bukan angka"
                ) from exc

        hasil.append(
            Sumber(
                berkas=nama,
                judul=(b.get("judul") or "").strip(),
                penutur=(b.get("penutur") or "").strip(),
                tautan_sumber=(b.get("tautan_sumber") or "").strip(),
                tautan_unduh=(b.get("tautan_unduh") or "").strip(),
                lisensi=(b.get("lisensi") or "").strip(),
                tanggal_terbit=(b.get("tanggal_terbit") or "").strip(),
                tanggal_ambil=(b.get("tanggal_ambil") or "").strip(),
                durasi_detik=angka("durasi_detik"),
                durasi_asli_detik=angka("durasi_asli_detik"),
                jendela=int(angka("jendela")) if angka("jendela") else None,
                izin=(b.get("izin") or "").strip(),
                catatan=(b.get("catatan") or "").strip(),
            )
        )

    ganda = _cari_ganda([s.berkas for s in hasil])
    if ganda:
        raise ValueError(
            f"{path} memuat nama berkas ganda: {', '.join(sorted(ganda))}. "
            "Tiap rekaman harus punya tepat satu baris provenans."
        )
    return hasil


def _cari_ganda(nama: Sequence[str]) -> set[str]:
    terlihat: set[str] = set()
    ganda: set[str] = set()
    for n in nama:
        (ganda if n in terlihat else terlihat).add(n)
    return ganda


def perlu_unduh(tujuan: Path, n_bait: int | None) -> bool:
    """Putuskan apakah berkas masih perlu diunduh.

    `n_bait` None berarti server tidak memberi `Content-Length`; dalam keadaan
    itu keberadaan berkas terpaksa dianggap cukup, sebab tidak ada angka
    pembanding. Keadaan itu dilaporkan pemanggil, bukan disembunyikan.
    """
    if not tujuan.exists():
        return True
    if n_bait is None:
        return False
    return tujuan.stat().st_size != n_bait


def _ganti_nama(
    sementara: Path, tujuan: Path, *, percobaan: int = 12, jeda: float = 0.25
) -> None:
    """Ubah `.part` menjadi nama sebenarnya, tahan kunci sesaat dari Windows.

    Di Windows, pemindai virus lazim membuka berkas yang baru selesai ditulis
    untuk diperiksa. Selama pemeriksaan itu `os.replace` gagal dengan
    `PermissionError` (WinError 32) walaupun program ini sudah menutup
    berkasnya — jadi kegagalannya bukan kesalahan program dan tidak dapat
    diperbaiki dengan menutup apa pun. Kuncinya berumur pendek, maka
    penggantian nama diulang beberapa kali sebelum menyerah.

    Menyerah pun tidak merusak apa-apa: `.part` tetap utuh dan lengkap, dan
    jalankan ulang akan menemukannya. Di Linux — termasuk Colab — cabang ini
    tidak pernah terpakai.
    """
    for sisa in range(percobaan, 0, -1):
        try:
            sementara.replace(tujuan)
            return
        except PermissionError:
            if sisa == 1:
                raise
            time.sleep(jeda)


def unduh(
    url: str,
    tujuan: Path | str,
    *,
    pembuka: Callable = urllib.request.urlopen,
    blok: int = BLOK,
    lapor: Callable[[str], None] | None = None,
) -> tuple[Path, str]:
    """Unduh `url` ke `tujuan`, verifikasi panjangnya, kembalikan status.

    Status yang mungkin: `"ada"` (sudah lengkap, tidak diunduh ulang),
    `"terunduh"`, atau `"terunduh-tanpa-verifikasi"` bila server tidak memberi
    `Content-Length`.

    `pembuka` dapat diganti agar jalur ini teruji tanpa jaringan.
    """
    tujuan = Path(tujuan)
    tujuan.parent.mkdir(parents=True, exist_ok=True)

    with pembuka(url, timeout=120) as resp:
        kepala = resp.headers.get("Content-Length")
        n_bait = int(kepala) if kepala is not None else None

        if not perlu_unduh(tujuan, n_bait):
            return tujuan, "ada"

        sementara = tujuan.with_suffix(tujuan.suffix + ".part")
        ditulis = 0
        with sementara.open("wb") as fh:
            while True:
                potongan = resp.read(blok)
                if not potongan:
                    break
                fh.write(potongan)
                ditulis += len(potongan)
                if lapor and n_bait:
                    lapor(f"{ditulis / n_bait:6.1%}  {ditulis / 1e6:7.1f} MB")

    if n_bait is not None and ditulis != n_bait:
        # Berkas .part sengaja TIDAK dihapus: ia bukti panjang berapa yang
        # sampai, dan menghapusnya membuat kegagalan berulang tak terlacak.
        raise IOError(
            f"unduhan {url} terpotong: {ditulis} bait diterima, "
            f"{n_bait} diharapkan. Sisa ada di {sementara.name}."
        )

    _ganti_nama(sementara, tujuan)
    return tujuan, "terunduh" if n_bait is not None else "terunduh-tanpa-verifikasi"


def unduh_semua(
    provenans: Path | str,
    media_dir: Path | str,
    *,
    hanya: Sequence[str] | None = None,
    pembuka: Callable = urllib.request.urlopen,
    lapor: Callable[[str], None] | None = print,
) -> list[tuple[Sumber, Path, str]]:
    """Unduh setiap rekaman yang terdaftar di provenans dan belum lengkap.

    `hanya` membatasi ke nama berkas tertentu. Baris tanpa `tautan_sumber`
    dilewati dengan status `"tanpa-tautan"` — itu keadaan yang sah bagi
    rekaman yang disalin manual, dan bukan kesalahan.
    """
    media_dir = Path(media_dir)
    daftar = baca_provenans(provenans)
    if hanya:
        pilih = set(hanya)
        tak_dikenal = pilih - {s.berkas for s in daftar}
        if tak_dikenal:
            raise ValueError(
                f"tidak ada baris provenans untuk: {', '.join(sorted(tak_dikenal))}. "
                f"Tambahkan barisnya di {provenans} lebih dulu."
            )
        daftar = [s for s in daftar if s.berkas in pilih]

    hasil: list[tuple[Sumber, Path, str]] = []
    for s in daftar:
        tujuan = media_dir / s.berkas
        if not s.alamat_unduh:
            hasil.append((s, tujuan, "tanpa-tautan"))
            if lapor:
                ada = "ada" if tujuan.exists() else "TIDAK ADA"
                lapor(f"  {s.berkas:<24} tanpa tautan, salin manual ({ada})")
            continue

        _, status = unduh(s.alamat_unduh, tujuan, pembuka=pembuka)
        hasil.append((s, tujuan, status))
        if lapor:
            ukuran = tujuan.stat().st_size / 1e6 if tujuan.exists() else 0.0
            lapor(f"  {s.berkas:<24} {status:<28} {ukuran:7.1f} MB")

    return hasil


def ringkas_etika(daftar: Sequence[Sumber]) -> list[str]:
    """Susun peringatan untuk tiap sumber yang medan etikanya belum lengkap."""
    pesan: list[str] = []
    for s in daftar:
        if s.belum_dicatat:
            pesan.append(f"  {s.berkas:<24} belum dicatat: {', '.join(s.belum_dicatat)}")
    return pesan


def total_durasi(daftar: Sequence[Sumber]) -> tuple[float, int]:
    """Kembalikan (total detik, jumlah baris yang durasinya belum diisi)."""
    total = sum(s.durasi_detik or 0.0 for s in daftar)
    kosong = sum(1 for s in daftar if s.durasi_detik is None)
    return total, kosong
