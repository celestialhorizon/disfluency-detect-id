"""Eksekusi Edit Decision List menjadi berkas video.

`edl.py` berhenti pada naskah ffmpeg; modul inilah yang menjalankannya.
Pemisahan itu disengaja: penyusunan potongan murni aritmetika penanda waktu dan
dapat diuji tanpa satu pun berkas media, sedangkan pemanggilan ffmpeg
bergantung pada perkakas luar. Karena itu seluruh isi modul ini menyusun
argumen terlebih dahulu dan menjalankannya belakangan, sehingga perintahnya
dapat diperiksa (`--dry-run`) tanpa video apa pun.

Dua mode:

1. `penuh` -- seluruh segmen yang disimpan disambung menjadi rekaman utuh tanpa
   disfluensi.
2. `bukti` -- hanya di sekitar tiap potongan, dipasangkan SEBELUM lalu SESUDAH.
   Mode ini yang layak ditonton: pada rekaman sungguhan pemangkasannya hanya
   sekitar satu persen durasi, jadi menonton hasil penuh nyaris tidak
   memperlihatkan apa yang berubah.

Penyambungan memakai trim/atrim + concat dalam SATU proses ffmpeg, bukan
memotong menjadi banyak berkas lalu menggabungnya. Alasannya menentukan:
concat demuxer menuntut tiap potongan mulai pada keyframe, dan pembulatan ke
keyframe terdekat menggeser titik potong sampai beberapa detik -- persis yang
tidak boleh terjadi pada potongan selebar 0,3 detik.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

#: Berkas fonta untuk label SEBELUM/SESUDAH. Tidak ada yang wajib ada; bila
#: tidak satu pun ditemukan, labelnya dilewati alih-alih menggagalkan render.
FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def find_font() -> str | None:
    return next((f for f in FONT_CANDIDATES if Path(f).exists()), None)


@dataclass(frozen=True)
class Span:
    """Satu potong garis waktu yang ikut ke berkas keluaran."""

    start: float
    end: float
    label: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class RenderConfig:
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    crf: int = 23
    preset: str = "veryfast"
    font: str | None = None


# --------------------------------------------------------------------------
# Membaca EDL
# --------------------------------------------------------------------------


def load_edl(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def timeline_bounds(payload: dict) -> tuple[float, float]:
    """Ujung kiri dan kanan garis waktu EDL.

    Diambil dari isi EDL, bukan dari `durasi_asli_detik`: garis waktunya mulai
    pada kata pertama, yang belum tentu detik nol.
    """
    tepi = [(s["mulai"], s["selesai"]) for s in payload.get("segmen_disimpan", [])]
    tepi += [(c["mulai"], c["selesai"]) for c in payload.get("potongan", [])]
    if not tepi:
        return 0.0, 0.0
    return min(a for a, _ in tepi), max(b for _, b in tepi)


def keep_spans(payload: dict) -> list[Span]:
    """Mode `penuh`: seluruh segmen yang dipertahankan, apa adanya."""
    return [
        Span(s["mulai"], s["selesai"])
        for s in payload.get("segmen_disimpan", [])
        if s["selesai"] > s["mulai"]
    ]


def probe_duration(path: Path | str) -> float | None:
    """Durasi berkas media menurut ffprobe, atau None bila tidak terbaca."""
    if shutil.which("ffprobe") is None:
        return None
    hasil = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(hasil.stdout.strip())
    except ValueError:
        return None


def extend_to_media(spans: Sequence[Span], media_duration: float | None = None) -> list[Span]:
    """Rentangkan segmen pertama ke detik nol dan yang terakhir ke ujung berkas.

    Garis waktu EDL mulai pada kata pertama dan berhenti pada kata terakhir,
    sehingga menyambung segmen apa adanya ikut membuang senyap pembuka dan
    penutup -- padahal yang diminta hanya membuang disfluensi. Tanpa ini,
    rekaman 15,9 detik dengan potongan 1,0 detik keluar sebagai 10,7 detik.
    """
    if not spans:
        return []
    keluar = list(spans)
    keluar[0] = Span(0.0, keluar[0].end, keluar[0].label)
    if media_duration is not None and media_duration > keluar[-1].end:
        keluar[-1] = Span(keluar[-1].start, media_duration, keluar[-1].label)
    return keluar


def proof_spans(
    payload: dict,
    pad: float = 2.0,
    max_cuts: int | None = None,
) -> list[Span]:
    """Mode `bukti`: tiap potongan jadi sepasang cuplikan SEBELUM lalu SESUDAH.

    Jendela di kiri dan kanan dipangkas agar tidak menyeberang ke potongan
    tetangga. Tanpa itu, cuplikan SESUDAH bisa memuat disfluensi lain yang
    belum dipotong -- yang membuat bukti justru membantah dirinya sendiri.
    """
    cuts = payload.get("potongan", [])
    if max_cuts is not None:
        cuts = cuts[:max_cuts]
    lo, hi = timeline_bounds(payload)
    semua = payload.get("potongan", [])

    spans: list[Span] = []
    for idx, cut in enumerate(cuts):
        mulai, selesai = cut["mulai"], cut["selesai"]
        batas_kiri = semua[idx - 1]["selesai"] if idx > 0 else lo
        batas_kanan = semua[idx + 1]["mulai"] if idx + 1 < len(semua) else hi

        a = max(lo, batas_kiri, mulai - pad)
        b = min(hi, batas_kanan, selesai + pad)
        if b <= a:
            continue

        spans.append(Span(a, b, "SEBELUM"))
        if mulai - a > 1e-3:
            spans.append(Span(a, mulai, "SESUDAH"))
        if b - selesai > 1e-3:
            spans.append(Span(selesai, b, "SESUDAH"))
    return spans


# --------------------------------------------------------------------------
# Menyusun perintah
# --------------------------------------------------------------------------


def _escape_text(teks: str) -> str:
    return teks.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def filter_complex(spans: Sequence[Span], cfg: RenderConfig | None = None) -> str:
    """Rantai trim/atrim + concat untuk seluruh span, video dan audio sejalan."""
    cfg = cfg or RenderConfig()
    if not spans:
        raise ValueError("tidak ada segmen tersisa; periksa EDL atau konfigurasi potongan")

    bagian: list[str] = []
    for i, sp in enumerate(spans):
        v = f"[0:v]trim=start={sp.start:.3f}:end={sp.end:.3f},setpts=PTS-STARTPTS"
        if cfg.fps:
            v += f",fps={cfg.fps}"
        if cfg.width and cfg.height:
            v += (
                f",scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=decrease"
                f",pad={cfg.width}:{cfg.height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
            )
        if sp.label and cfg.font:
            v += (
                f",drawtext=fontfile='{cfg.font}':text='{_escape_text(sp.label)}'"
                ":x=24:y=24:fontsize=32:fontcolor=white"
                ":box=1:boxcolor=black@0.6:boxborderw=12"
            )
        bagian.append(v + f"[v{i}]")
        bagian.append(
            f"[0:a]atrim=start={sp.start:.3f}:end={sp.end:.3f},asetpts=PTS-STARTPTS[a{i}]"
        )

    aliran = "".join(f"[v{i}][a{i}]" for i in range(len(spans)))
    return ";".join(bagian) + f";{aliran}concat=n={len(spans)}:v=1:a=1[v][a]"


def ffmpeg_argv(
    source: Path | str,
    output: Path | str,
    spans: Sequence[Span],
    cfg: RenderConfig | None = None,
) -> list[str]:
    """Argumen ffmpeg lengkap, sebagai daftar -- tidak pernah lewat shell."""
    cfg = cfg or RenderConfig()
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-filter_complex", filter_complex(spans, cfg),
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output),
    ]


def render(
    source: Path | str,
    output: Path | str,
    spans: Sequence[Span],
    cfg: RenderConfig | None = None,
    dry_run: bool = False,
) -> Path:
    """Jalankan ffmpeg. `dry_run` mengembalikan lintasan tanpa menyentuh apa pun."""
    output = Path(output)
    argv = ffmpeg_argv(source, output, spans, cfg)
    if dry_run:
        return output
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg tidak ditemukan di PATH")
    if not Path(source).exists():
        raise FileNotFoundError(f"berkas sumber tidak ada: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    hasil = subprocess.run(argv, capture_output=True, text=True)
    if hasil.returncode != 0:
        raise RuntimeError(f"ffmpeg gagal (kode {hasil.returncode}):\n{hasil.stderr[-2000:]}")
    return output


def resolve_source(payload: dict, media_dir: Path | str) -> Path:
    """Berkas media yang dirujuk EDL.

    Naskah `.sh` bawaan `edl.py` menulis `input.mp4` sebagai singgahan; nama
    berkas yang sebenarnya hanya hidup di kunci `sumber`. Fungsi ini yang
    menutup celah itu.
    """
    return Path(media_dir) / Path(payload.get("sumber", "")).name


def durasi(spans: Sequence[Span]) -> float:
    return sum(s.duration for s in spans)
