"""Penyusunan Edit Decision List dari hasil deteksi.

Deteksi disfluensi baru bernilai bagi penyunting konten bila ia berujung
pada keputusan potong yang konkret. Modul ini menerjemahkan label token
menjadi daftar potongan bertanda waktu, lalu menurunkannya menjadi tiga
keluaran: berkas EDL berformat JSON, perintah ffmpeg siap jalan, dan
transkrip bersih.

Dua keputusan perancangan yang perlu dicatat dalam laporan:

1. Potongan diperluas ke separuh senyap di kiri dan kanannya. Memotong
   tepat pada batas kata menyisakan senyap menggantung yang terdengar
   janggal; mengambil separuh senyap di kedua sisi menghasilkan sambungan
   yang lebih alami tanpa memakan onset kata tetangga.
2. Pengaman `guard` mencegah potongan menyentuh kata yang dipertahankan.
   Kesalahan memotong satu fonem awal jauh lebih terdengar daripada
   menyisakan senyap 30 milidetik.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .schema import (
    CONSERVATIVE_CUT,
    LABEL_DESC,
    O,
    Utterance,
)


def timecode(seconds: float) -> str:
    """Format HH:MM:SS.mmm, konvensi penyuntingan video."""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@dataclass(frozen=True)
class EDLConfig:
    cut_labels: frozenset[str] = CONSERVATIVE_CUT
    #: Bagian senyap tetangga yang ikut dipotong (0 = tidak sama sekali).
    silence_share: float = 0.5
    #: Jarak aman minimum ke kata yang dipertahankan, dalam detik.
    guard: float = 0.03
    #: Potongan lebih pendek dari ini diabaikan (tidak sepadan).
    min_cut: float = 0.05
    #: Dua potongan yang terpisah kurang dari ini digabung.
    merge_gap: float = 0.12


@dataclass
class Cut:
    start: float
    end: float
    labels: list[str]
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "mulai": round(self.start, 3),
            "selesai": round(self.end, 3),
            "durasi": round(self.duration, 3),
            "mulai_tc": timecode(self.start),
            "selesai_tc": timecode(self.end),
            "label": self.labels,
            "teks": self.text,
            "alasan": [LABEL_DESC.get(l, l) for l in dict.fromkeys(self.labels)],
        }


@dataclass
class EDL:
    source: str
    duration: float
    cuts: list[Cut] = field(default_factory=list)
    keeps: list[tuple[float, float]] = field(default_factory=list)
    transcript_before: str = ""
    transcript_after: str = ""
    config: EDLConfig = field(default_factory=EDLConfig)

    @property
    def removed(self) -> float:
        return sum(c.duration for c in self.cuts)

    @property
    def kept(self) -> float:
        return max(0.0, self.duration - self.removed)

    @property
    def reduction(self) -> float:
        return self.removed / self.duration if self.duration else 0.0

    def stats(self) -> dict:
        return {
            "durasi_asli_detik": round(self.duration, 3),
            "durasi_akhir_detik": round(self.kept, 3),
            "durasi_dipotong_detik": round(self.removed, 3),
            "persen_pemangkasan": round(100 * self.reduction, 2),
            "jumlah_potongan": len(self.cuts),
            "jumlah_segmen_disimpan": len(self.keeps),
        }

    def to_dict(self) -> dict:
        return {
            "sumber": self.source,
            "konfigurasi": {
                "label_dipotong": sorted(self.config.cut_labels),
                "porsi_senyap_diambil": self.config.silence_share,
                "jarak_aman_detik": self.config.guard,
                "potongan_minimum_detik": self.config.min_cut,
                "gabung_jika_jarak_kurang_dari": self.config.merge_gap,
            },
            "statistik": self.stats(),
            "potongan": [c.to_dict() for c in self.cuts],
            "segmen_disimpan": [
                {
                    "mulai": round(a, 3),
                    "selesai": round(b, 3),
                    "mulai_tc": timecode(a),
                    "selesai_tc": timecode(b),
                }
                for a, b in self.keeps
            ],
            "transkrip_sebelum": self.transcript_before,
            "transkrip_sesudah": self.transcript_after,
        }


# --------------------------------------------------------------------------
# Penyusunan
# --------------------------------------------------------------------------


def build_edl(
    utterances: Sequence[Utterance],
    cfg: EDLConfig | None = None,
    use_pred: bool = True,
    source: str = "audio.wav",
) -> EDL:
    """Ubah label token menjadi daftar potongan pada satu garis waktu."""
    cfg = cfg or EDLConfig()
    if not utterances:
        return EDL(source=source, duration=0.0, config=cfg)

    # Satukan seluruh token dari semua ujaran ke satu garis waktu.
    flat: list[tuple[Utterance, int]] = []
    for utt in utterances:
        for i in range(len(utt)):
            flat.append((utt, i))
    flat.sort(key=lambda pair: pair[0].tokens[pair[1]].start)

    def label_of(utt: Utterance, i: int) -> str:
        tok = utt.tokens[i]
        lab = tok.pred if (use_pred and tok.pred is not None) else tok.label
        return lab or O

    n = len(flat)
    timeline_start = flat[0][0].tokens[flat[0][1]].start
    timeline_end = max(u.tokens[i].end for u, i in flat)

    raw: list[Cut] = []
    k = 0
    while k < n:
        utt, i = flat[k]
        if label_of(utt, i) not in cfg.cut_labels:
            k += 1
            continue

        # Rentetan token yang sama-sama dipotong.
        j = k
        labels: list[str] = []
        texts: list[str] = []
        while j < n:
            u2, i2 = flat[j]
            lab = label_of(u2, i2)
            if lab not in cfg.cut_labels:
                break
            labels.append(lab)
            texts.append(u2.tokens[i2].text)
            j += 1

        span_start = flat[k][0].tokens[flat[k][1]].start
        span_end = flat[j - 1][0].tokens[flat[j - 1][1]].end

        # Perluas ke separuh senyap tetangga tanpa menyentuh kata tetangga.
        prev_end = (
            flat[k - 1][0].tokens[flat[k - 1][1]].end if k > 0 else timeline_start
        )
        next_start = (
            flat[j][0].tokens[flat[j][1]].start if j < n else timeline_end
        )

        lead = max(0.0, span_start - prev_end)
        trail = max(0.0, next_start - span_end)
        start = span_start - lead * cfg.silence_share
        end = span_end + trail * cfg.silence_share

        if k > 0:
            start = max(start, prev_end + cfg.guard)
        if j < n:
            end = min(end, next_start - cfg.guard)
        start = max(start, prev_end)
        end = min(end, next_start)

        if end - start >= cfg.min_cut:
            raw.append(Cut(start=start, end=end, labels=labels, text=" ".join(texts)))
        k = j

    cuts = _merge_cuts(raw, cfg.merge_gap)
    keeps = _complement(cuts, timeline_start, timeline_end)

    before = " ".join(t.text for u in utterances for t in u.tokens)
    after = " ".join(
        u.clean_text(cut=cfg.cut_labels, use_pred=use_pred) for u in utterances
    ).strip()

    return EDL(
        source=source,
        duration=timeline_end - timeline_start,
        cuts=cuts,
        keeps=keeps,
        transcript_before=before,
        transcript_after=" ".join(after.split()),
        config=cfg,
    )


def _merge_cuts(cuts: list[Cut], merge_gap: float) -> list[Cut]:
    """Satukan potongan yang nyaris berdempet agar sambungan tidak pecah."""
    if not cuts:
        return []
    merged = [cuts[0]]
    for cut in cuts[1:]:
        last = merged[-1]
        if cut.start - last.end <= merge_gap:
            merged[-1] = Cut(
                start=last.start,
                end=max(last.end, cut.end),
                labels=last.labels + cut.labels,
                text=f"{last.text} {cut.text}".strip(),
            )
        else:
            merged.append(cut)
    return merged


def _complement(
    cuts: list[Cut], start: float, end: float
) -> list[tuple[float, float]]:
    """Segmen yang dipertahankan = garis waktu dikurangi seluruh potongan."""
    keeps: list[tuple[float, float]] = []
    cursor = start
    for cut in cuts:
        if cut.start > cursor:
            keeps.append((cursor, cut.start))
        cursor = max(cursor, cut.end)
    if cursor < end:
        keeps.append((cursor, end))
    return [(a, b) for a, b in keeps if b - a > 1e-6]


# --------------------------------------------------------------------------
# Ekspor
# --------------------------------------------------------------------------


def to_ffmpeg(edl: EDL, input_path: str = "input.mp4", output_path: str = "output.mp4") -> str:
    """Perintah ffmpeg yang menyambung ulang seluruh segmen yang disimpan.

    Menggunakan trim/atrim + concat sehingga video dan audio tetap sinkron.
    Untuk berkas audio saja, buang bagian video secara manual.
    """
    if not edl.keeps:
        return "# tidak ada segmen tersisa; periksa kembali konfigurasi potongan"

    parts: list[str] = []
    for idx, (a, b) in enumerate(edl.keeps):
        parts.append(
            f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{idx}];"
            f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[a{idx}]"
        )
    streams = "".join(f"[v{i}][a{i}]" for i in range(len(edl.keeps)))
    filt = ";".join(parts) + f";{streams}concat=n={len(edl.keeps)}:v=1:a=1[v][a]"

    return (
        f'ffmpeg -i "{input_path}" -filter_complex "{filt}" '
        f'-map "[v]" -map "[a]" "{output_path}"'
    )


def _nama_keluaran(source: str, stem: str) -> str:
    """Nama berkas hasil potong, berdampingan dengan sumbernya."""
    akhiran = Path(source).suffix or ".mp4"
    return f"{stem}_terpotong{akhiran}"


def write_outputs(edl: EDL, outdir: Path | str, stem: str = "edl") -> dict[str, Path]:
    """Tulis EDL sebagai JSON, CSV potongan, dan skrip ffmpeg."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    json_path = outdir / f"{stem}.json"
    json_path.write_text(
        json.dumps(edl.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["json"] = json_path

    csv_path = outdir / f"{stem}_potongan.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mulai_detik", "selesai_detik", "durasi", "mulai_tc", "selesai_tc", "label", "teks"])
        for c in edl.cuts:
            writer.writerow(
                [
                    f"{c.start:.3f}",
                    f"{c.end:.3f}",
                    f"{c.duration:.3f}",
                    timecode(c.start),
                    timecode(c.end),
                    "+".join(dict.fromkeys(c.labels)),
                    c.text,
                ]
            )
    paths["csv"] = csv_path

    sh_path = outdir / f"{stem}_ffmpeg.sh"
    sh_path.write_text(
        "#!/bin/sh\n"
        "# Dihasilkan otomatis oleh prototipe deteksi disfluensi.\n"
        f"# Segmen disimpan: {len(edl.keeps)} | pemangkasan: {100 * edl.reduction:.2f}%\n"
        f"{to_ffmpeg(edl, edl.source, _nama_keluaran(edl.source, stem))}\n",
        encoding="utf-8",
    )
    paths["ffmpeg"] = sh_path

    txt_path = outdir / f"{stem}_transkrip.txt"
    txt_path.write_text(
        "== SEBELUM ==\n"
        f"{edl.transcript_before}\n\n"
        "== SESUDAH ==\n"
        f"{edl.transcript_after}\n",
        encoding="utf-8",
    )
    paths["transkrip"] = txt_path

    return paths
