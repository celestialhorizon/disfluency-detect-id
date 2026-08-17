"""Audit mutu transkrip ASR sebelum dipakai sebagai data penelitian.

Transkrip otomatis bukan pengganti transkrip manusia, dan kekurangannya
tidak acak melainkan berpola. Dua pola merusak penelitian ini secara
khusus, dan modul ini mengukur keduanya supaya besarnya diketahui, bukan
diduga:

1.  **Penghapusan filler.** Whisper dilatih menghasilkan transkrip yang
    rapi dan enak dibaca, sehingga jeda terisi ('eee', 'anu') cenderung
    dibuang diam-diam. Yang dibuang itu justru objek penelitian ini.
    Laju filler yang jauh di bawah laju ujaran spontan lisan adalah
    tanda penghapusan, bukan tanda penutur yang fasih.

2.  **Penanda waktu yang runtuh.** Penanda waktu Whisper diturunkan dari
    cross-attention, bukan dari penjajaran akustik. Akibatnya batas kata
    kerap dilaporkan bersambung persis: akhir kata sebelumnya sama
    dengan awal kata berikutnya, jeda 0,000 detik. Jeda nol semacam itu
    bukan pengukuran melainkan artefak.

Pola kedua berbahaya justru karena tampak masuk akal. Modul reduplikasi
membaca jeda rapat sebagai bukti reduplikasi gramatikal -- alasannya
reduplikasi diucapkan sebagai satu satuan prosodi. Bila ASR melaporkan
jeda nol di mana-mana, bukti itu terkirim ke seluruh pasangan kata
berulang, dan sistem condong mempertahankan repetisi yang seharusnya
dipotong. Biasnya berarah, bukan derau.

Kesimpulan praktisnya: fitur prosodi hanya sah dihitung di atas
penanda waktu hasil forced alignment. Angka dari modul inilah yang
menjadi dasar kuantitatif pernyataan tersebut.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .lexicon import CAT_DM, CAT_EDIT, CAT_FP, CAT_FP_AMB, Lexicon
from .reduplication import classify_adjacent_repeat, find_repeat_events
from .schema import Utterance

#: Di bawah nilai ini penanda waktu dianggap bersambung, bukan terukur.
ZERO_GAP_EPS = 1e-6

#: Laju DISFLUENSI (jeda terisi + pengulangan + restart) pada percakapan
#: spontan: 5,97 per 100 kata, rerata 96 penutur pada korpus percakapan
#: tugas-terarah -- Bortfeld, Leon, Bloom, Schober & Brennan (2001),
#: "Disfluency rates in conversation", Language and Speech 44(2), 123-147.
#: Oviatt (1995) melaporkan kisaran yang sejalan: 5,50-8,83 per 100 kata
#: pada dialog, 3,60 pada monolog.
#:
#: Angka ini TIDAK sebanding dengan medan `FillerAudit.rate` di bawah.
#: Yang dihitung Bortfeld dkk. adalah jeda terisi non-leksikal (uh, um, er)
#: bersama pengulangan dan restart, dan penanda wacana TIDAK ikut. `rate`
#: menghitung jeda terisi + penanda wacana + penanda ralat, tanpa
#: pengulangan. Dua definisi yang berbeda; membandingkannya langsung adalah
#: jebakan penyebut yang sama seperti pada persentase jeda nol.
LAJU_DISFLUENSI_LISAN_SPONTAN = 0.0597

#: Laju JEDA TERISI saja pada korpus yang sama: 2,56 per 100 kata (rerata
#: keseluruhan; per sel rancangan 2,04-3,03, dan 3,30 pada peran director
#: yang beban perencanaannya lebih berat). Inilah pembanding yang sah untuk
#: `FillerAudit.fp_rate`, sebab keduanya menghitung hal yang sama.
#:
#: Rujukannya berbahasa Inggris, dan itu keterbatasan yang harus disebut:
#: satu-satunya kajian filler Bahasa Indonesia yang ditemukan -- Nugraha &
#: Tarmini (2023), Deskripsi Bahasa 6(2), 60-74 -- melaporkan KOMPOSISI
#: filler (67% non-leksikal dari 221 filler) tanpa penyebut token, sehingga
#: laju per 100 kata tidak dapat diturunkan darinya.
LAJU_JEDA_TERISI_LISAN_SPONTAN = 0.0256

#: Nama lama, dipertahankan supaya rujukan luar tidak putus.
LAJU_FILLER_LISAN_SPONTAN = LAJU_DISFLUENSI_LISAN_SPONTAN


@dataclass
class GapAudit:
    """Ringkasan sebaran jeda antar-kata pada satu himpunan ujaran."""

    total: int = 0
    zero: int = 0
    in_redup_zone: int = 0
    in_grey_zone: int = 0
    in_disfluency_zone: int = 0
    values: list[float] = field(default_factory=list)

    @property
    def zero_share(self) -> float:
        return self.zero / self.total if self.total else 0.0

    @property
    def median(self) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    @property
    def prosody_is_trustworthy(self) -> bool:
        """Jeda nol yang melimpah menandakan penanda waktu tidak terukur.

        Ambang 20% dipilih longgar dengan sengaja: penjajaran yang benar
        pun menghasilkan sebagian kecil jeda nol pada kata bersambung.
        Melampaui seperlima, penjelasan yang lebih masuk akal adalah
        artefak penjajaran.
        """
        return self.zero_share < 0.20


@dataclass
class FillerAudit:
    """Hitung token filler yang benar-benar muncul di transkrip."""

    tokens: int = 0
    filled_pause: int = 0
    filled_pause_ambiguous: int = 0
    discourse_marker: int = 0
    editing_term: int = 0

    #: Kata yang terdaftar sebagai penanda ralat tetapi tidak kuat, sehingga
    #: keanggotaannya belum berarti ia sedang menandai ralat. 'bukan' adalah
    #: contohnya: pengingkar paling lazim dalam Bahasa Indonesia, dan hanya
    #: sesekali penanda koreksi. Dihitung terpisah dengan alasan yang sama
    #: seperti `filled_pause_ambiguous`, dan sama-sama tidak masuk `total`.
    editing_term_ambiguous: int = 0

    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.filled_pause + self.discourse_marker + self.editing_term

    @property
    def rate(self) -> float:
        return self.total / self.tokens if self.tokens else 0.0

    @property
    def fp_rate(self) -> float:
        return self.filled_pause / self.tokens if self.tokens else 0.0

    @property
    def suspected_deletion(self) -> bool:
        """Jeda terisi yang nyaris nihil menandakan ASR membuangnya.

        Diuji pada jeda terisi saja, bukan seluruh filler: penanda wacana
        ('kayak', 'gitu') adalah kata sungguhan yang lolos transkripsi,
        sedangkan 'eee' dan 'mmm' justru yang dirapikan ASR.

        Ambangnya SETENGAH laju rujukan (2,56% -> 1,28%). Faktor dua dipilih
        supaya yang tertangkap adalah penghapusan besar-besaran, bukan ragam
        antar-penutur yang wajar -- pada korpus rujukannya sendiri laju per
        sel rancangan bergerak 2,04-3,30%.

        Versi sebelumnya membagi 6% dengan tiga. Pembilangnya keliru: 6%
        adalah laju SELURUH disfluensi termasuk pengulangan dan restart,
        bukan laju jeda terisi, jadi ambang lamanya aritmetika di atas
        besaran yang salah. Putusan atas korpus proyek ini tidak berubah --
        laju jeda terisi kelima berkasnya 2,59-6,93%, di atas kedua ambang.
        """
        return self.fp_rate < LAJU_JEDA_TERISI_LISAN_SPONTAN / 2


@dataclass
class RepeatAudit:
    """Putusan modul reduplikasi atas pasangan kata berulang yang ditemukan."""

    events: int = 0
    reduplication: int = 0
    disfluency: int = 0
    ambiguous: int = 0
    decided_by_prosody_alone: int = 0
    samples: list[tuple[str, str, float, float]] = field(default_factory=list)

    @property
    def prosody_dependent_share(self) -> float:
        return self.decided_by_prosody_alone / self.events if self.events else 0.0


@dataclass
class AsrAudit:
    """Hasil audit lengkap satu korpus hasil ASR."""

    utterances: int
    gaps: GapAudit
    fillers: FillerAudit
    repeats: RepeatAudit
    duration: float = 0.0


def _gap_audit(utts: Sequence[Utterance], lex: Lexicon) -> GapAudit:
    a = GapAudit()
    for utt in utts:
        for prev, cur in zip(utt.tokens, utt.tokens[1:]):
            gap = max(0.0, cur.start - prev.end)
            a.total += 1
            a.values.append(gap)
            if gap <= ZERO_GAP_EPS:
                a.zero += 1
            if gap <= lex.gap_redup_max:
                a.in_redup_zone += 1
            elif gap >= lex.gap_disfluency_min:
                a.in_disfluency_zone += 1
            else:
                a.in_grey_zone += 1
    return a


def _filler_audit(utts: Sequence[Utterance], lex: Lexicon) -> FillerAudit:
    a = FillerAudit()
    for utt in utts:
        for tok in utt.tokens:
            a.tokens += 1
            cat = lex.category(tok.text)
            if cat is None:
                continue
            key = tok.norm
            a.counts[key] = a.counts.get(key, 0) + 1
            if cat == CAT_FP:
                a.filled_pause += 1
            elif cat == CAT_FP_AMB:
                a.filled_pause_ambiguous += 1
            elif cat == CAT_DM:
                a.discourse_marker += 1
            elif cat == CAT_EDIT:
                if lex.is_strong_editing_term(tok.text):
                    a.editing_term += 1
                else:
                    a.editing_term_ambiguous += 1
    return a


def _neutral_gap(lex: Lexicon) -> float:
    """Jeda yang menyumbang bobot nol, yaitu titik tengah zona abu-abu.

    Interpolasi zona abu-abu bernilai 1,5 - 3,0f dengan f pecahan posisi;
    ia nol tepat di tengah. Memakai jeda ini setara dengan mematikan bukti
    prosodi tanpa perlu menyentuh pengklasifikasinya.
    """
    return (lex.gap_redup_max + lex.gap_disfluency_min) / 2


def _repeat_audit(utts: Sequence[Utterance], lex: Lexicon) -> RepeatAudit:
    a = RepeatAudit()
    neutral = _neutral_gap(lex)
    for utt in utts:
        for ev in find_repeat_events(utt, lex):
            a.events += 1
            if ev.kind == "reduplication":
                a.reduplication += 1
            elif ev.kind == "disfluency":
                a.disfluency += 1
            else:
                a.ambiguous += 1

            # Uji kontrafaktual: adakah putusan berubah bila bukti prosodi
            # dicabut? Menjalankan ulang pengklasifikasi jauh lebih jujur
            # daripada menaksir bobot dari teks alasan, yang bentuknya bisa
            # berubah kapan saja.
            # Durasi dan panjang rentetan dipertahankan apa adanya supaya
            # yang berubah benar-benar hanya jeda.
            tanpa_prosodi, _, _ = classify_adjacent_repeat(
                base=ev.base, gap=neutral, lex=lex, run_length=ev.run_length
            )
            if tanpa_prosodi != ev.kind:
                a.decided_by_prosody_alone += 1

            if len(a.samples) < 25:
                a.samples.append((utt.uid, ev.base, ev.gap, ev.score))
    return a


def audit(utts: Sequence[Utterance], lex: Lexicon | None = None) -> AsrAudit:
    """Jalankan seluruh pemeriksaan atas korpus hasil ASR."""
    lex = lex or Lexicon.load()
    duration = 0.0
    for utt in utts:
        if utt.tokens:
            duration += utt.tokens[-1].end - utt.tokens[0].start
    return AsrAudit(
        utterances=len(utts),
        gaps=_gap_audit(utts, lex),
        fillers=_filler_audit(utts, lex),
        repeats=_repeat_audit(utts, lex),
        duration=duration,
    )


def format_audit(a: AsrAudit, title: str = "Audit transkrip ASR") -> str:
    """Laporan teks yang bisa langsung disalin ke catatan penelitian."""
    g, f, r = a.gaps, a.fillers, a.repeats

    def pct(n: int) -> str:
        return f"{n} ({n / g.total:.1%})" if g.total else str(n)

    lines = [
        title,
        "=" * len(title),
        "",
        f"Ujaran           : {a.utterances}",
        f"Token            : {f.tokens}",
        f"Durasi bicara    : {a.duration:.1f} s ({a.duration / 60:.1f} menit)",
        "",
        "1. Keandalan penanda waktu",
        "-" * 26,
        f"  Jeda antar-kata terukur : {g.total}",
        f"  Jeda persis 0,000 s     : {pct(g.zero)}",
        f"  Median jeda             : {g.median:.3f} s",
        f"  Zona reduplikasi        : {pct(g.in_redup_zone)}",
        f"  Zona abu-abu            : {pct(g.in_grey_zone)}",
        f"  Zona disfluensi         : {pct(g.in_disfluency_zone)}",
        "",
    ]
    if g.prosody_is_trustworthy:
        lines.append("  PUTUSAN: sebaran jeda wajar; fitur prosodi boleh dipakai.")
    else:
        lines += [
            "  PUTUSAN: penanda waktu RUNTUH.",
            f"  {g.zero_share:.1%} jeda dilaporkan persis nol -- itu artefak",
            "  penjajaran cross-attention, bukan hasil pengukuran akustik.",
            "  Fitur prosodi di atas transkrip ini TIDAK SAH: jeda nol",
            "  terbaca sebagai bukti reduplikasi, sehingga sistem condong",
            "  mempertahankan repetisi yang seharusnya dipotong. Biasnya",
            "  berarah, bukan derau, jadi tidak hilang dengan menambah data.",
            "  Tindakan: hitung ulang penanda waktu dengan forced alignment",
            "  (mis. WhisperX / Montreal Forced Aligner) sebelum dipakai.",
        ]

    lines += [
        "",
        "2. Kelengkapan filler",
        "-" * 21,
        f"  Jeda terisi (eee, mmm)  : {f.filled_pause} ({f.fp_rate:.2%} token)",
        f"    pembanding kepustakaan : {LAJU_JEDA_TERISI_LISAN_SPONTAN:.2%} token "
        "(Bortfeld dkk. 2001)",
        f"  Penanda wacana          : {f.discourse_marker}",
        f"  Penanda ralat (kuat)    : {f.editing_term}",
        f"  Total filler            : {f.total} ({f.rate:.2%} token)",
        "    Angka total ini TIDAK sebanding dengan laju kepustakaan mana pun:",
        "    ia memuat penanda wacana yang tidak dihitung sebagai disfluensi di",
        "    sana, dan tidak memuat pengulangan yang di sana dihitung.",
        "",
        "  Bentuk ambigu, dihitung terpisah dan TIDAK masuk total:",
        f"    Jeda terisi ambigu    : {f.filled_pause_ambiguous}  (eh, ah, oh)",
        f"    Penanda ralat ambigu  : {f.editing_term_ambiguous}  (bukan)",
        "    Keanggotaan leksikon bagi bentuk ini belum berarti fungsinya;",
        "    konteks yang memutuskan, dan itu tugas detektor bukan audit.",
        "",
    ]
    if f.suspected_deletion:
        lines += [
            "  PUTUSAN: jeda terisi DIDUGA DIHAPUS ASR.",
            f"  Laju {f.fp_rate:.2%} di bawah setengah "
            f"{LAJU_JEDA_TERISI_LISAN_SPONTAN:.2%},",
            "  laju jeda terisi percakapan spontan menurut Bortfeld dkk.",
            "  (2001). Penjelasan yang lebih masuk akal daripada penutur",
            "  yang luar biasa fasih adalah transkripsi yang merapikan",
            "  keluarannya.",
            "  Akibatnya recall filler terukur terlalu tinggi: yang hilang",
            "  di transkrip tidak pernah masuk hitungan sebagai kesalahan.",
            "  Tindakan: koreksi manual transkrip WAJIB, bukan opsional.",
        ]
    else:
        lines.append("  PUTUSAN: laju filler wajar untuk ujaran spontan.")

    if f.counts:
        top = sorted(f.counts.items(), key=lambda kv: -kv[1])[:12]
        lines += ["", "  Filler yang lolos: " + ", ".join(f"{w}({n})" for w, n in top)]

    lines += [
        "",
        "3. Pasangan kata berulang",
        "-" * 25,
        f"  Peristiwa ditemukan     : {r.events}",
        f"  Diputus reduplikasi     : {r.reduplication}",
        f"  Diputus disfluensi      : {r.disfluency}",
        f"  Ambigu                  : {r.ambiguous}",
        f"  Putusan berbalik tanpa prosodi : {r.decided_by_prosody_alone}"
        + (f" ({r.prosody_dependent_share:.1%})" if r.events else ""),
    ]
    if r.events and not g.prosody_is_trustworthy:
        lines += [
            "",
            "  Peringatan: putusan di atas ikut memakai jeda yang sudah",
            "  dinyatakan tidak sah pada bagian 1. Perlakukan sebagai",
            "  keluaran mentah untuk dianotasi ulang, bukan sebagai hasil.",
        ]
    if r.samples:
        lines += ["", "  Contoh:"]
        for uid, base, gap, score in r.samples[:10]:
            lines.append(f"    {uid:<12} '{base} {base}'  jeda {gap:.3f}s  skor {score:+.2f}")
    return "\n".join(lines)


def audit_dict(a: AsrAudit) -> dict:
    """Bentuk JSON untuk dilampirkan pada catatan penelitian."""
    g, f, r = a.gaps, a.fillers, a.repeats
    return {
        "ujaran": a.utterances,
        "token": f.tokens,
        "durasi_detik": round(a.duration, 2),
        "penanda_waktu": {
            "jeda_terukur": g.total,
            "jeda_nol": g.zero,
            "pangsa_jeda_nol": round(g.zero_share, 4),
            "median_jeda": round(g.median, 4),
            "zona_reduplikasi": g.in_redup_zone,
            "zona_abu_abu": g.in_grey_zone,
            "zona_disfluensi": g.in_disfluency_zone,
            "prosodi_layak_pakai": g.prosody_is_trustworthy,
        },
        "filler": {
            "jeda_terisi": f.filled_pause,
            "penanda_wacana": f.discourse_marker,
            "penanda_ralat": f.editing_term,
            "ambigu": f.filled_pause_ambiguous,
            "penanda_ralat_ambigu": f.editing_term_ambiguous,
            "total": f.total,
            "laju": round(f.rate, 4),
            "laju_jeda_terisi": round(f.fp_rate, 4),
            "diduga_dihapus_asr": f.suspected_deletion,
            "rincian": dict(sorted(f.counts.items(), key=lambda kv: -kv[1])),
        },
        "pengulangan": {
            "peristiwa": r.events,
            "reduplikasi": r.reduplication,
            "disfluensi": r.disfluency,
            "ambigu": r.ambiguous,
        },
    }
