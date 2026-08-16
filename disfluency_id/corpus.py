"""Pembacaan korpus beranotasi dan sintesis penanda waktu.

Berkas anotasi ditulis dalam format inline `kata/TAG`. Karena korpus
benih berupa teks tanpa audio, penanda waktu tingkat kata disintesis
oleh `simulate_timings` agar pipeline yang menuntut fitur prosodi tetap
dapat dijalankan dari ujung ke ujung.

PERINGATAN METODOLOGIS
Penanda waktu sintetis adalah PENGGANTI SEMENTARA, bukan pengukuran.
Simulator sengaja menumpangtindihkan sebaran jeda kedua kelas (lihat
`TimingProfile.overlap_*`) supaya fitur prosodi tidak dapat memisahkan
kelas secara sempurna dan bukti leksikal tetap diperlukan. Meski begitu,
angka evaluasi yang dihitung di atasnya hanya sahih sebagai uji
keberjalanan sistem. Pada penelitian sesungguhnya seluruh penanda waktu
di sini diganti keluaran forced alignment atas rekaman nyata.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .schema import (
    LABELS,
    O,
    FP,
    PW,
    REP,
    RPR,
    Token,
    Utterance,
    normalize_word,
    syllable_count,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED_CORPUS = DATA_DIR / "corpus" / "seed_id.txt"


# --------------------------------------------------------------------------
# Parsing anotasi inline
# --------------------------------------------------------------------------


def parse_annotated_line(line: str) -> list[tuple[str, str]]:
    """Ubah 'saya/REP saya mau' menjadi [('saya','REP'), ('saya','O'), ...]."""
    out: list[tuple[str, str]] = []
    for raw in line.split():
        if "/" in raw:
            text, _, tag = raw.rpartition("/")
            if text and tag in LABELS:
                out.append((text, tag))
                continue
        out.append((raw, O))
    return out


def read_annotated(path: Path | str = SEED_CORPUS) -> list[list[tuple[str, str]]]:
    """Baca seluruh baris beranotasi, abaikan komentar dan baris kosong."""
    rows: list[list[tuple[str, str]]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parsed = parse_annotated_line(line)
            if parsed:
                rows.append(parsed)
    return rows


# --------------------------------------------------------------------------
# Sintesis penanda waktu
# --------------------------------------------------------------------------


@dataclass
class TimingProfile:
    """Parameter sebaran durasi dan jeda, dalam detik.

    Nilai awal disusun agar konsisten dengan pengamatan umum ragam lisan:
    jeda terisi berdurasi panjang, reduplikasi diucapkan tanpa jeda
    antarparuh, sedangkan repetisi disfluen disela hesitasi.
    """

    # Durasi
    sec_per_syllable: float = 0.085
    dur_floor: float = 0.06
    dur_jitter: float = 0.25
    fp_dur: tuple[float, float] = (0.30, 0.65)
    pw_dur: tuple[float, float] = (0.10, 0.25)
    rep_shrink: tuple[float, float] = (0.55, 1.00)

    # Jeda
    fluent_gap: tuple[float, float] = (0.00, 0.12)
    gap_before_fp: tuple[float, float] = (0.08, 0.40)
    gap_after_fp: tuple[float, float] = (0.05, 0.30)
    gap_after_pw: tuple[float, float] = (0.03, 0.25)
    gap_before_repair: tuple[float, float] = (0.05, 0.35)
    gap_repetition: tuple[float, float] = (0.15, 0.45)
    gap_reduplication: tuple[float, float] = (0.00, 0.06)

    # Tumpang tindih yang disengaja: proporsi kasus yang penanda
    # prosodinya menyesatkan, sehingga bukti leksikal tetap dibutuhkan.
    overlap_redup_sounds_disfluent: float = 0.15
    overlap_rep_sounds_fluent: float = 0.20

    utterance_lead: float = 0.25
    #: Senyap antargiliran bicara ketika ujaran digelar pada satu garis
    #: waktu berkelanjutan, meniru rekaman podcast yang utuh.
    inter_utterance_pause: tuple[float, float] = (0.35, 1.10)


def _uniform(rng: random.Random, span: tuple[float, float]) -> float:
    return rng.uniform(span[0], span[1])


def simulate_timings(
    words: Sequence[tuple[str, str]],
    rng: random.Random,
    profile: TimingProfile | None = None,
    t0: float | None = None,
) -> list[Token]:
    """Bangun token bertanda waktu dari daftar (kata, label)."""
    p = profile or TimingProfile()
    tokens: list[Token] = []
    t = p.utterance_lead if t0 is None else t0

    for i, (text, label) in enumerate(words):
        prev_label = words[i - 1][1] if i > 0 else None
        prev_text = words[i - 1][0] if i > 0 else None
        same_as_prev = (
            prev_text is not None
            and normalize_word(prev_text) == normalize_word(text)
        )

        # ---- jeda sebelum token ----
        if i == 0:
            gap = 0.0
        elif label == FP:
            gap = _uniform(rng, p.gap_before_fp)
        elif prev_label == FP:
            gap = _uniform(rng, p.gap_after_fp)
        elif prev_label == PW:
            gap = _uniform(rng, p.gap_after_pw)
        elif prev_label == RPR and label != RPR:
            gap = _uniform(rng, p.gap_before_repair)
        elif same_as_prev and prev_label == REP:
            # Repetisi disfluen: umumnya disela hesitasi, tetapi sebagian
            # diucapkan cepat sehingga terdengar seperti reduplikasi.
            if rng.random() < p.overlap_rep_sounds_fluent:
                gap = _uniform(rng, p.gap_reduplication)
            else:
                gap = _uniform(rng, p.gap_repetition)
        elif same_as_prev and prev_label == O and label == O:
            # Reduplikasi yang ditulis ASR sebagai dua token terpisah:
            # satu kata prosodi, tetapi sebagian penutur tetap menyisipkan
            # jeda sehingga terdengar seperti disfluensi.
            if rng.random() < p.overlap_redup_sounds_disfluent:
                gap = _uniform(rng, p.gap_repetition)
            else:
                gap = _uniform(rng, p.gap_reduplication)
        else:
            gap = _uniform(rng, p.fluent_gap)

        t += gap

        # ---- durasi token ----
        if label == FP:
            dur = _uniform(rng, p.fp_dur)
        elif label == PW:
            dur = _uniform(rng, p.pw_dur)
        else:
            base = p.dur_floor + p.sec_per_syllable * syllable_count(text)
            jitter = 1.0 + rng.uniform(-p.dur_jitter, p.dur_jitter)
            dur = base * jitter
            if label == REP:
                dur *= _uniform(rng, p.rep_shrink)
        dur = max(0.05, dur)

        tokens.append(Token(text=text, start=round(t, 3), end=round(t + dur, 3), label=label))
        t += dur

    return tokens


def neutral_timings(
    words: Sequence[tuple[str, str]],
    gap: float = 0.13,
    profile: TimingProfile | None = None,
) -> list[Token]:
    """Penanda waktu netral untuk masukan teks tanpa audio.

    Setiap jeda diberi nilai yang persis berada di tengah antara ambang
    reduplikasi dan ambang disfluensi, sehingga bukti prosodi menyumbang
    nol dan keputusan sepenuhnya bersandar pada bukti leksikal. Ini
    mencegah masukan tanpa audio diam-diam dinilai seolah punya bukti
    prosodi yang sebenarnya tidak ada.
    """
    p = profile or TimingProfile()
    tokens: list[Token] = []
    t = p.utterance_lead
    for i, (text, label) in enumerate(words):
        if i > 0:
            t += gap
        dur = max(0.05, p.dur_floor + p.sec_per_syllable * syllable_count(text))
        tokens.append(Token(text=text, start=round(t, 3), end=round(t + dur, 3), label=label))
        t += dur
    return tokens


# --------------------------------------------------------------------------
# Pembangunan korpus
# --------------------------------------------------------------------------


def build_corpus(
    path: Path | str = SEED_CORPUS,
    seed: int = 20260812,
    profile: TimingProfile | None = None,
    source: str = "seed-sintetis",
    continuous: bool = True,
) -> list[Utterance]:
    """Baca berkas anotasi lalu lekatkan penanda waktu sintetis.

    Dengan `continuous=True` seluruh ujaran digelar berurutan pada satu
    garis waktu, seperti satu berkas rekaman utuh. Ini yang membuat Edit
    Decision List lintas-ujaran bermakna: tanpa itu setiap ujaran mulai
    dari detik yang sama dan seluruh potongan saling tindih.

    Penataan ini tidak memengaruhi pemodelan. Seluruh fitur dihitung dari
    jeda dan durasi relatif di dalam satu ujaran, bukan dari waktu mutlak.
    """
    rows = read_annotated(path)
    rng = random.Random(seed)
    p = profile or TimingProfile()
    utterances: list[Utterance] = []
    cursor = 0.0

    for idx, words in enumerate(rows, start=1):
        t0 = cursor + p.utterance_lead if continuous else None
        tokens = simulate_timings(words, rng, p, t0=t0)
        if continuous and tokens:
            cursor = tokens[-1].end + _uniform(rng, p.inter_utterance_pause)
        utterances.append(
            Utterance(
                uid=f"seed-{idx:04d}",
                tokens=tokens,
                source=source,
                meta={
                    "timing": "sintetis",
                    "timing_seed": seed,
                    "peringatan": "penanda waktu bukan hasil pengukuran audio nyata",
                },
            )
        )
    return utterances


# --------------------------------------------------------------------------
# Pembagian data
# --------------------------------------------------------------------------


def _bucket(uid: str, mod: int) -> int:
    """Ember deterministik dari uid; stabil lintas mesin dan sesi."""
    digest = hashlib.md5(uid.encode("utf-8")).hexdigest()
    return int(digest, 16) % mod


def split(
    utterances: Sequence[Utterance],
    test_ratio: float = 0.3,
) -> tuple[list[Utterance], list[Utterance]]:
    """Pisah latih/uji secara deterministik berdasarkan hash uid."""
    cut = int(round(test_ratio * 100))
    train, test = [], []
    for utt in utterances:
        (test if _bucket(utt.uid, 100) < cut else train).append(utt)
    return train, test


def kfold(
    utterances: Sequence[Utterance],
    k: int = 5,
) -> Iterator[tuple[list[Utterance], list[Utterance]]]:
    """Validasi silang k-lipat deterministik.

    Dipakai karena korpus benih terlalu kecil untuk satu kali pisah
    latih/uji: satu partisi tunggal membuat metrik sangat sensitif
    terhadap kebetulan pembagian.
    """
    folds: list[list[Utterance]] = [[] for _ in range(k)]
    for utt in utterances:
        folds[_bucket(utt.uid, k)].append(utt)
    for i in range(k):
        test = folds[i]
        train = [u for j, f in enumerate(folds) if j != i for u in f]
        yield train, test


# --------------------------------------------------------------------------
# Ringkasan
# --------------------------------------------------------------------------


def label_counts(utterances: Iterable[Utterance]) -> dict[str, int]:
    counts = {lab: 0 for lab in LABELS}
    for utt in utterances:
        for tok in utt:
            counts[tok.label] += 1
    return counts


def describe(utterances: Sequence[Utterance]) -> dict:
    counts = label_counts(utterances)
    n_tokens = sum(counts.values())
    disfluent = n_tokens - counts[O]
    return {
        "jumlah_ujaran": len(utterances),
        "jumlah_token": n_tokens,
        "token_disfluen": disfluent,
        "rasio_disfluen": round(disfluent / n_tokens, 4) if n_tokens else 0.0,
        "durasi_total_detik": round(sum(u.end - u.start for u in utterances), 2),
        "distribusi_label": counts,
    }
