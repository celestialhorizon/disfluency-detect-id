"""Pengambilan data nyata: dari audio ke ujaran bertanda waktu.

Prototipe inti sengaja tidak bergantung pada pustaka apa pun agar selalu
dapat dijalankan. Modul ini adalah jembatan opsional menuju data
sungguhan, dan merupakan jalur yang akan dipakai pada penelitian penuh:

    berkas lokal di data/media/
        -> asr.transcribe    CrisperWhisper, penanda waktu tingkat kata
        -> from_whisper_json konversi ke objek Utterance
        -> anotasi manual    pemberian label disfluensi oleh dua anotator
        -> korpus JSONL      masukan bagi seluruh pipeline lainnya

Transkripsinya ada di `disfluency_id/asr.py`, dan **hanya** CrisperWhisper.
Alasannya ada di docstring modul itu: model Whisper arus utama menghapus jeda
terisi, yakni objek penelitian ini.

Modul ini TIDAK mengunduh apa pun. Satu-satunya jalur pengunduhan ada di
`disfluency_id/media.py`, dan ia hanya mau menyentuh berkas yang sudah punya
baris di `data/media/provenans.csv`. Pemisahan itu yang membuat syarat
"provenans diisi sebelum berkas diproses" berlaku dengan sendirinya: tidak ada
pintu lain yang bisa memasukkan rekaman tanpa provenans.

Pustaka berat (transformers, torch) TIDAK wajib dipasang. Bila belum ada,
fungsi yang membutuhkannya memberi pesan pemasangan yang jelas alih-alih gagal
dengan ImportError yang membingungkan. Jalur `ingest --whisper-json` — yang
dipakai sehari-hari — sama sekali tidak menyentuhnya.

CATATAN ETIKA. Rekaman yang diproses di sini menyentuh konten milik orang
lain. Sebelum dijalankan pada penelitian sesungguhnya:

- gunakan hanya konten yang tersedia untuk publik, dan hormati ketentuan
  layanan serta berkas robots dari platform sumber;
- simpan audio mentah hanya selama proses anotasi berlangsung, dan jangan
  redistribusikan; yang dipublikasikan cukup transkrip beranotasi beserta
  penanda waktu dan tautan sumber;
- samarkan nama diri dan penanda identitas lain pada transkrip yang
  dipublikasikan;
- catat lisensi tiap sumber dalam berkas provenans yang menyertai korpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .asr import (ASR_HINT, JENDELA_BAWAAN, MAKS_TOKEN_BAWAAN, MODEL_ID,
                  transcribe)
from .schema import O, Token, Utterance, normalize_word

def check_dependencies() -> dict[str, bool]:
    """Laporkan pustaka opsional mana yang tersedia di lingkungan ini."""
    status: dict[str, bool] = {}
    for name in ("transformers", "torch"):
        try:
            __import__(name)
            status[name] = True
        except ImportError:
            status[name] = False
    return status


# --------------------------------------------------------------------------
# Konversi keluaran Whisper
# --------------------------------------------------------------------------


def _iter_words(payload: Any) -> Iterable[dict]:
    """Ambil daftar kata bertanda waktu dari beberapa bentuk keluaran Whisper.

    Ditangani tiga bentuk: keluaran openai-whisper (`segments[].words[]`),
    keluaran yang sudah diratakan (`words[]`), dan daftar kata polos.
    """
    if isinstance(payload, list):
        yield from payload
        return
    if not isinstance(payload, dict):
        return
    if "segments" in payload:
        for seg in payload["segments"]:
            for w in seg.get("words") or []:
                yield w
        return
    if "words" in payload:
        yield from payload["words"]


def merge_hyphen_continuations(tokens: list[Token]) -> list[Token]:
    """Satukan kembali kata bertanda hubung yang dipecah ASR.

    Whisper memecah 'kira-kira' menjadi dua token, ' kira' dan '-kira'.
    Bagi bahasa yang menandai reduplikasi dengan tanda hubung, kerusakan
    ini serius dan berlipat:

    - satu kata terhitung dua token, sehingga setiap metrik berbasis laju
      memakai penyebut yang salah;
    - batas semu di tengah kata dilaporkan berjeda 0,000 detik, dan angka
      itu lalu terbaca sebagai bukti prosodi 'satu satuan ucap' padahal
      tidak ada jeda yang pernah diukur di sana;
    - jalur yang seharusnya menangani kasus ini, `is_hyphenated_redup`,
      tidak pernah aktif karena tak satu pun token memuat bentuk utuhnya;
    - anotator manusia menerima korpus yang satu katanya terbelah dua.

    Hanya tanda hubung di AWAL token yang disatukan. Tanda hubung di
    akhir sengaja dibiarkan: itu justru lambang kata terpotong (PW) pada
    pedoman anotasi, dan menyatukannya akan menghapus fenomena yang
    sedang diteliti.
    """
    if not tokens:
        return tokens
    out = [tokens[0]]
    for tok in tokens[1:]:
        text = tok.text.strip()
        prev = out[-1]
        if text.startswith("-") and len(text) > 1 and not prev.text.strip().endswith("-"):
            out[-1] = Token(
                text=prev.text.strip() + text,
                start=prev.start,
                end=max(prev.end, tok.end),
                label=prev.label,
            )
        else:
            out.append(tok)
    return out


def from_whisper_json(
    path: Path | str,
    max_gap: float = 0.8,
    uid_prefix: str = "asr",
    source: str | None = None,
    merge_hyphens: bool = True,
) -> list[Utterance]:
    """Ubah keluaran Whisper bertanda waktu kata menjadi daftar Utterance.

    Kata dipecah menjadi ujaran terpisah setiap kali senyap melebihi
    `max_gap`. Ambang 0,8 detik dipilih karena cukup panjang untuk tidak
    memotong hesitasi di tengah kalimat, tetapi cukup pendek untuk
    memisahkan giliran bicara.

    Whisper WAJIB dijalankan dengan `word_timestamps=True`; tanpa penanda
    waktu tingkat kata seluruh fitur prosodi tidak dapat dihitung.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    words = [
        w
        for w in _iter_words(payload)
        if w.get("word") is not None or w.get("text") is not None
    ]
    if not words:
        raise ValueError(
            f"tidak ada kata bertanda waktu di {path}. "
            "Jalankan Whisper dengan word_timestamps=True."
        )

    utterances: list[Utterance] = []
    current: list[Token] = []
    prev_end: float | None = None

    def flush() -> None:
        if not current:
            return
        toks = merge_hyphen_continuations(list(current)) if merge_hyphens else list(current)
        utterances.append(
            Utterance(
                uid=f"{uid_prefix}-{len(utterances) + 1:04d}",
                tokens=toks,
                source=source or path.stem,
                meta={"timing": "asr", "asal": str(path)},
            )
        )
        current.clear()

    for w in words:
        text = (w.get("word") or w.get("text") or "").strip()
        if not text or not normalize_word(text):
            continue
        start = float(w["start"])
        end = float(w["end"])
        if end < start:
            continue
        if prev_end is not None and start - prev_end > max_gap:
            flush()
        current.append(Token(text=text, start=start, end=end, label=O))
        prev_end = end

    flush()
    return utterances


# --------------------------------------------------------------------------
# Transkripsi berkas lokal
# --------------------------------------------------------------------------


def ingest_media(
    media_path: Path | str,
    language: str = "id",
    out_json: Path | str | None = None,
    max_gap: float = 0.8,
    merge_hyphens: bool = True,
    jendela: int = JENDELA_BAWAAN,
    maks_token: int = MAKS_TOKEN_BAWAAN,
    koreksi_senyap: bool = True,
    progress: bool = True,
) -> list[Utterance]:
    """Transkripsikan berkas video/audio lokal lalu ubah jadi Utterance.

    Ini jalur bagi rekaman yang sudah ada di cakram: tidak ada pengunduhan,
    sehingga tidak menyentuh konten pihak lain. Kewajiban provenans tetap
    berlaku bila rekaman itu bukan milik peneliti sendiri.

    JSON transkrip sengaja ikut disimpan. Transkripsi adalah langkah termahal
    dalam rangkaian ini, dan menyimpan keluarannya membuat percobaan berulang
    tidak perlu mengulang ASR.
    """
    media_path = Path(media_path)
    if not media_path.exists():
        raise FileNotFoundError(f"berkas tidak ditemukan: {media_path}")

    js = transcribe(
        media_path,
        out_json=out_json,
        language=language,
        jendela=jendela,
        maks_token=maks_token,
        koreksi_senyap=koreksi_senyap,
        progress=progress,
    )
    return from_whisper_json(
        js, max_gap=max_gap, source=media_path.name, merge_hyphens=merge_hyphens
    )
