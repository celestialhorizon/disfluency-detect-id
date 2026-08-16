"""Transkripsi verbatim dengan CrisperWhisper.

Satu-satunya jalur ASR proyek ini. `nyrahealth/CrisperWhisper` dipilih karena
ia dilatih **verbatim**: jeda terisi (`uhm`, `eee`) ikut dituliskan. Model
Whisper arus utama dilatih dari subtitle web yang sudah dirapikan penyubtitle,
sehingga ia belajar bahwa jeda terisi "bukan teks" dan menghapusnya — yakni
menghapus persis objek penelitian ini.

Rangkaiannya:

    media (mp4/mkv/wav/...)
        -> ke_wav             ekstrak WAV 16 kHz mono lewat ffmpeg
        -> pipeline HF        kata + penanda waktu DTW
        -> cari_loop          buang rentang degenerasi, CATAT lubangnya
        -> profil_senyap      ukur senyap langsung dari gelombang
        -> kikis_ke_bunyi     kikis senyap dari ujung tiap kata
        -> bikin_segmen       potong jadi segmen di jeda >= 0,8 dtk
        -> JSON               siap dibaca `from_whisper_json`

Empat fungsi di tengah rangkaian itu memakai pustaka standar Python saja,
sehingga dapat diuji tanpa GPU maupun bobot model. Hanya `transcribe` yang
menuntut `torch` dan `transformers`.

BATAS YANG TIDAK BOLEH DILANGGAR. `repetition_penalty` dan
`no_repeat_ngram_size` **tidak pernah** disetel di sini. Keduanya menghukum
pengulangan, padahal pengulangan adalah fenomena yang diteliti (`lu lu`,
`tiba tiba`). Menyetelnya sama dengan menghapus data sendiri lalu melaporkan
datanya tidak ada. Degenerasi ditangani SESUDAH bangkitan oleh `cari_loop`,
yang ambangnya dipilih supaya pengulangan sah tetap lolos.

CATATAN ETIKA. Berlaku pada rekaman siapa pun selain peneliti sendiri:
gunakan hanya konten publik dan hormati ketentuan layanan platform; simpan
audio mentah hanya selama anotasi dan jangan diredistribusi; samarkan nama
diri pada transkrip yang dipublikasikan; catat lisensi tiap sumber pada
berkas provenans yang menyertai korpus.
"""

from __future__ import annotations

import array
import json
import math
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

#: Bobot model. Tokenizer dan generation_config diambil dari model dasar
#: karena `added_tokens.json` CrisperWhisper tidak tergabung di transformers
#: arus utama; tanpa itu bangkitan keluar KOSONG dan meledak belakangan
#: sebagai "size of tensor a (2) must match tensor b (0)". Arsitekturnya
#: identik, jadi penukaran ini aman.
MODEL_ID = "nyrahealth/CrisperWhisper"
BASIS_ID = "openai/whisper-large-v3"

#: Jendela 30 dtk dipilih setelah 30/20/15 diadu pada rekaman uji: ketiganya
#: muat di VRAM dan ketiganya degenerasi, tetapi 30 dtk yang rentang rusaknya
#: paling pendek.
JENDELA_BAWAAN = 30

#: BUKAN penghematan waktu melainkan syarat muat. `return_timestamps="word"`
#: menahan cross-attention 32 lapisan dekoder untuk TIAP langkah dekode. Tanpa
#: batas, bangkitan lari sampai 448 langkah (degenerasi) dan puncaknya 13,71 GB
#: -> OOM pada T4 yang hanya 14,56 GB. Dengan batas 160, puncaknya 10,89 GB.
#: Diukur pada kernel bersih, bukan ditaksir. 30 dtk ucapan spontan kira-kira
#: 90 kata kira-kira 130 token, jadi 160 masih lapang.
MAKS_TOKEN_BAWAAN = 160

ASR_HINT = (
    "Pasang lebih dulu:\n"
    "    pip install transformers torch\n"
    "Perlu GPU. Pada CPU, transkripsi CrisperWhisper tidak praktis."
)


# --------------------------------------------------------------------------
# Praproses audio
# --------------------------------------------------------------------------

def ke_wav(sumber: Path | str, tujuan_dir: Path | str | None = None) -> Path:
    """Ekstrak trek audio jadi WAV 16 kHz mono.

    Pipeline HF **tidak bisa** disodori `.mp4` langsung: ia menyerah dengan
    "Soundfile is either not in the correct format or is malformed", yang
    menyesatkan sebab berkasnya sebetulnya sehat.
    """
    sumber = Path(sumber)
    tujuan_dir = Path(tujuan_dir) if tujuan_dir else sumber.parent
    tujuan_dir.mkdir(parents=True, exist_ok=True)
    tujuan = tujuan_dir / f"{sumber.stem}.16k.wav"
    if not tujuan.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(sumber),
             "-ac", "1", "-ar", "16000", str(tujuan)],
            check=True,
        )
    return tujuan


def profil_senyap(
    wav_path: Path | str, hop: float = 0.01, ambang_relatif: float = 3.0
) -> tuple[list[bool], float]:
    """Kembalikan (senyap, hop): satu bool per bingkai `hop`, True = senyap.

    Ambangnya **relatif terhadap lantai derau berkas itu sendiri** (persentil
    10 dikali `ambang_relatif`), bukan angka mutlak. Rekaman lapangan tingkat
    deraunya berbeda-beda, dan ambang tetap pasti salah pada salah satunya.
    """
    with wave.open(str(wav_path)) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(
                f"{wav_path}: butuh WAV 16-bit mono, dapat "
                f"{w.getsampwidth() * 8}-bit {w.getnchannels()} kanal"
            )
        sr, n = w.getframerate(), w.getnframes()
        smp = array.array("h", w.readframes(n))
    H = int(sr * hop)
    if H <= 0 or len(smp) < H:
        return [], hop
    rms = [
        math.sqrt(sum(x * x for x in smp[i:i + H]) / H)
        for i in range(0, len(smp) - H, H)
    ]
    if not rms:
        return [], hop
    urut = sorted(rms)
    lantai = urut[len(urut) // 10] or 1.0
    ambang = max(lantai * ambang_relatif, urut[-1] * 0.02)
    return [r < ambang for r in rms], hop


def kikis_ke_bunyi(
    kata: list[dict], senyap: list[bool], hop: float, min_sisa: float = 0.04
) -> tuple[list[dict], int]:
    """Kikis senyap dari ujung depan dan belakang tiap kata.

    Ini bukan mengarang jeda: jedanya memang ada di audio, hanya tidak
    dilaporkan model. Dan ini bukan pengganti forced alignment — yang
    dikoreksi hanya batas LUAR tiap kata; batas di dalam rentetan kata
    bersambung tetap tebakan DTW.

    Bingkai ke-k mencakup [k*hop, (k+1)*hop). Kata bersebelahan berbagi
    bingkai batas, jadi menulis akhir sebagai (q+1)*hop membuat kata ke-i
    menabrak awal kata ke-i+1 sebanyak satu bingkai. Jedanya jadi -0,01 dtk,
    dan pemeriksa yang menjepit jeda negatif ke nol akan melaporkannya sebagai
    "jeda persis nol" — persis gejala yang hendak diperbaiki. Karena itu ada
    lintasan kedua yang memastikan akhir kata tidak pernah melewati awal kata
    berikutnya.

    Kata yang seluruhnya jatuh di senyap dibiarkan apa adanya: mengikisnya
    sampai nol menghasilkan token tanpa durasi, dan itu lebih menyesatkan
    daripada penanda waktu yang longgar.

    Mengembalikan (kata_terkoreksi, jumlah_yang_berubah).
    """
    n = len(senyap)
    hasil: list[dict] = []
    dikikis = 0
    for w in kata:
        a, b = w["timestamp"]
        ia, ib = int(a / hop), min(int(b / hop), n - 1)
        if n == 0 or ia >= n or ib <= ia:
            hasil.append(dict(w))
            continue
        p, q = ia, ib
        while p < q and senyap[p]:
            p += 1
        while q > p and senyap[q]:
            q -= 1
        na, nb = p * hop, (q + 1) * hop
        if nb - na < min_sisa or q <= p:
            hasil.append(dict(w))
            continue
        if abs(na - a) > 1e-9 or abs(nb - b) > 1e-9:
            dikikis += 1
        hasil.append({**w, "timestamp": (round(na, 3), round(nb, 3))})

    for i in range(len(hasil) - 1):
        a, b = hasil[i]["timestamp"]
        a2, _ = hasil[i + 1]["timestamp"]
        if b > a2:
            hasil[i]["timestamp"] = (a, round(max(a, a2), 3))
    return hasil, dikikis


# --------------------------------------------------------------------------
# Penanganan degenerasi bangkitan
# --------------------------------------------------------------------------

def cari_loop(
    kata: list[dict], n_maks: int = 4, min_putaran: int = 4
) -> list[dict]:
    """Temukan rentang yang jelas degenerasi bangkitan, bukan pengulangan sah.

    Ambang `min_putaran = 4` dipilih supaya `lu lu` dan `tiba tiba` (2
    putaran) **lolos** — itu data, bukan sampah — sementara putaran runaway
    26x tertangkap.

    Deteksi n-gram sampai `n_maks`, karena degenerasi tidak selalu satu kata:
    jendela 15 dtk pernah memuntahkan loop DUA kata (`ya kan. ya kan. ...`)
    yang lolos dari detektor satu-kata.

    Rentang yang tumpang-tindih DILEBUR. Tanpa peleburan, satu rentetan
    `k k k k ...` terlaporkan empat kali (n=1,2,3,4 semuanya cocok) dan durasi
    rusaknya terhitung empat kali lipat — 11,4 dtk padahal sesungguhnya 2,9.
    """
    tok = [w["text"].strip().lower() for w in kata]
    calon: list[tuple[int, int, int, int, str]] = []
    for i in range(len(tok)):
        for n in range(1, n_maks + 1):
            if i + n * min_putaran > len(tok):
                continue
            putaran = 1
            while (i + (putaran + 1) * n <= len(tok)
                   and tok[i + putaran * n: i + (putaran + 1) * n] == tok[i: i + n]):
                putaran += 1
            if putaran >= min_putaran:
                calon.append((i, i + putaran * n, putaran, n, " ".join(tok[i:i + n])))
    if not calon:
        return []

    calon.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    lebur: list[dict] = []
    for c in calon:
        if lebur and c[0] < lebur[-1]["henti_idx"]:
            k = lebur[-1]
            k["henti_idx"] = max(k["henti_idx"], c[1])
            if (c[1] - c[0]) > k["_lebar"]:
                k.update(_lebar=c[1] - c[0], putaran=c[2], n=c[3], isi=c[4])
        else:
            lebur.append({"mulai_idx": c[0], "henti_idx": c[1],
                          "_lebar": c[1] - c[0], "putaran": c[2],
                          "n": c[3], "isi": c[4]})
    for k in lebur:
        k.pop("_lebar")
    return lebur


def bikin_segmen(kata: list[dict], jeda_pisah: float = 0.8) -> list[dict]:
    """Kata -> segmen, dipotong di jeda >= `jeda_pisah`.

    Ambangnya disamakan dengan `max_gap` milik `from_whisper_json` supaya
    segmen di transkrip dan ujaran di korpus jatuh di tempat yang sama. Kalau
    berbeda, transkrip dan korpus akan bercerita hal berlainan tentang berkas
    yang sama.
    """
    segmen: list[dict] = []
    kini: list[dict] = []
    for i, w in enumerate(kata):
        kini.append(w)
        putus = i == len(kata) - 1
        if not putus and kata[i + 1]["timestamp"][0] - w["timestamp"][1] >= jeda_pisah:
            putus = True
        if putus:
            segmen.append({
                "start": kini[0]["timestamp"][0],
                "end": kini[-1]["timestamp"][1],
                "text": "".join(x["text"] for x in kini),
                "words": [{"word": x["text"], "start": x["timestamp"][0],
                           "end": x["timestamp"][1]} for x in kini],
            })
            kini = []
    return segmen


# --------------------------------------------------------------------------
# Transkripsi
# --------------------------------------------------------------------------

def transcribe(
    audio_path: Path | str,
    out_json: Path | str | None = None,
    language: str = "id",
    jendela: int = JENDELA_BAWAAN,
    maks_token: int = MAKS_TOKEN_BAWAAN,
    koreksi_senyap: bool = True,
    workdir: Path | str | None = None,
    progress: bool = True,
) -> Path:
    """Transkripsikan berkas media dengan penanda waktu tingkat kata.

    Menerima berkas audio maupun video; trek audio diekstrak lebih dulu.

    Mengembalikan lokasi berkas JSON yang siap dibaca `from_whisper_json`.
    Berkas itu memuat medan `rentang_rusak`: rentang degenerasi yang katanya
    dibuang. Membiarkannya masuk akan mencemari korpus dengan token palsu;
    membuangnya diam-diam akan menyembunyikan bahwa ada audio tanpa transkrip
    sah. Anotator harus tahu di mana lubangnya.
    """
    try:
        import torch
        from transformers import (AutoModelForSpeechSeq2Seq, AutoProcessor,
                                  GenerationConfig, pipeline)
    except ImportError as exc:
        raise RuntimeError(f"transformers/torch tidak tersedia.\n{ASR_HINT}") from exc

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"berkas tidak ditemukan: {audio_path}")

    peranti = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(peranti)
    prosesor = AutoProcessor.from_pretrained(BASIS_ID)
    mdl.generation_config = GenerationConfig.from_pretrained(BASIS_ID)

    n_tok = len(prosesor.tokenizer)
    n_emb = mdl.get_input_embeddings().weight.shape[0]
    if n_tok != n_emb:
        raise RuntimeError(
            f"tokenizer {n_tok} != embedding {n_emb} -- bangkitan akan kosong"
        )

    pipa = pipeline(
        "automatic-speech-recognition",
        model=mdl, tokenizer=prosesor.tokenizer,
        feature_extractor=prosesor.feature_extractor,
        chunk_length_s=jendela, stride_length_s=(jendela // 6, jendela // 6),
        batch_size=1, return_timestamps="word", dtype=dtype, device=peranti,
    )

    wav = ke_wav(audio_path, workdir)
    mulai = time.monotonic()
    keluar = pipa(str(wav), generate_kwargs={
        "language": language, "task": "transcribe", "max_new_tokens": maks_token})
    lama = time.monotonic() - mulai

    kata = [w for w in keluar.get("chunks", []) if None not in w["timestamp"]]
    rusak = cari_loop(kata)
    buang = {i for r in rusak for i in range(r["mulai_idx"], r["henti_idx"])}
    rentang_rusak = [
        {"mulai": kata[r["mulai_idx"]]["timestamp"][0],
         "henti": kata[r["henti_idx"] - 1]["timestamp"][1],
         "putaran": r["putaran"], "isi": r["isi"]}
        for r in rusak
    ]
    bersih = [w for i, w in enumerate(kata) if i not in buang]

    dikikis = 0
    if koreksi_senyap and bersih:
        senyap, hop = profil_senyap(wav)
        if senyap:
            bersih, dikikis = kikis_ke_bunyi(bersih, senyap, hop)

    segmen = bikin_segmen(bersih)
    durasi = max((w["timestamp"][1] for w in kata), default=0.0)

    payload: dict[str, Any] = {
        "bahasa": language,
        "durasi": durasi,
        "model": MODEL_ID,
        "setelan": {
            "jendela_dtk": jendela,
            "maks_token": maks_token,
            "tokenizer_dari": BASIS_ID,
            "koreksi_senyap": koreksi_senyap,
            # Dicatat eksplisit sebagai None supaya terbaca sebagai keputusan
            # rancangan, bukan kelalaian. Lihat docstring modul.
            "repetition_penalty": None,
            "no_repeat_ngram_size": None,
        },
        "rentang_rusak": rentang_rusak,
        "segments": segmen,
    }

    out_json = Path(out_json) if out_json else audio_path.with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    if progress:
        jeda = [b["timestamp"][0] - a["timestamp"][1]
                for a, b in zip(bersih, bersih[1:])]
        nol = sum(1 for g in jeda if abs(g) < 1e-6)
        hilang = sum(r["henti"] - r["mulai"] for r in rentang_rusak)
        sys.stderr.write(
            f"  {lama:.0f} dtk untuk {durasi:.0f} dtk audio pada {peranti}\n"
            f"  {len(segmen)} segmen, {len(bersih)} kata"
            f" | dikikis {dikikis}"
            f" | jeda nol {nol}/{max(len(jeda), 1)}"
            f" ({nol / max(len(jeda), 1):.1%})\n"
        )
        for r in rentang_rusak:
            sys.stderr.write(
                f"  !! degenerasi dibuang {r['mulai']:.2f}-{r['henti']:.2f} dtk"
                f" ({r['putaran']}x {r['isi']!r})\n"
            )
        if rentang_rusak and durasi:
            sys.stderr.write(
                f"  !! total {hilang:.1f} dtk ({hilang / durasi:.1%})"
                f" audio tanpa transkrip sah\n"
            )
        sys.stderr.flush()

    return out_json
