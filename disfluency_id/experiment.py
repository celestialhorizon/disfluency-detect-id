"""Eksperimen ujung ke ujung: perbandingan sistem dan ablasi fitur.

Rancangan eksperimen mengikuti apa yang akan dijalankan pada penelitian
sesungguhnya, hanya dengan korpus yang jauh lebih kecil:

Perbandingan sistem
    naif        memindahkan asumsi Bahasa Inggris apa adanya
    aturan      aturan sadar-reduplikasi tanpa pelatihan
    perceptron  penanda sekuens terpelajar

Ablasi fitur pada perceptron
    Menyalakan dan mematikan rumpun fitur untuk menguji apakah bukti
    prosodi dan bukti reduplikasi menyumbang di luar bukti leksikal.
    Inilah yang menjawab hipotesis penelitian secara langsung.

Validasi silang k-lipat dipakai, bukan satu kali pisah latih/uji, karena
korpus benih terlalu kecil sehingga satu partisi tunggal membuat metrik
sangat bergantung pada kebetulan pembagian.

BATAS TAFSIR. Seluruh angka di sini dihitung atas korpus benih tulisan
tangan dengan penanda waktu sintetis. Angka tersebut membuktikan
pipeline berjalan dan rancangan eksperimen dapat dieksekusi; ia BUKAN
temuan empiris tentang ucapan Bahasa Indonesia. Uji signifikansi
sengaja tidak dijalankan karena pada data sintetis hasilnya akan
menyesatkan; pada korpus nyata nanti dipakai uji McNemar berpasangan
untuk selisih akurasi token dan bootstrap berpasangan untuk selisih F1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .baseline import NaiveDetector, RuleBasedDetector
from .corpus import SEED_CORPUS, build_corpus, describe, kfold
from .edl import EDLConfig, build_edl, write_outputs
from .evaluate import (
    EvalResult,
    error_samples,
    evaluate,
    format_comparison,
    format_confusion,
    format_report,
    merge,
)
from .features import FeatureConfig
from .lexicon import Lexicon
from .model import DisfluencyTagger
from .schema import CONSERVATIVE_CUT, Utterance

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


@dataclass
class SystemSpec:
    """Satu sistem yang diuji: nama, cara membangun, perlu latih atau tidak."""

    name: str
    build: Callable[[Lexicon], object]
    trainable: bool
    note: str = ""


def default_systems() -> list[SystemSpec]:
    return [
        SystemSpec(
            "naif (asumsi Bahasa Inggris)",
            lambda lex: NaiveDetector(lex),
            False,
            "Setiap kata identik berdampingan dianggap repetisi disfluen.",
        ),
        SystemSpec(
            "aturan (sadar reduplikasi)",
            lambda lex: RuleBasedDetector(lex),
            False,
            "Penjagaan konteks leksikal + pemisahan reduplikasi berbasis bukti.",
        ),
        SystemSpec(
            "perceptron [lex]",
            lambda lex: DisfluencyTagger(lex, FeatureConfig(True, False, False)),
            True,
            "Ablasi: hanya fitur leksikal.",
        ),
        SystemSpec(
            "perceptron [lex+prosodi]",
            lambda lex: DisfluencyTagger(lex, FeatureConfig(True, True, False)),
            True,
            "Ablasi: leksikal + prosodi, tanpa fitur reduplikasi.",
        ),
        SystemSpec(
            "perceptron [lex+reduplikasi]",
            lambda lex: DisfluencyTagger(lex, FeatureConfig(True, False, True)),
            True,
            "Ablasi: leksikal + reduplikasi, tanpa prosodi.",
        ),
        SystemSpec(
            "perceptron [prosodi+reduplikasi]",
            lambda lex: DisfluencyTagger(lex, FeatureConfig(False, True, True)),
            True,
            "Ablasi: tanpa fitur leksikal sama sekali.",
        ),
        SystemSpec(
            "perceptron [penuh]",
            lambda lex: DisfluencyTagger(lex, FeatureConfig(True, True, True)),
            True,
            "Seluruh rumpun fitur dinyalakan.",
        ),
    ]


# --------------------------------------------------------------------------
# Menjalankan satu sistem
# --------------------------------------------------------------------------


def run_system(
    spec: SystemSpec,
    utterances: Sequence[Utterance],
    lex: Lexicon,
    k: int = 5,
    epochs: int = 15,
    seed: int = 13,
) -> tuple[EvalResult, dict[str, list[str]]]:
    """Jalankan validasi silang k-lipat untuk satu sistem."""
    fold_results: list[EvalResult] = []
    preds: dict[str, list[str]] = {}

    for train, test in kfold(utterances, k=k):
        if not test:
            continue
        detector = spec.build(lex)
        if spec.trainable:
            detector.fit(train, epochs=epochs, seed=seed)  # type: ignore[attr-defined]
        fold_preds = [detector.predict(u) for u in test]  # type: ignore[attr-defined]
        for utt, p in zip(test, fold_preds):
            preds[utt.uid] = p
        fold_results.append(evaluate(test, fold_preds, lex, system=spec.name))

    return merge(fold_results, spec.name), preds


# --------------------------------------------------------------------------
# Eksperimen penuh
# --------------------------------------------------------------------------


def run_experiment(
    corpus_path: Path | str = SEED_CORPUS,
    outdir: Path | str = REPORTS_DIR,
    k: int = 5,
    epochs: int = 15,
    seed: int = 13,
    timing_seed: int = 20260812,
    systems: list[SystemSpec] | None = None,
    verbose: bool = True,
) -> dict:
    """Jalankan seluruh perbandingan dan tulis laporan ke `outdir`."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lex = Lexicon.load()
    utterances = build_corpus(corpus_path, seed=timing_seed)
    stats = describe(utterances)
    specs = systems or default_systems()

    if verbose:
        print(f"Korpus: {stats['jumlah_ujaran']} ujaran, {stats['jumlah_token']} token")
        print(f"Token disfluen: {stats['token_disfluen']} ({stats['rasio_disfluen']:.1%})")
        print()

    results: list[EvalResult] = []
    all_preds: dict[str, dict[str, list[str]]] = {}

    for spec in specs:
        res, preds = run_system(spec, utterances, lex, k=k, epochs=epochs, seed=seed)
        results.append(res)
        all_preds[spec.name] = preds
        if verbose:
            _, _, f1 = res.micro
            print(
                f"  {spec.name:<34} F1mikro={f1:.4f}  "
                f"akurasi={res.accuracy:.4f}  "
                f"pelestarian_reduplikasi={res.redup_preservation:.4f}"
            )

    # -- analisis galat memakai sistem penuh ------------------------------
    best_name = specs[-1].name
    for utt in utterances:
        utt.assign(all_preds[best_name][utt.uid])
    errors = error_samples(utterances, limit=25)
    redup_errors = error_samples(utterances, limit=15, focus="O")

    # -- demo EDL ---------------------------------------------------------
    demo = [u for u in utterances if any(t.pred != t.label for t in u)][:3]
    if not demo:
        demo = list(utterances[:3])
    edl = build_edl(
        demo,
        EDLConfig(cut_labels=CONSERVATIVE_CUT),
        use_pred=True,
        source="contoh_podcast.wav",
    )
    edl_paths = write_outputs(edl, outdir / "edl_demo", stem="demo")

    payload = {
        "korpus": stats,
        "pengaturan": {
            "lipatan_validasi_silang": k,
            "epoch": epochs,
            "seed_model": seed,
            "seed_penanda_waktu": timing_seed,
        },
        "peringatan": (
            "Angka dihitung atas korpus benih tulisan tangan dengan penanda waktu "
            "sintetis. Sahih sebagai bukti keberjalanan pipeline, TIDAK sahih "
            "sebagai temuan empiris tentang ucapan Bahasa Indonesia."
        ),
        "hasil": [r.to_dict() for r in results],
        "catatan_sistem": {s.name: s.note for s in specs},
        "contoh_galat": errors,
        "galat_pada_token_fluen": redup_errors,
        "edl_demo": edl.stats(),
    }

    (outdir / "hasil_eksperimen.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "hasil_eksperimen.md").write_text(
        _render_markdown(stats, specs, results, errors, redup_errors, edl, k, epochs),
        encoding="utf-8",
    )

    if verbose:
        print()
        print(f"Laporan  : {outdir / 'hasil_eksperimen.md'}")
        print(f"Data JSON: {outdir / 'hasil_eksperimen.json'}")
        print(f"EDL demo : {edl_paths['json']}")

    return payload


# --------------------------------------------------------------------------
# Laporan Markdown
# --------------------------------------------------------------------------


def _render_markdown(
    stats: dict,
    specs: list[SystemSpec],
    results: list[EvalResult],
    errors: list[dict],
    redup_errors: list[dict],
    edl,
    k: int,
    epochs: int,
) -> str:
    lines = [
        "# Hasil Eksperimen Prototipe",
        "",
        "> **Batas tafsir.** Seluruh angka pada laporan ini dihitung atas korpus",
        "> benih yang ditulis tangan peneliti, dengan penanda waktu tingkat kata",
        "> yang disintesis, bukan diukur dari audio. Angka ini membuktikan bahwa",
        "> pipeline berjalan utuh dan rancangan eksperimen dapat dieksekusi. Ia",
        "> **bukan** temuan empiris mengenai ucapan Bahasa Indonesia dan tidak",
        "> boleh dilaporkan sebagai hasil penelitian.",
        "",
        "## 1. Korpus",
        "",
        f"- Jumlah ujaran: {stats['jumlah_ujaran']}",
        f"- Jumlah token: {stats['jumlah_token']}",
        f"- Token disfluen: {stats['token_disfluen']} ({stats['rasio_disfluen']:.2%})",
        f"- Total durasi sintetis: {stats['durasi_total_detik']} detik",
        "",
        "| Label | Jumlah token |",
        "|---|---:|",
    ]
    for lab, c in stats["distribusi_label"].items():
        lines.append(f"| {lab} | {c} |")

    lines += [
        "",
        "## 2. Pengaturan eksperimen",
        "",
        f"- Validasi silang {k} lipatan, pembagian deterministik berbasis hash uid",
        f"- Perceptron terstruktur, {epochs} epoch, bobot terata-rata",
        "- Agregasi antarlipatan dilakukan pada tingkat tp/fp/fn (mikro), bukan",
        "  rerata F1, agar lipatan kecil tidak memberi bobot berlebih",
        "",
        "## 3. Perbandingan sistem",
        "",
        format_comparison(results),
        "",
        "Keterangan sistem:",
        "",
    ]
    for s in specs:
        lines.append(f"- **{s.name}** — {s.note}")

    lines += [
        "",
        "### Metrik pelestarian reduplikasi",
        "",
        "Kolom terakhir tabel di atas adalah metrik yang paling menentukan bagi",
        "masalah ini. Memotong reduplikasi gramatikal mengubah makna kalimat",
        "(`anak-anak` menjadi `anak`), sedangkan menyisakan satu filler hanya",
        "menyisakan kekasaran gaya. Kedua kesalahan itu berbobot sama pada F1",
        "agregat, sehingga metrik terpisah diperlukan agar perbedaan biayanya",
        "terlihat.",
        "",
        "## 4. Rincian per sistem",
        "",
    ]
    for r in results:
        lines.append(format_report(r))
        lines.append("<details><summary>Matriks kekeliruan</summary>")
        lines.append("")
        lines.append(format_confusion(r))
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines += [
        "## 5. Analisis galat (sistem penuh)",
        "",
        "| uid | token | acuan | prediksi | jeda sebelum | konteks |",
        "|---|---|---|---|---:|---|",
    ]
    for e in errors:
        lines.append(
            f"| {e['uid']} | `{e['token']}` | {e['gold']} | {e['predicted']} | "
            f"{e['gap_before']:.3f} | {e['context']} |"
        )
    if not errors:
        lines.append("| - | - | - | - | - | tidak ada galat pada lipatan uji |")

    lines += [
        "",
        "### Galat pada token fluen (pemotongan yang merusak)",
        "",
        "| uid | token | prediksi | jeda sebelum | konteks |",
        "|---|---|---|---:|---|",
    ]
    for e in redup_errors:
        lines.append(
            f"| {e['uid']} | `{e['token']}` | {e['predicted']} | "
            f"{e['gap_before']:.3f} | {e['context']} |"
        )
    if not redup_errors:
        lines.append("| - | - | - | - | tidak ada token fluen yang keliru dipotong |")

    st = edl.stats()
    lines += [
        "",
        "## 6. Contoh keluaran Edit Decision List",
        "",
        f"- Durasi asli: {st['durasi_asli_detik']} detik",
        f"- Durasi setelah pemotongan: {st['durasi_akhir_detik']} detik",
        f"- Dipangkas: {st['durasi_dipotong_detik']} detik ({st['persen_pemangkasan']}%)",
        f"- Jumlah potongan: {st['jumlah_potongan']}, segmen dipertahankan: {st['jumlah_segmen_disimpan']}",
        "",
        "Transkrip sebelum:",
        "",
        "```",
        edl.transcript_before,
        "```",
        "",
        "Transkrip sesudah:",
        "",
        "```",
        edl.transcript_after,
        "```",
        "",
        "Berkas lengkap (JSON EDL, CSV potongan, skrip ffmpeg, transkrip) ada di",
        "`reports/edl_demo/`.",
        "",
        "## 7. Rencana uji statistik pada data sungguhan",
        "",
        "Uji signifikansi sengaja tidak dijalankan di sini karena pada data",
        "sintetis hasilnya menyesatkan: selisih antarsistem sebagian ditentukan",
        "oleh simulator penanda waktu, bukan oleh fenomena kebahasaan. Pada",
        "korpus ucapan sungguhan nanti dipakai:",
        "",
        "- **Uji McNemar berpasangan** untuk selisih akurasi token antardua",
        "  sistem pada data uji yang sama;",
        "- **Bootstrap berpasangan** (10.000 resampel pada tingkat ujaran) untuk",
        "  selang kepercayaan selisih F1 dan selisih laju potong salah;",
        "- **Koefisien Cohen kappa** antaranotator untuk menakar keterandalan",
        "  anotasi sebelum metrik apa pun dilaporkan.",
        "",
    ]
    return "\n".join(lines)
