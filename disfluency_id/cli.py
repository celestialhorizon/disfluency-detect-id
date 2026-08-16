"""Antarmuka baris perintah.

    python -m disfluency_id <perintah> [opsi]

Perintah yang tersedia:

    stats       ringkasan korpus benih
    build       bangun korpus JSONL bertanda waktu
    experiment  perbandingan sistem + ablasi fitur, tulis laporan
    train       latih model penuh lalu simpan
    detect      jalankan detektor pada teks atau berkas JSONL
    explain     tampilkan alasan putusan reduplikasi vs repetisi
    media       daftar / unduh rekaman yang tercatat di provenans.csv
    ingest      impor rekaman / transkrip ASR menjadi korpus JSONL
    audit       periksa mutu transkrip ASR sebelum dipakai
    edl         susun Edit Decision List dari berkas JSONL
    render      jalankan EDL menjadi berkas video lewat ffmpeg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .asr import JENDELA_BAWAAN, MAKS_TOKEN_BAWAAN
from .baseline import NaiveDetector, RuleBasedDetector
from .corpus import SEED_CORPUS, build_corpus, describe, neutral_timings
from .edl import EDLConfig, build_edl, write_outputs
from .evaluate import evaluate, format_report
from .render import (
    RenderConfig,
    ffmpeg_argv,
    durasi,
    extend_to_media,
    keep_spans,
    load_edl,
    probe_duration,
    proof_spans,
    render,
    resolve_source,
    find_font,
)
from .experiment import REPORTS_DIR, run_experiment
from .features import FeatureConfig
from .lexicon import Lexicon
from .model import DisfluencyTagger
from .reduplication import find_repeat_events
from .schema import (
    AGGRESSIVE_CUT,
    CONSERVATIVE_CUT,
    O,
    Utterance,
    iter_spans,
    read_jsonl,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "perceptron.json"
DEFAULT_JSONL = ROOT / "data" / "corpus" / "seed_id.jsonl"
MEDIA_DIR = ROOT / "data" / "media"
PROVENANS = MEDIA_DIR / "provenans.csv"
TRANSCRIPTS_DIR = ROOT / "data" / "transkrip"


# --------------------------------------------------------------------------
# Bantu
# --------------------------------------------------------------------------


def _utt_from_text(text: str, uid: str = "input-0001") -> Utterance:
    """Bangun ujaran dari teks polos dengan penanda waktu netral."""
    words = [(w, O) for w in text.split()]
    if not words:
        raise SystemExit("teks masukan kosong")
    return Utterance(
        uid=uid,
        tokens=neutral_timings(words),
        source="teks-polos",
        meta={"timing": "netral", "peringatan": "tanpa audio, fitur prosodi netral"},
    )


def _load_detector(args, lex: Lexicon):
    if args.detector == "naif":
        return NaiveDetector(lex)
    if args.detector == "aturan":
        return RuleBasedDetector(lex)
    path = Path(args.model)
    if not path.exists():
        raise SystemExit(
            f"model tidak ditemukan: {path}\n"
            "Latih dulu: python -m disfluency_id train"
        )
    return DisfluencyTagger.load(path, lex=lex)


def _render(utt: Utterance, cut: frozenset[str]) -> str:
    parts = []
    for tok in utt.tokens:
        lab = tok.pred if tok.pred is not None else tok.label
        parts.append(f"{tok.text}/{lab}" if lab != O else tok.text)
    lines = [" ".join(parts), ""]
    spans = list(iter_spans(utt, use_pred=True))
    if spans:
        lines.append("Span disfluensi terdeteksi:")
        for s in spans:
            mark = "potong" if s.label in cut else "simpan"
            lines.append(
                f"  [{s.label}] {s.start:7.3f}-{s.end:7.3f}s "
                f"({s.duration:.3f}s) {mark:>6}  \"{s.text}\""
            )
    else:
        lines.append("Tidak ada disfluensi terdeteksi.")
    lines += ["", f"Transkrip bersih: {utt.clean_text(cut=cut)}"]
    return "\n".join(lines)


def _cut_set(name: str) -> frozenset[str]:
    return AGGRESSIVE_CUT if name == "agresif" else CONSERVATIVE_CUT


# --------------------------------------------------------------------------
# Perintah
# --------------------------------------------------------------------------


def cmd_stats(args) -> int:
    utts = build_corpus(args.corpus, seed=args.timing_seed)
    print(json.dumps(describe(utts), ensure_ascii=False, indent=2))
    return 0


def cmd_build(args) -> int:
    utts = build_corpus(args.corpus, seed=args.timing_seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(out, utts)
    print(f"Ditulis {n} ujaran ke {out}")
    print(json.dumps(describe(utts), ensure_ascii=False, indent=2))
    return 0


def cmd_experiment(args) -> int:
    run_experiment(
        corpus_path=args.corpus,
        outdir=args.outdir,
        k=args.folds,
        epochs=args.epochs,
        seed=args.seed,
        timing_seed=args.timing_seed,
    )
    return 0


def cmd_train(args) -> int:
    lex = Lexicon.load()
    utts = build_corpus(args.corpus, seed=args.timing_seed)
    tagger = DisfluencyTagger(lex, FeatureConfig(True, True, True))
    tagger.fit(utts, epochs=args.epochs, seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tagger.save(out)

    preds = [tagger.predict(u) for u in utts]
    res = evaluate(utts, preds, lex, system="perceptron [penuh] - data latih")
    print(format_report(res))
    print(
        "\nCatatan: angka di atas dihitung pada data yang sama dengan data latih,\n"
        "sehingga terlalu optimistis. Pakai `experiment` untuk angka validasi silang."
    )
    print(f"\nModel disimpan ke {out}")
    return 0


def cmd_detect(args) -> int:
    lex = Lexicon.load()
    detector = _load_detector(args, lex)
    cut = _cut_set(args.cut)

    if args.text:
        utts = [_utt_from_text(args.text)]
        print("(masukan teks polos: fitur prosodi disetel netral)\n")
    elif args.jsonl:
        utts = read_jsonl(args.jsonl)
    else:
        raise SystemExit("berikan --text atau --jsonl")

    for utt in utts[: args.limit]:
        utt.assign(detector.predict(utt))
        print(f"--- {utt.uid} ---")
        print(_render(utt, cut))
        print()
    return 0


def cmd_explain(args) -> int:
    lex = Lexicon.load()
    if args.text:
        utts = [_utt_from_text(args.text)]
    elif args.jsonl:
        # Seluruh ujaran ditelusuri, bukan yang pertama saja. Versi sebelumnya
        # membaca `[0]` dan diam soal sisanya, sehingga korpus 33 ujaran
        # terbaca sebagai "cuma satu pasangan kata yang ditemukan".
        utts = read_jsonl(args.jsonl)
    else:
        raise SystemExit("berikan --text atau --jsonl")

    total = 0
    for utt in utts[: args.limit]:
        events = find_repeat_events(utt, lex)
        if not events:
            continue
        total += len(events)
        print(f"--- {utt.uid} ---")
        print(f"Ujaran: {utt.text}\n")
        for ev in events:
            verdict = {
                "reduplication": "REDUPLIKASI (pertahankan)",
                "disfluency": "DISFLUENSI (potong salinan pertama)",
                "ambiguous": "AMBIGU (default: pertahankan)",
            }[ev.kind]
            print(f"Token {ev.i}-{ev.j}  '{ev.base} {ev.base}'  jeda {ev.gap:.3f}s")
            print(f"  Putusan   : {verdict}")
            print(f"  Skor bukti: {ev.score:+.2f}  (keyakinan {ev.confidence:.2f})")
            for r in ev.reasons:
                print(f"    - {r}")
            print()

    if not total:
        print("Tidak ada pasangan kata identik berdampingan pada masukan ini.")
    else:
        print(f"{total} pasangan pada {len(utts[: args.limit])} ujaran yang diperiksa.")
    return 0


def cmd_media(args) -> int:
    from .media import (
        baca_provenans,
        ringkas_etika,
        total_durasi,
        unduh_semua,
    )

    daftar = baca_provenans(args.provenans)
    media_dir = Path(args.media_dir)

    if args.unduh:
        print(f"Mengunduh ke {media_dir}\n")
        unduh_semua(args.provenans, media_dir, hanya=args.hanya or None)
        print()

    print(f"{'berkas':<20} {'durasi':>9} {'porsi':>7}  {'lokal':>9}  penutur")
    print("-" * 78)
    for s in daftar:
        lokal = media_dir / s.berkas
        ada = f"{lokal.stat().st_size / 1e6:.1f} MB" if lokal.exists() else "-"
        dur = f"{s.durasi_detik:.2f}s" if s.durasi_detik is not None else "-"
        porsi = f"{s.porsi_kutipan:.1%}" if s.porsi_kutipan is not None else "-"
        print(f"{s.berkas:<20} {dur:>9} {porsi:>7}  {ada:>9}  {s.penutur[:28]}")

    total, kosong = total_durasi(daftar)
    ekor = f"  ({kosong} baris tanpa durasi)" if kosong else ""
    print("-" * 78)
    print(f"{len(daftar)} rekaman, {total:.2f} dtk = {total / 60:.2f} menit{ekor}")

    asli = sum(s.durasi_asli_detik or 0.0 for s in daftar)
    if asli:
        print(f"Karya asli {asli / 60:.1f} menit; terpakai {total / asli:.2%}.")

    peringatan = ringkas_etika(daftar)
    if peringatan:
        print(
            f"\nPERINGATAN: {len(peringatan)} rekaman belum lengkap provenansnya.\n"
            "Berkasnya boleh ditranskripsikan untuk uji coba, tetapi angka\n"
            "hasilnya belum boleh dikutip di dokumen penelitian."
        )
        print("\n".join(peringatan))
    return 0


def _ingest_semua(args) -> int:
    """Transkripsikan setiap rekaman yang terdaftar di provenans.

    Transkripsi adalah langkah termahal dalam rangkaian ini, dan sesi Colab
    bisa putus di tengah jalan. Berkas yang transkripnya sudah ada karena itu
    **tidak ditranskripsikan ulang**; ia cukup dibaca kembali menjadi korpus.
    Jalankan ulang perintah yang sama untuk melanjutkan dari tempat terputus.
    """
    from .ingest import MODEL_ID, from_whisper_json, ingest_media
    from .media import baca_provenans

    daftar = baca_provenans(args.provenans)
    media_dir = Path(args.media_dir)
    transkrip_dir = TRANSCRIPTS_DIR / "crisperwhisper"
    korpus_dir = Path(args.outdir)
    merge = not args.no_merge_hyphens

    # Nama boleh ditulis dengan atau tanpa akhiran. Nama yang tidak dikenal
    # dihentikan, bukan dilewati: salah ketik yang diabaikan diam-diam terbaca
    # sebagai "berkasnya sudah mutakhir".
    if getattr(args, "hanya", None):
        pilih = {Path(n).stem for n in args.hanya}
        dikenal = {Path(s.berkas).stem for s in daftar}
        asing = pilih - dikenal
        if asing:
            raise SystemExit(
                f"tidak ada baris provenans untuk: {', '.join(sorted(asing))}. "
                f"Tambahkan barisnya di {args.provenans} lebih dulu."
            )
        daftar = [s for s in daftar if Path(s.berkas).stem in pilih]

    print(f"{len(daftar)} rekaman terdaftar; model {MODEL_ID}\n")
    ringkas: list[tuple[str, str, int, int]] = []

    for s in daftar:
        media = media_dir / s.berkas
        stem = media.stem
        js = transkrip_dir / f"{stem}.json"
        out = korpus_dir / f"{stem}.jsonl"

        # Transkrip diperiksa LEBIH DULU: bila ia sudah ada, medianya tidak
        # diperlukan sama sekali. Menuntut media lebih dulu akan membuat
        # artefak termahal rangkaian ini tak terpakai hanya karena berkas
        # sumbernya belum diunduh ulang di sesi baru.
        if js.exists() and not args.ulangi:
            print(f"[{stem}] transkrip sudah ada, dipakai lagi: {js.name}")
            utts = from_whisper_json(
                js, max_gap=args.max_gap, source=s.berkas, merge_hyphens=merge
            )
            status = "transkrip lama"
        elif not media.exists():
            print(f"[{stem}] LEWAT - belum ada transkrip maupun berkasnya "
                  f"(python -m disfluency_id media --unduh)")
            ringkas.append((stem, "belum diunduh", 0, 0))
            continue
        else:
            # Jendela diambil per berkas dari provenans bila tercatat.
            # Nilai global 30 dipilih dengan mengadu 30/20/15 di SATU rekaman
            # dan ternyata tidak menyeberang: di dua berkas lain ia membuang
            # 35% dan 27% audio sebagai degenerasi, sementara jendela 20
            # membuang 5,6 dtk. Kaidah pemilihannya ditetapkan lebih dulu --
            # paling sedikit membuang audio, seri diputus jumlah token --
            # supaya ia bukan pemilihan setelah melihat hasil.
            jendela = s.jendela or args.jendela
            print(f"[{stem}] mentranskripsikan {s.durasi_detik or 0:.0f} dtk "
                  f"(jendela {jendela}) ...")
            utts = ingest_media(
                media,
                language=args.language,
                out_json=js,
                max_gap=args.max_gap,
                merge_hyphens=merge,
                jendela=jendela,
                maks_token=args.maks_token,
                koreksi_senyap=not args.tanpa_koreksi_senyap,
                progress=not args.no_progress,
            )
            status = "baru"

        out.parent.mkdir(parents=True, exist_ok=True)
        n = write_jsonl(out, utts)
        token = sum(len(u.tokens) for u in utts)
        print(f"[{stem}] {n} ujaran, {token} token -> {out}\n")
        ringkas.append((stem, status, n, token))

    print("-" * 62)
    print(f"{'berkas':<20} {'status':<16} {'ujaran':>7} {'token':>7}")
    for stem, status, n, token in ringkas:
        print(f"{stem:<20} {status:<16} {n:>7} {token:>7}")
    total = sum(t for _, _, _, t in ringkas)
    print("-" * 62)
    print(f"{total} token seluruhnya. Semua berlabel O; pelabelan tugas anotator.")
    print("\nPeriksa mutunya sebelum dipakai:")
    print(f"  python -m disfluency_id audit {korpus_dir.as_posix()}/*.jsonl")
    return 0


def cmd_ingest(args) -> int:
    from .ingest import (
        MODEL_ID,
        check_dependencies,
        from_whisper_json,
        ingest_media,
    )

    if args.check:
        status = check_dependencies()
        for name, ok in status.items():
            print(f"  {name:<16} {'tersedia' if ok else 'belum dipasang'}")
        return 0

    if args.semua:
        return _ingest_semua(args)

    merge = not args.no_merge_hyphens
    if args.whisper_json:
        utts = from_whisper_json(
            args.whisper_json, max_gap=args.max_gap, merge_hyphens=merge
        )
    elif args.audio:
        media = Path(args.audio)
        js = Path(args.transcript) if args.transcript else TRANSCRIPTS_DIR / f"{media.stem}.json"
        print(f"Mentranskripsikan {media.name} dengan {MODEL_ID}...")
        print("Sekali jalan saja; transkripnya disimpan supaya tidak perlu diulang.")
        print()
        utts = ingest_media(
            media,
            language=args.language,
            out_json=js,
            max_gap=args.max_gap,
            merge_hyphens=merge,
            jendela=args.jendela,
            maks_token=args.maks_token,
            koreksi_senyap=not args.tanpa_koreksi_senyap,
            progress=not args.no_progress,
        )
        print(f"Transkrip mentah  {js}")
    else:
        raise SystemExit("berikan --audio, --whisper-json, --semua, atau --check")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(out, utts)
    total = sum(len(u.tokens) for u in utts)
    print(f"Ditulis {n} ujaran ({total} token) ke {out}")
    print("Seluruh token berlabel O: pelabelan disfluensi adalah tugas anotator.")
    return 0


def cmd_audit(args) -> int:
    from .asr_audit import audit, audit_dict, format_audit

    paths = [Path(p) for p in args.jsonl]
    utts: list[Utterance] = []
    for p in paths:
        utts.extend(read_jsonl(p))
    if not utts:
        raise SystemExit("tidak ada ujaran pada berkas yang diberikan")

    # Dengan lebih dari satu berkas, ringkasan gabungan SAJA menyesatkan: satu
    # rekaman yang ASR-nya nyaris tidak mendengar apa pun tertimbun oleh yang
    # sehat, dan putusan "laju filler wajar" menjadi pernyataan tentang
    # kumpulan, bukan tentang bahan yang akan dianotasi satu per satu. Ini
    # bukan kemungkinan teoretis: pada korpus enam berkas proyek ini, putusan
    # gabungannya LULUS sementara tiga berkas tidak layak pakai.
    if len(paths) > 1:
        # Durasi rekaman diambil dari provenans. Ini bukan kelengkapan:
        # `audit` hanya mengetahui durasi BICARA, yakni jumlah rentang ujaran
        # yang berhasil ditranskripsikan. Membagi token dengan angka itu
        # mengukur laju bicara di dalam bagian yang terbaca, dan justru
        # menyembunyikan berkas yang sebagian besar audionya hilang -- sebab
        # sisa yang terbaca tetap rapat. Yang menyingkapnya adalah CAKUPAN:
        # berapa bagian rekaman yang punya transkrip sama sekali.
        durasi_media: dict[str, float] = {}
        try:
            from .media import baca_provenans

            for s in baca_provenans(args.provenans):
                if s.durasi_detik:
                    durasi_media[Path(s.berkas).stem] = s.durasi_detik
        except (FileNotFoundError, ValueError):
            pass

        print(f"Per berkas ({len(paths)})")
        print("=" * 78)
        print(f"{'berkas':<20} {'token':>6} {'kata/dtk':>9} {'cakupan':>8} "
              f"{'jeda nol':>9} {'filler':>7}  catatan")
        print("-" * 78)
        meragukan: list[str] = []
        for p in paths:
            satu = read_jsonl(p)
            if not satu:
                continue
            a = audit(satu, Lexicon.load())
            laju = a.fillers.tokens / a.duration if a.duration else 0.0
            media = durasi_media.get(p.stem)
            cakupan = a.duration / media if media else None

            catatan = []
            # Laju bicara Bahasa Indonesia lisan wajarnya 2-4 kata/detik.
            if laju < 1.5:
                catatan.append("LAJU BICARA MUSTAHIL RENDAH")
            # Cakupan rendah = sebagian besar rekaman tanpa transkrip sah,
            # entah karena degenerasi dibuang atau ASR tidak mendengar.
            if cakupan is not None and cakupan < 0.7:
                catatan.append(f"CAKUPAN {cakupan:.0%} -- {1 - cakupan:.0%} "
                               "rekaman tanpa transkrip")
            if a.fillers.suspected_deletion:
                catatan.append("filler diduga dihapus ASR")
            if catatan:
                meragukan.append(p.stem)

            teks_cakupan = f"{cakupan:7.0%}" if cakupan is not None else "      ?"
            print(f"{p.stem:<20} {a.fillers.tokens:6d} {laju:9.2f} "
                  f"{teks_cakupan} {a.gaps.zero_share:8.1%} {a.fillers.rate:6.1%}  "
                  f"{'; '.join(catatan)}")
        print("-" * 78)
        if not durasi_media:
            print("(cakupan tak terhitung: provenans tidak terbaca)")
        if meragukan:
            print(f"\n{len(meragukan)} dari {len(paths)} berkas MERAGUKAN: "
                  f"{', '.join(meragukan)}")
            print("Laporkan kelayakan per berkas, bukan rata-ratanya. Ringkasan")
            print("gabungan di bawah menimbun kerusakan ini.\n")
        else:
            print()

    label = paths[0].stem if len(paths) == 1 else f"{len(paths)} berkas"
    res = audit(utts, Lexicon.load())
    print(format_audit(res, title=f"Audit transkrip ASR - {label}"))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(audit_dict(res), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nRingkasan JSON  {out}")
    return 0


def cmd_edl(args) -> int:
    lex = Lexicon.load()
    detector = _load_detector(args, lex)
    utts = read_jsonl(args.jsonl) if args.jsonl else build_corpus(args.corpus)

    for utt in utts:
        utt.assign(detector.predict(utt))

    edl = build_edl(
        utts,
        EDLConfig(cut_labels=_cut_set(args.cut)),
        use_pred=True,
        source=args.source,
    )
    paths = write_outputs(edl, args.outdir, stem=args.stem)

    print(json.dumps(edl.stats(), ensure_ascii=False, indent=2))
    print()
    for kind, path in paths.items():
        print(f"  {kind:<10} {path}")
    return 0


def cmd_render(args) -> int:
    muatan = load_edl(args.edl)
    sumber = Path(args.media) if args.media else resolve_source(muatan, args.media_dir)

    if args.mode == "penuh":
        spans = keep_spans(muatan)
        if not args.hanya_ujaran:
            spans = extend_to_media(spans, probe_duration(sumber))
    else:
        spans = proof_spans(muatan, pad=args.pad, max_cuts=args.max_cuts)

    if not spans:
        print("tidak ada segmen untuk dirender; EDL ini tidak memuat potongan")
        return 1

    lebar = tinggi = None
    if args.scale:
        lebar, tinggi = (int(x) for x in args.scale.lower().split("x"))

    cfg = RenderConfig(
        width=lebar,
        height=tinggi,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        font=None if args.no_label else find_font(),
    )

    print(f"sumber   {sumber}")
    print(f"mode     {args.mode}  |  {len(spans)} segmen  |  {durasi(spans):.2f} dtk keluaran")
    if args.dry_run:
        print("\n-- dry run, ffmpeg tidak dijalankan --")
        print(" ".join(ffmpeg_argv(sumber, args.out, spans, cfg))[:600] + " ...")
        return 0

    hasil = render(sumber, args.out, spans, cfg)
    print(f"keluaran {hasil}  ({hasil.stat().st_size:,} byte)")
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="disfluency_id",
        description="Prototipe deteksi filler word & disfluensi Bahasa Indonesia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="perintah", required=True)

    def add_corpus(sp):
        sp.add_argument("--corpus", default=str(SEED_CORPUS), help="berkas anotasi inline")
        sp.add_argument("--timing-seed", type=int, default=20260812, dest="timing_seed")

    sp = sub.add_parser("stats", help="ringkasan korpus benih")
    add_corpus(sp)
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("build", help="bangun korpus JSONL bertanda waktu")
    add_corpus(sp)
    sp.add_argument("--out", default=str(DEFAULT_JSONL))
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("experiment", help="perbandingan sistem + ablasi fitur")
    add_corpus(sp)
    sp.add_argument("--outdir", default=str(REPORTS_DIR))
    sp.add_argument("--folds", type=int, default=5)
    sp.add_argument("--epochs", type=int, default=15)
    sp.add_argument("--seed", type=int, default=13)
    sp.set_defaults(func=cmd_experiment)

    sp = sub.add_parser("train", help="latih model penuh lalu simpan")
    add_corpus(sp)
    sp.add_argument("--out", default=str(DEFAULT_MODEL))
    sp.add_argument("--epochs", type=int, default=15)
    sp.add_argument("--seed", type=int, default=13)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("detect", help="jalankan detektor pada teks atau JSONL")
    sp.add_argument("--text")
    sp.add_argument("--jsonl")
    sp.add_argument("--detector", choices=["naif", "aturan", "model"], default="aturan")
    sp.add_argument("--model", default=str(DEFAULT_MODEL))
    sp.add_argument("--cut", choices=["konservatif", "agresif"], default="konservatif")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("explain", help="alasan putusan reduplikasi vs repetisi")
    sp.add_argument("--text")
    sp.add_argument("--jsonl")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_explain)

    sp = sub.add_parser("media", help="daftar / unduh rekaman dari provenans.csv")
    sp.add_argument("--unduh", action="store_true",
                    help="unduh berkas yang belum ada atau yang panjangnya tidak cocok")
    sp.add_argument("--hanya", nargs="+", metavar="BERKAS",
                    help="batasi ke nama berkas tertentu")
    sp.add_argument("--provenans", default=str(PROVENANS))
    sp.add_argument("--media-dir", default=str(MEDIA_DIR), dest="media_dir")
    sp.set_defaults(func=cmd_media)

    sp = sub.add_parser("ingest", help="impor transkrip ASR menjadi korpus JSONL")
    sp.add_argument("--audio", help=f"berkas video/audio lokal (taruh di {MEDIA_DIR})")
    sp.add_argument("--semua", action="store_true",
                    help="transkripsikan seluruh rekaman di provenans.csv; "
                         "yang transkripnya sudah ada dilewati")
    sp.add_argument("--ulangi", action="store_true",
                    help="dengan --semua: transkripsikan ulang walau transkripnya ada")
    sp.add_argument("--provenans", default=str(PROVENANS))
    sp.add_argument("--media-dir", default=str(MEDIA_DIR), dest="media_dir")
    sp.add_argument("--outdir", default=str(ROOT / "data" / "corpus" / "gpu"),
                    help="dengan --semua: folder korpus JSONL keluaran")
    sp.add_argument("--whisper-json", dest="whisper_json", help="keluaran Whisper word_timestamps")
    sp.add_argument("--hanya", nargs="+", metavar="BERKAS",
                    help="dengan --semua: batasi ke nama berkas tertentu "
                         "(boleh dengan atau tanpa akhiran)")
    sp.add_argument("--check", action="store_true", help="periksa pustaka opsional")
    sp.add_argument("--jendela", type=int, default=JENDELA_BAWAAN,
                    help=f"panjang jendela transkripsi, detik (bawaan {JENDELA_BAWAAN})")
    sp.add_argument("--maks-token", dest="maks_token", type=int,
                    default=MAKS_TOKEN_BAWAAN,
                    help="batas token per jendela; ini syarat muat di VRAM, bukan "
                         "penghematan waktu -- lihat disfluency_id/asr.py")
    sp.add_argument("--tanpa-koreksi-senyap", dest="tanpa_koreksi_senyap",
                    action="store_true",
                    help="jangan kikis senyap dari ujung kata lewat pengukuran "
                         "akustik (bawaan: dikikis)")
    sp.add_argument("--no-progress", dest="no_progress", action="store_true",
                    help="matikan baris kemajuan (baris itu ditulis ke stderr)")
    sp.add_argument("--language", default="id")
    sp.add_argument("--max-gap", dest="max_gap", type=float, default=0.8,
                    help="senyap (detik) yang memisahkan dua ujaran")
    sp.add_argument("--transcript", help="lokasi simpan JSON transkrip mentah")
    sp.add_argument("--no-merge-hyphens", dest="no_merge_hyphens", action="store_true",
                    help="biarkan 'kira' + '-kira' terpecah (untuk ablasi)")
    sp.add_argument("--out", default=str(DEFAULT_JSONL.with_name("asr.jsonl")))
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("audit", help="periksa mutu transkrip ASR sebelum dipakai")
    sp.add_argument("jsonl", nargs="+", help="satu atau lebih korpus JSONL hasil ingest")
    sp.add_argument("--json", help="tulis ringkasan JSON ke berkas ini")
    sp.add_argument("--provenans", default=str(PROVENANS),
                    help="sumber durasi rekaman untuk menghitung cakupan")
    sp.set_defaults(func=cmd_audit)

    sp = sub.add_parser("edl", help="susun Edit Decision List")
    add_corpus(sp)
    sp.add_argument("--jsonl")
    sp.add_argument("--detector", choices=["naif", "aturan", "model"], default="aturan")
    sp.add_argument("--model", default=str(DEFAULT_MODEL))
    sp.add_argument("--cut", choices=["konservatif", "agresif"], default="konservatif")
    sp.add_argument("--outdir", default=str(REPORTS_DIR / "edl"))
    sp.add_argument("--stem", default="edl")
    sp.add_argument("--source", default="audio.wav")
    sp.set_defaults(func=cmd_edl)

    sp = sub.add_parser("render", help="jalankan EDL menjadi berkas video")
    sp.add_argument("--edl", required=True, help="berkas edl.json hasil perintah edl")
    sp.add_argument("--media", help="berkas sumber; bawaannya diambil dari kunci `sumber` EDL")
    sp.add_argument("--media-dir", default="data/media", dest="media_dir")
    sp.add_argument("--out", required=True)
    # Bawaannya `penuh`: itu yang dipakai seluruh notebook, dan bawaan yang
    # tak pernah dipakai adalah ranjau bagi siapa pun yang menjalankan tanpa
    # membaca bantuannya.
    sp.add_argument("--mode", choices=["penuh", "bukti"], default="penuh")
    sp.add_argument("--pad", type=float, default=2.0, help="detik konteks di kiri-kanan tiap potongan (mode bukti)")
    sp.add_argument("--max-cuts", type=int, default=None, dest="max_cuts")
    sp.add_argument("--scale", help="mis. 640x360; bawaannya resolusi asli")
    sp.add_argument("--fps", type=int)
    sp.add_argument("--crf", type=int, default=23)
    sp.add_argument("--preset", default="veryfast")
    sp.add_argument("--hanya-ujaran", action="store_true", dest="hanya_ujaran",
                    help="mode penuh: berhenti pada kata pertama/terakhir, buang senyap ujung")
    sp.add_argument("--no-label", action="store_true", dest="no_label")
    sp.add_argument("--dry-run", action="store_true", dest="dry_run")
    sp.set_defaults(func=cmd_render)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
