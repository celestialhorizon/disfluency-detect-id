"""Prototipe deteksi filler word & disfluensi ucapan Bahasa Indonesia.

Paket ini adalah bukti konsep yang mendampingi proposal penelitian
"Deteksi Filler Words & Disfluency pada Ucapan Bahasa Indonesia untuk
Automasi Penyuntingan Video/Podcast".

Alur lengkapnya: korpus beranotasi -> ekstraksi fitur (leksikal, prosodi,
reduplikasi) -> penandaan sekuens -> evaluasi -> Edit Decision List.

Titik masuk yang lazim dipakai:

    from disfluency_id import Lexicon, build_corpus, RuleBasedDetector
    from disfluency_id.experiment import run_experiment

Atau lewat baris perintah:

    python -m disfluency_id experiment
"""

from .corpus import build_corpus, describe, kfold, split
from .lexicon import Lexicon
from .schema import (
    DISFLUENT_LABELS,
    LABELS,
    Token,
    Utterance,
    iter_spans,
    read_jsonl,
    write_jsonl,
)

__version__ = "0.1.0"

__all__ = [
    "LABELS",
    "DISFLUENT_LABELS",
    "Token",
    "Utterance",
    "Lexicon",
    "iter_spans",
    "read_jsonl",
    "write_jsonl",
    "build_corpus",
    "describe",
    "split",
    "kfold",
    "__version__",
]


def __getattr__(name: str):
    """Impor malas untuk komponen berat agar `import disfluency_id` tetap ringan."""
    if name in {"RuleBasedDetector", "NaiveDetector"}:
        from . import baseline

        return getattr(baseline, name)
    if name in {"DisfluencyTagger", "StructuredPerceptron"}:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} tidak memiliki atribut {name!r}")
