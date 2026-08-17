"""Penanda sekuens: structured averaged perceptron.

Pilihan model sengaja jatuh pada perceptron terstruktur (Collins, 2002)
alih-alih pustaka pihak ketiga, dengan tiga alasan yang relevan bagi
penelitian ini:

1. Pemodelan sekuens. Disfluensi adalah fenomena berentang, bukan
   keputusan per kata yang saling bebas; skor transisi antarlabel
   menangkap kecenderungan seperti REP yang tidak pernah mengikuti REP
   pada rentetan panjang.
2. Ketertelusuran. Bobot bersifat linear terhadap fitur biner sehingga
   sumbangan tiap fitur dapat dibaca langsung -- penting saat pipeline
   harus dipertanggungjawabkan secara ilmiah.
3. Nol ketergantungan. Seluruhnya pustaka standar Python, sehingga hasil
   dapat direproduksi tanpa menyamakan versi pustaka numerik.

Model ini adalah garis dasar terpelajar. Pada penelitian penuh, ia
dibandingkan dengan penala IndoBERT dan CRF; antarmuka `fit`/`predict`
di sini sengaja dibuat sama agar model tersebut dapat dipasang tanpa
mengubah pipeline evaluasi.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .features import FeatureConfig, featurize
from .lexicon import Lexicon
from .schema import LABELS, O, Utterance

START = "<s>"


class StructuredPerceptron:
    """Perceptron terstruktur dengan dekode Viterbi dan bobot terata-rata."""

    def __init__(self, labels: Sequence[str] = LABELS) -> None:
        self.labels: list[str] = list(labels)
        self.weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.trans: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._w_totals: dict[tuple[str, str], float] = defaultdict(float)
        self._w_stamps: dict[tuple[str, str], int] = defaultdict(int)
        self._t_totals: dict[tuple[str, str], float] = defaultdict(float)
        self._t_stamps: dict[tuple[str, str], int] = defaultdict(int)
        self._steps = 0
        self.averaged = False

    # ----------------------------------------------------------- internal

    def _bump_weight(self, feat: str, label: str, delta: float) -> None:
        key = (feat, label)
        self._w_totals[key] += (self._steps - self._w_stamps[key]) * self.weights[feat][label]
        self._w_stamps[key] = self._steps
        self.weights[feat][label] += delta

    def _bump_trans(self, prev: str, label: str, delta: float) -> None:
        key = (prev, label)
        self._t_totals[key] += (self._steps - self._t_stamps[key]) * self.trans[prev][label]
        self._t_stamps[key] = self._steps
        self.trans[prev][label] += delta

    def _emissions(self, feats: Sequence[str]) -> dict[str, float]:
        scores = {lab: 0.0 for lab in self.labels}
        for f in feats:
            col = self.weights.get(f)
            if not col:
                continue
            for lab, val in col.items():
                if val:
                    scores[lab] += val
        return scores

    # ------------------------------------------------------------ decode

    def decode(self, feat_seq: Sequence[Sequence[str]]) -> list[str]:
        """Cari urutan label berskor tertinggi dengan Viterbi."""
        n = len(feat_seq)
        if n == 0:
            return []

        emit0 = self._emissions(feat_seq[0])
        best: dict[str, float] = {
            lab: emit0[lab] + self.trans[START][lab] for lab in self.labels
        }
        back: list[dict[str, str]] = []

        for i in range(1, n):
            emit = self._emissions(feat_seq[i])
            cur: dict[str, float] = {}
            ptr: dict[str, str] = {}
            for lab in self.labels:
                best_prev = None
                best_score = float("-inf")
                for prev in self.labels:
                    score = best[prev] + self.trans[prev][lab]
                    if score > best_score:
                        best_score = score
                        best_prev = prev
                cur[lab] = best_score + emit[lab]
                ptr[lab] = best_prev or O
            best = cur
            back.append(ptr)

        last = max(self.labels, key=lambda lab: best[lab])
        seq = [last]
        for ptr in reversed(back):
            last = ptr[last]
            seq.append(last)
        seq.reverse()
        return seq

    # --------------------------------------------------------------- fit

    def fit(
        self,
        data: Sequence[tuple[list[list[str]], list[str]]],
        epochs: int = 15,
        seed: int = 13,
    ) -> list[float]:
        """Latih model; kembalikan galat token per epoch untuk diagnosis."""
        rng = random.Random(seed)
        order = list(range(len(data)))
        history: list[float] = []

        for _ in range(epochs):
            rng.shuffle(order)
            wrong = total = 0
            for idx in order:
                feat_seq, gold = data[idx]
                if not gold:
                    continue
                self._steps += 1
                pred = self.decode(feat_seq)

                total += len(gold)
                wrong += sum(1 for a, b in zip(pred, gold) if a != b)
                if pred == gold:
                    continue

                for i, (g, p) in enumerate(zip(gold, pred)):
                    if g == p:
                        continue
                    for f in feat_seq[i]:
                        self._bump_weight(f, g, 1.0)
                        self._bump_weight(f, p, -1.0)

                gold_prev = START
                pred_prev = START
                for g, p in zip(gold, pred):
                    if (gold_prev, g) != (pred_prev, p):
                        self._bump_trans(gold_prev, g, 1.0)
                        self._bump_trans(pred_prev, p, -1.0)
                    gold_prev, pred_prev = g, p

            history.append(wrong / total if total else 0.0)

        self._average()
        return history

    def _average(self) -> None:
        """Ganti bobot akhir dengan rerata sepanjang pelatihan.

        Perataan meredam guncangan contoh-contoh terakhir dan terbukti
        memperbaiki generalisasi, terutama pada korpus kecil seperti ini.
        """
        for (feat, lab), total in self._w_totals.items():
            total += (self._steps - self._w_stamps[(feat, lab)]) * self.weights[feat][lab]
            avg = total / self._steps if self._steps else 0.0
            if avg:
                self.weights[feat][lab] = avg
            else:
                self.weights[feat].pop(lab, None)
        for (prev, lab), total in self._t_totals.items():
            total += (self._steps - self._t_stamps[(prev, lab)]) * self.trans[prev][lab]
            self.trans[prev][lab] = total / self._steps if self._steps else 0.0
        self.averaged = True

    # -------------------------------------------------------------- I/O

    def to_dict(self) -> dict:
        return {
            "labels": self.labels,
            "averaged": self.averaged,
            "weights": {f: dict(col) for f, col in self.weights.items() if col},
            "trans": {p: dict(col) for p, col in self.trans.items() if col},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructuredPerceptron":
        m = cls(d.get("labels", LABELS))
        for f, col in d.get("weights", {}).items():
            m.weights[f] = defaultdict(float, col)
        for p, col in d.get("trans", {}).items():
            m.trans[p] = defaultdict(float, col)
        m.averaged = d.get("averaged", True)
        return m


# --------------------------------------------------------------------------
# Pembungkus tingkat ujaran
# --------------------------------------------------------------------------


@dataclass
class DisfluencyTagger:
    """Rangkai leksikon, konfigurasi fitur, dan model menjadi satu detektor."""

    lex: Lexicon
    cfg: FeatureConfig = field(default_factory=FeatureConfig)
    model: StructuredPerceptron = field(default_factory=StructuredPerceptron)
    name: str = "perceptron"
    train_history: list[float] = field(default_factory=list)

    def fit(
        self,
        utterances: Sequence[Utterance],
        epochs: int = 15,
        seed: int = 13,
    ) -> "DisfluencyTagger":
        data = [
            (featurize(u, self.lex, self.cfg), u.labels)
            for u in utterances
        ]
        self.train_history = self.model.fit(data, epochs=epochs, seed=seed)
        return self

    def predict(self, utt: Utterance) -> list[str]:
        return self.model.decode(featurize(utt, self.lex, self.cfg))

    # -------------------------------------------------------------- I/O

    def save(self, path: Path | str) -> None:
        payload = {
            "name": self.name,
            "config": {
                "lexical": self.cfg.lexical,
                "prosody": self.cfg.prosody,
                "reduplication": self.cfg.reduplication,
            },
            "train_history": self.train_history,
            "model": self.model.to_dict(),
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path | str, lex: Lexicon | None = None) -> "DisfluencyTagger":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = FeatureConfig(**payload.get("config", {}))
        return cls(
            lex=lex or Lexicon.load(),
            cfg=cfg,
            model=StructuredPerceptron.from_dict(payload["model"]),
            name=payload.get("name", "perceptron"),
            train_history=payload.get("train_history", []),
        )
