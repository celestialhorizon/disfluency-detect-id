"""Scores for disfluency detection.

Besides the usual precision/recall/F1, we keep one score of our own:

    redup_false_cut_rate
        How many grammatical reduplication tokens were wrongly marked as
        disfluent. It is reported on its own because this mistake breaks
        the meaning ('anak-anak' becomes 'anak'), while the opposite
        mistake only leaves the audio a bit rough. One F1 number treats
        both the same, so it cannot show that difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .lexicon import Lexicon, canonical
from .reduplication import is_hyphenated_reduplication
from .schema import (
    DISFLUENT_LABELS,
    LABELS,
    O,
    REP,
    Utterance,
    iter_spans,
)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


@dataclass
class Score:
    label: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    @property
    def prf(self) -> tuple[float, float, float]:
        return _prf(self.tp, self.fp, self.fn)

    def to_dict(self) -> dict:
        p, r, f = self.prf
        return {
            "label": self.label,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": self.support,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass
class EvalResult:
    system: str
    n_utterances: int = 0
    n_tokens: int = 0
    per_label: dict[str, Score] = field(default_factory=dict)
    binary: Score = field(default_factory=lambda: Score("disfluent"))
    span: Score = field(default_factory=lambda: Score("span"))
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    # Angka khusus reduplikasi
    redup_tokens: int = 0
    redup_wrongly_cut: int = 0
    fluent_tokens: int = 0
    fluent_wrongly_cut: int = 0

    # ---------------------------------------------------------- aggregate

    @property
    def micro(self) -> tuple[float, float, float]:
        tp = sum(s.tp for lab, s in self.per_label.items() if lab != O)
        fp = sum(s.fp for lab, s in self.per_label.items() if lab != O)
        fn = sum(s.fn for lab, s in self.per_label.items() if lab != O)
        return _prf(tp, fp, fn)

    @property
    def macro(self) -> tuple[float, float, float]:
        rows = [s.prf for lab, s in self.per_label.items() if lab != O and s.support]
        if not rows:
            return 0.0, 0.0, 0.0
        n = len(rows)
        return (
            sum(r[0] for r in rows) / n,
            sum(r[1] for r in rows) / n,
            sum(r[2] for r in rows) / n,
        )

    @property
    def accuracy(self) -> float:
        correct = sum(self.confusion.get(l, {}).get(l, 0) for l in LABELS)
        return correct / self.n_tokens if self.n_tokens else 0.0

    @property
    def redup_false_cut_rate(self) -> float:
        return (
            self.redup_wrongly_cut / self.redup_tokens if self.redup_tokens else 0.0
        )

    @property
    def redup_preservation(self) -> float:
        return 1.0 - self.redup_false_cut_rate

    @property
    def fluent_false_cut_rate(self) -> float:
        return (
            self.fluent_wrongly_cut / self.fluent_tokens if self.fluent_tokens else 0.0
        )

    def to_dict(self) -> dict:
        mp, mr, mf = self.micro
        Mp, Mr, Mf = self.macro
        bp, br, bf = self.binary.prf
        sp, sr, sf = self.span.prf
        return {
            "system": self.system,
            "utterances": self.n_utterances,
            "tokens": self.n_tokens,
            "token_accuracy": round(self.accuracy, 4),
            "micro": {"precision": round(mp, 4), "recall": round(mr, 4), "f1": round(mf, 4)},
            "macro": {"precision": round(Mp, 4), "recall": round(Mr, 4), "f1": round(Mf, 4)},
            "binary_disfluent": {
                "precision": round(bp, 4),
                "recall": round(br, 4),
                "f1": round(bf, 4),
            },
            "span_exact": {
                "precision": round(sp, 4),
                "recall": round(sr, 4),
                "f1": round(sf, 4),
                "support": self.span.support,
            },
            "per_label": {lab: s.to_dict() for lab, s in self.per_label.items()},
            "reduplication": {
                "redup_tokens": self.redup_tokens,
                "wrongly_cut": self.redup_wrongly_cut,
                "false_cut_rate": round(self.redup_false_cut_rate, 4),
                "preservation_rate": round(self.redup_preservation, 4),
            },
            "fluent": {
                "fluent_tokens": self.fluent_tokens,
                "wrongly_cut": self.fluent_wrongly_cut,
                "false_cut_rate": round(self.fluent_false_cut_rate, 4),
            },
            "confusion_matrix": self.confusion,
        }


# --------------------------------------------------------------------------
# Menandai token reduplikasi
# --------------------------------------------------------------------------


def reduplication_token_indices(utt: Utterance, lex: Lexicon) -> set[int]:
    """Tokens the gold labels treat as grammatical reduplication.

    Two shapes count: one hyphenated token (anak-anak), and two identical
    tokens side by side that are both gold O (anak anak, as ASR often
    writes it without the hyphen).
    """
    idx: set[int] = set()
    toks = utt.tokens
    for i, tok in enumerate(toks):
        if tok.label == O and is_hyphenated_reduplication(tok.text, lex):
            idx.add(i)
    for i in range(len(toks) - 1):
        if (
            toks[i].label == O
            and toks[i + 1].label == O
            and canonical(toks[i].text) == canonical(toks[i + 1].text)
            and canonical(toks[i].text)
        ):
            idx.add(i)
            idx.add(i + 1)
    return idx


# --------------------------------------------------------------------------
# Penilaian
# --------------------------------------------------------------------------


def evaluate(
    utterances: list[Utterance],
    predictions: list[list[str]],
    lex: Lexicon,
    system: str = "system",
) -> EvalResult:
    """Compare predictions against the gold labels over all utterances."""
    if len(utterances) != len(predictions):
        raise ValueError("utterance count and prediction count differ")

    res = EvalResult(system=system, n_utterances=len(utterances))
    res.per_label = {lab: Score(lab) for lab in LABELS}
    res.confusion = {g: {p: 0 for p in LABELS} for g in LABELS}

    for utt, pred in zip(utterances, predictions):
        if len(pred) != len(utt):
            raise ValueError(f"prediction length does not match on {utt.uid}")
        utt.assign(pred)
        gold = utt.labels
        res.n_tokens += len(gold)

        redup_idx = reduplication_token_indices(utt, lex)

        for i, (g, p) in enumerate(zip(gold, pred)):
            res.confusion[g][p] += 1
            if g == p:
                res.per_label[g].tp += 1
            else:
                res.per_label[p].fp += 1
                res.per_label[g].fn += 1

            g_dis = g in DISFLUENT_LABELS
            p_dis = p in DISFLUENT_LABELS
            if g_dis and p_dis:
                res.binary.tp += 1
            elif p_dis and not g_dis:
                res.binary.fp += 1
            elif g_dis and not p_dis:
                res.binary.fn += 1

            if g == O:
                res.fluent_tokens += 1
                if p_dis:
                    res.fluent_wrongly_cut += 1
            if i in redup_idx:
                res.redup_tokens += 1
                if p_dis:
                    res.redup_wrongly_cut += 1

        gold_spans = {(s.label, s.i, s.j) for s in iter_spans(utt)}
        pred_spans = {(s.label, s.i, s.j) for s in iter_spans(utt, use_pred=True)}
        res.span.tp += len(gold_spans & pred_spans)
        res.span.fp += len(pred_spans - gold_spans)
        res.span.fn += len(gold_spans - pred_spans)

    return res


def merge(results: list[EvalResult], system: str) -> EvalResult:
    """Merge fold results into one by adding tp/fp/fn first.

    Adding counts first and computing precision after is fairer than
    averaging F1 over folds, which leans on the small folds.
    """
    out = EvalResult(system=system)
    out.per_label = {lab: Score(lab) for lab in LABELS}
    out.confusion = {g: {p: 0 for p in LABELS} for g in LABELS}

    for r in results:
        out.n_utterances += r.n_utterances
        out.n_tokens += r.n_tokens
        out.redup_tokens += r.redup_tokens
        out.redup_wrongly_cut += r.redup_wrongly_cut
        out.fluent_tokens += r.fluent_tokens
        out.fluent_wrongly_cut += r.fluent_wrongly_cut
        for lab, s in r.per_label.items():
            out.per_label[lab].tp += s.tp
            out.per_label[lab].fp += s.fp
            out.per_label[lab].fn += s.fn
        for key in ("binary", "span"):
            src = getattr(r, key)
            dst = getattr(out, key)
            dst.tp += src.tp
            dst.fp += src.fp
            dst.fn += src.fn
        for g, row in r.confusion.items():
            for p, c in row.items():
                out.confusion[g][p] += c
    return out


# --------------------------------------------------------------------------
# Tampilan
# --------------------------------------------------------------------------


def format_report(res: EvalResult) -> str:
    """One system as a Markdown table, ready to paste into the write-up."""
    mp, mr, mf = res.micro
    Mp, Mr, Mf = res.macro
    bp, br, bf = res.binary.prf
    sp, sr, sf = res.span.prf

    lines = [
        f"### System: {res.system}",
        "",
        f"- Utterances tested: {res.n_utterances}, tokens: {res.n_tokens}",
        f"- Token accuracy: {res.accuracy:.4f}",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for lab in LABELS:
        s = res.per_label.get(lab)
        if s is None or (s.support == 0 and s.fp == 0):
            continue
        p, r, f = s.prf
        lines.append(f"| {lab} | {p:.4f} | {r:.4f} | {f:.4f} | {s.support} |")

    lines += [
        f"| **micro (disfluent)** | {mp:.4f} | {mr:.4f} | {mf:.4f} | {sum(s.support for l, s in res.per_label.items() if l != O)} |",
        f"| **macro (disfluent)** | {Mp:.4f} | {Mr:.4f} | {Mf:.4f} | - |",
        f"| **binary disfluent/clean** | {bp:.4f} | {br:.4f} | {bf:.4f} | {res.binary.support} |",
        f"| **span exact match** | {sp:.4f} | {sr:.4f} | {sf:.4f} | {res.span.support} |",
        "",
        "**Reduplication check**",
        "",
        f"- Grammatical reduplication tokens in the test data: {res.redup_tokens}",
        f"- Wrongly marked disfluent: {res.redup_wrongly_cut}",
        f"- Reduplication false cut rate: {res.redup_false_cut_rate:.4f}",
        f"- Reduplication kept rate: {res.redup_preservation:.4f}",
        f"- False cut rate over all clean tokens: {res.fluent_false_cut_rate:.4f}",
        "",
    ]
    return "\n".join(lines)


def format_confusion(res: EvalResult) -> str:
    header = "| gold \\ predicted | " + " | ".join(LABELS) + " |"
    sep = "|---|" + "---:|" * len(LABELS)
    rows = [header, sep]
    for g in LABELS:
        cells = " | ".join(str(res.confusion[g][p]) for p in LABELS)
        rows.append(f"| **{g}** | {cells} |")
    return "\n".join(rows)


def format_comparison(results: list[EvalResult]) -> str:
    """Side-by-side table of every system."""
    lines = [
        "| System | Accuracy | micro F1 | macro F1 | binary F1 | span F1 | Reduplication kept |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        _, _, mf = r.micro
        _, _, Mf = r.macro
        _, _, bf = r.binary.prf
        _, _, sf = r.span.prf
        lines.append(
            f"| {r.system} | {r.accuracy:.4f} | {mf:.4f} | {Mf:.4f} | "
            f"{bf:.4f} | {sf:.4f} | {r.redup_preservation:.4f} |"
        )
    return "\n".join(lines)


def error_samples(
    utterances: list[Utterance],
    limit: int = 12,
    focus: str | None = None,
) -> list[dict]:
    """Collect wrong calls to read by hand; `focus` picks one gold label."""
    out: list[dict] = []
    for utt in utterances:
        for i, tok in enumerate(utt.tokens):
            pred = tok.pred if tok.pred is not None else O
            if pred == tok.label:
                continue
            if focus and tok.label != focus:
                continue
            lo, hi = max(0, i - 3), min(len(utt), i + 4)
            out.append(
                {
                    "uid": utt.uid,
                    "position": i,
                    "token": tok.text,
                    "gold": tok.label,
                    "predicted": pred,
                    "gap_before": round(utt.gap_before(i), 3),
                    "context": " ".join(
                        (f"[{t.text}]" if k == i else t.text)
                        for k, t in enumerate(utt.tokens[lo:hi], start=lo)
                    ),
                }
            )
            if len(out) >= limit:
                return out
    return out
