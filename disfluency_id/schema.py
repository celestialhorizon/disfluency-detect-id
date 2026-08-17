"""Basic data types: Token, Utterance, Span, and JSONL read/write.

Each token gets one label. A span is a run of tokens with the same label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

# --------------------------------------------------------------------------
# Tagset
# --------------------------------------------------------------------------

O = "O"
FP = "FP"
DM = "DM"
REP = "REP"
RPR = "RPR"
PW = "PW"

LABELS: tuple[str, ...] = (O, FP, DM, REP, RPR, PW)

LABEL_DESC: dict[str, str] = {
    O: "clean word - keep it",
    FP: "filled pause - hesitation sound (eee, emm, hmm)",
    DM: "discourse marker used as filler (kayak, gitu, apa ya)",
    REP: "repetition - same word said twice by mistake",
    RPR: "reparandum - words the speaker dropped, then fixed",
    PW: "partial word - half a word (sa- sapi)",
}

#: Label yang layak dipotong dari audio final.
DISFLUENT_LABELS: frozenset[str] = frozenset({FP, DM, REP, RPR, PW})

#: Konservatif mempertahankan DM: membuang penanda wacana mengubah gaya
#: tutur, dan itu keputusan editorial, bukan pembersihan.
AGGRESSIVE_CUT: frozenset[str] = DISFLUENT_LABELS
CONSERVATIVE_CUT: frozenset[str] = DISFLUENT_LABELS - {DM}


# --------------------------------------------------------------------------
# Token & Utterance
# --------------------------------------------------------------------------


@dataclass
class Token:
    """One ASR word with its time boundaries in seconds."""

    text: str
    start: float
    end: float
    label: str = O
    #: Diisi detektor; label acuan tetap di `label`.
    pred: str | None = None

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"end < start on token {self.text!r}")
        if self.label not in LABELS:
            raise ValueError(f"unknown label: {self.label!r}")

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def norm(self) -> str:
        """Normalised form used for lexicon lookup."""
        return normalize_word(self.text)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "label": self.label,
        }
        if self.pred is not None:
            d["pred"] = self.pred
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Token":
        return cls(
            text=d["text"],
            start=float(d["start"]),
            end=float(d["end"]),
            label=d.get("label", O),
            pred=d.get("pred"),
        )


@dataclass
class Utterance:
    """A run of tokens from one audio source."""

    uid: str
    tokens: list[Token]
    source: str = "seed"
    speaker: str = "S1"
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def start(self) -> float:
        return self.tokens[0].start if self.tokens else 0.0

    @property
    def end(self) -> float:
        return self.tokens[-1].end if self.tokens else 0.0

    @property
    def labels(self) -> list[str]:
        return [t.label for t in self.tokens]

    @property
    def preds(self) -> list[str]:
        return [t.pred if t.pred is not None else O for t in self.tokens]

    def gap_before(self, i: int) -> float:
        """Silence before token i; the first token counts as 0."""
        if i <= 0:
            return 0.0
        return max(0.0, self.tokens[i].start - self.tokens[i - 1].end)

    def gap_after(self, i: int) -> float:
        if i >= len(self.tokens) - 1:
            return 0.0
        return max(0.0, self.tokens[i + 1].start - self.tokens[i].end)

    def assign(self, labels: Iterable[str]) -> None:
        """Write predictions into each token's `pred` field."""
        labels = list(labels)
        if len(labels) != len(self.tokens):
            raise ValueError(
                f"label count ({len(labels)}) != token count ({len(self.tokens)})"
            )
        for tok, lab in zip(self.tokens, labels):
            tok.pred = lab

    def clean_text(self, cut: frozenset[str] = CONSERVATIVE_CUT, use_pred: bool = True) -> str:
        """Transcript with disfluent material removed."""
        keep = []
        for tok in self.tokens:
            lab = (tok.pred if use_pred and tok.pred is not None else tok.label)
            if lab not in cut:
                keep.append(tok.text)
        return " ".join(keep)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source": self.source,
            "speaker": self.speaker,
            "text": self.text,
            "tokens": [t.to_dict() for t in self.tokens],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Utterance":
        return cls(
            uid=d["uid"],
            tokens=[Token.from_dict(t) for t in d["tokens"]],
            source=d.get("source", "seed"),
            speaker=d.get("speaker", "S1"),
            meta=d.get("meta", {}),
        )


# --------------------------------------------------------------------------
# Span
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    """A run of consecutive tokens sharing one label."""

    label: str
    i: int  # indeks token awal, inklusif
    j: int  # indeks token akhir, eksklusif
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "token_range": [self.i, self.j],
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "text": self.text,
        }


def iter_spans(
    utt: Utterance,
    labels: Iterable[str] | None = None,
    use_pred: bool = False,
) -> Iterator[Span]:
    """Merge consecutive same-label tokens into spans; O is always skipped."""
    wanted = set(labels) if labels is not None else set(DISFLUENT_LABELS)
    seq = utt.preds if use_pred else utt.labels

    i = 0
    n = len(seq)
    while i < n:
        lab = seq[i]
        if lab == O or lab not in wanted:
            i += 1
            continue
        j = i + 1
        while j < n and seq[j] == lab:
            j += 1
        yield Span(
            label=lab,
            i=i,
            j=j,
            start=utt.tokens[i].start,
            end=utt.tokens[j - 1].end,
            text=" ".join(t.text for t in utt.tokens[i:j]),
        )
        i = j


# --------------------------------------------------------------------------
# Normalisasi
# --------------------------------------------------------------------------

_PUNCT = ".,!?;:\"'()[]{}"


def normalize_word(word: str) -> str:
    """Lowercase and strip edge punctuation.

    Hyphens inside a word (anak-anak) and trailing hyphens (sa-) are kept:
    both carry reduplication or partial-word information.
    """
    w = word.strip().lower()
    w = w.lstrip(_PUNCT + "-")
    w = w.rstrip(_PUNCT)
    return w


def syllable_count(word: str) -> int:
    """Approximate Indonesian syllable count from vowel groups.

    Indonesian is almost entirely CV/CVC, so vowel groups approximate it well.
    """
    w = normalize_word(word).replace("-", "")
    vowels = set("aiueo")
    count = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(1, count)


# --------------------------------------------------------------------------
# I/O JSONL
# --------------------------------------------------------------------------


def write_jsonl(path, utterances: Iterable[Utterance]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for utt in utterances:
            fh.write(json.dumps(utt.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path) -> list[Utterance]:
    out: list[Utterance] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Utterance.from_dict(json.loads(line)))
    return out
