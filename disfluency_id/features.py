"""Make features for each token.

Three groups, each can be turned on or off so we can test what helps:

    lexical       the word, its neighbours, word list, position
    prosody       gaps and how long the word takes
    reduplication repeat marks and the reduplication answer

Every feature is a plain string, so a linear model can use it as is.
"""

from __future__ import annotations

from dataclasses import dataclass

from .baseline import find_repair_windows
from .lexicon import CAT_DM, CAT_EDIT, CAT_FP, CAT_FP_AMB, Lexicon, canonical
from .reduplication import (
    AMBIGUOUS,
    DISFLUENCY,
    REDUPLICATION,
    find_repeat_events,
    is_hyphenated_reduplication,
    is_partial_word,
)
from .schema import Utterance

#: Batas kelompok jeda (detik), rapat di bawah 0,2 s karena di situ
#: reduplikasi dan repetisi berpisah.
GAP_EDGES = (0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50)

#: Batas kelompok durasi (detik).
DUR_EDGES = (0.10, 0.15, 0.20, 0.30, 0.40, 0.60)

#: Batas rasio durasi token terhadap rerata ujaran.
RATIO_EDGES = (0.5, 0.8, 1.2, 1.8)


def bucket(value: float, edges: tuple[float, ...]) -> str:
    """Turn a number into a group name."""
    for i, edge in enumerate(edges):
        if value < edge:
            return f"b{i}"
    return f"b{len(edges)}"


@dataclass(frozen=True)
class FeatureConfig:
    lexical: bool = True
    prosody: bool = True
    reduplication: bool = True

    @property
    def name(self) -> str:
        parts = [
            n
            for n, on in (
                ("lex", self.lexical),
                ("pros", self.prosody),
                ("redup", self.reduplication),
            )
            if on
        ]
        return "+".join(parts) if parts else "none"


def _shape(word: str) -> str:
    """Rough look of a word: capital, digit, hyphen."""
    out = []
    if word[:1].isupper():
        out.append("Cap")
    if any(ch.isdigit() for ch in word):
        out.append("Num")
    if word.endswith("-"):
        out.append("TrailHyphen")
    elif "-" in word:
        out.append("Hyphen")
    return "|".join(out) if out else "plain"


def featurize(
    utt: Utterance,
    lex: Lexicon,
    cfg: FeatureConfig | None = None,
) -> list[list[str]]:
    """Build the feature list for every token in one utterance."""
    cfg = cfg or FeatureConfig()
    toks = utt.tokens
    n = len(toks)
    words = [canonical(t.text) for t in toks]

    mean_dur = (sum(t.duration for t in toks) / n) if n else 0.0

    # Putusan reduplikasi dihitung sekali, lalu dipetakan ke indeks token.
    verdict_first: dict[int, str] = {}
    verdict_second: dict[int, str] = {}
    if cfg.reduplication:
        for ev in find_repeat_events(utt, lex):
            verdict_first[ev.i] = ev.kind
            verdict_second[ev.j] = ev.kind

    # Tanpa ini model harus menebak batas reparandum dari kata saja.
    reparandum_idx: set[int] = set()
    editterm_idx: set[int] = set()
    if cfg.lexical:
        for start, edit_i in find_repair_windows(utt, lex):
            editterm_idx.add(edit_i)
            reparandum_idx.update(range(start, edit_i))

    all_feats: list[list[str]] = []
    for i in range(n):
        f: list[str] = ["bias"]
        w = words[i]

        # ---------------- leksikal ----------------
        if cfg.lexical:
            f.append(f"w={w}")
            f.append(f"shape={_shape(toks[i].text)}")
            f.append(f"len={min(len(w), 12)}")
            f.append(f"suf3={w[-3:]}")
            f.append(f"pre3={w[:3]}")

            prev1 = words[i - 1] if i >= 1 else "<bos>"
            prev2 = words[i - 2] if i >= 2 else "<bos>"
            next1 = words[i + 1] if i < n - 1 else "<eos>"
            next2 = words[i + 2] if i < n - 2 else "<eos>"
            f.append(f"w-1={prev1}")
            f.append(f"w-2={prev2}")
            f.append(f"w+1={next1}")
            f.append(f"w+2={next2}")
            f.append(f"w-1|w={prev1}|{w}")
            f.append(f"w|w+1={w}|{next1}")

            cat = lex.category(w) or "none"
            f.append(f"cat={cat}")
            f.append(f"cat-1={lex.category(prev1) or 'none'}")
            f.append(f"cat+1={lex.category(next1) or 'none'}")
            f.append(f"isfp={cat in (CAT_FP, CAT_FP_AMB)}")
            f.append(f"isdm={cat == CAT_DM}")
            f.append(f"isedit={cat == CAT_EDIT}")
            f.append(f"isfunc={lex.is_function_word(w)}")

            if i == 0:
                f.append("pos=bos")
            elif i == n - 1:
                f.append("pos=eos")
            else:
                f.append("pos=mid")
            f.append(f"relpos={min(4, int(5 * i / max(1, n)))}")

            # Struktur ralat
            f.append(f"in_reparandum={i in reparandum_idx}")
            f.append(f"is_editterm={i in editterm_idx}")
            f.append(f"after_editterm={(i - 1) in editterm_idx}")
            ahead = next(
                (d for d in range(1, 5) if (i + d) in editterm_idx), 0
            )
            f.append(f"editterm_ahead={ahead}")

        # ---------------- prosodi ----------------
        if cfg.prosody:
            gb = utt.gap_before(i)
            ga = utt.gap_after(i)
            f.append(f"gapb={bucket(gb, GAP_EDGES)}")
            f.append(f"gapa={bucket(ga, GAP_EDGES)}")
            f.append(f"dur={bucket(toks[i].duration, DUR_EDGES)}")
            ratio = toks[i].duration / mean_dur if mean_dur > 0 else 1.0
            f.append(f"durratio={bucket(ratio, RATIO_EDGES)}")
            f.append(f"gapb_long={gb >= lex.gap_disfluency_min}")
            f.append(f"gapb_tight={gb <= lex.gap_redup_max}")
            if cfg.lexical:
                f.append(f"w|gapb={w}|{bucket(gb, GAP_EDGES)}")

        # ---------------- reduplikasi ----------------
        if cfg.reduplication:
            prev1 = words[i - 1] if i >= 1 else "<bos>"
            next1 = words[i + 1] if i < n - 1 else "<eos>"
            eq_prev = i >= 1 and w == prev1
            eq_next = i < n - 1 and w == next1
            f.append(f"eqprev={eq_prev}")
            f.append(f"eqnext={eq_next}")
            f.append(f"hyphredup={is_hyphenated_reduplication(toks[i].text, lex)}")
            f.append(f"reduplicable={lex.is_reduplicable(w)}")

            nxt_text = toks[i + 1].text if i < n - 1 else None
            f.append(
                f"partial={is_partial_word(toks[i].text, nxt_text, utt.gap_after(i), lex)}"
            )

            v1 = verdict_first.get(i)
            v2 = verdict_second.get(i)
            f.append(f"repeat_first={v1 or 'none'}")
            f.append(f"repeat_second={v2 or 'none'}")
            if v1 == DISFLUENCY:
                f.append("verdict=disfluency")
            elif v1 == REDUPLICATION:
                f.append("verdict=reduplication")
            elif v1 == AMBIGUOUS:
                f.append("verdict=ambiguous")

        all_feats.append(f)

    return all_feats
