"""Loading and lookup for the Indonesian filler / reduplication lexicon."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .schema import normalize_word

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicon"

#: Kategori yang dikembalikan `Lexicon.category`.
CAT_FP = "filled_pause"
CAT_FP_AMB = "filled_pause_ambiguous"
CAT_DM = "discourse_marker"
CAT_EDIT = "editing_term"

#: Cadangan bila leksikon tidak memuat `editing_term_kuat`, supaya leksikon
#: lama tetap berperilaku seperti semula.
DEFAULT_EDITING_TERM_STRONG = frozenset(
    {"eh", "maaf", "sori", "sorry", "ralat", "maksudnya"}
)


def collapse_repeats(word: str, max_run: int = 2) -> str:
    """Damp letter runs: 'eeee' -> 'ee', 'hmmmm' -> 'hmm'.

    ASR spells filled pauses at inconsistent lengths; this folds every
    variant onto one canonical form.
    """
    if not word:
        return word
    out: list[str] = []
    run = 0
    prev = ""
    for ch in word:
        if ch == prev:
            run += 1
        else:
            run = 1
            prev = ch
        if run <= max_run:
            out.append(ch)
    return "".join(out)


def canonical(word: str) -> str:
    """Lexicon lookup key."""
    return collapse_repeats(normalize_word(word))


@dataclass
class Lexicon:
    """Filler lexicon plus reduplication resources, stored canonicalised."""

    filled_pause: set[str] = field(default_factory=set)
    filled_pause_ambiguous: set[str] = field(default_factory=set)
    discourse_marker: set[str] = field(default_factory=set)
    editing_term: set[str] = field(default_factory=set)

    #: Penanda ralat yang keanggotaannya sudah berarti fungsinya ('eh',
    #: 'ralat'). Sisanya ambigu: 'bukan' umumnya pengingkar biasa. Disimpan
    #: di leksikon supaya detektor dan audit tidak berbeda pendapat.
    editing_term_strong: set[str] = field(default_factory=set)

    multiword: dict[tuple[str, ...], str] = field(default_factory=dict)

    inherent_redup: set[str] = field(default_factory=set)
    lexicalized_redup: set[str] = field(default_factory=set)
    function_words: set[str] = field(default_factory=set)
    function_words_reduplicable: set[str] = field(default_factory=set)
    affix_patterns: list[re.Pattern[str]] = field(default_factory=list)

    gap_redup_max: float = 0.08
    gap_disfluency_min: float = 0.18

    #: Panjang frasa multiword terpanjang, dipakai saat pemindaian.
    max_multiword: int = 1

    # ---------------------------------------------------------------- load

    @classmethod
    def load(cls, data_dir: Path | str | None = None) -> "Lexicon":
        d = Path(data_dir) if data_dir is not None else DATA_DIR
        fillers = json.loads((d / "fillers_id.json").read_text(encoding="utf-8"))
        redup = json.loads((d / "reduplication_id.json").read_text(encoding="utf-8"))

        def canon_set(items: Sequence[str]) -> set[str]:
            return {canonical(w) for w in items if canonical(w)}

        multiword: dict[tuple[str, ...], str] = {}
        for phrase in fillers.get("discourse_marker_multiword", []):
            key = tuple(canonical(w) for w in phrase.split())
            if len(key) > 1:
                multiword[key] = CAT_DM
        for phrase in fillers.get("editing_term", []):
            parts = phrase.split()
            if len(parts) > 1:
                multiword[tuple(canonical(w) for w in parts)] = CAT_EDIT
        # Entri satu kata pada daftar discourse_marker boleh berupa frasa juga.
        for phrase in fillers.get("discourse_marker", []):
            parts = phrase.split()
            if len(parts) > 1:
                multiword[tuple(canonical(w) for w in parts)] = CAT_DM

        thresholds = redup.get("ambang_prosodi", {})
        patterns = [
            re.compile(p)
            for p in redup.get("pola_afiks_reduplikasi", {}).get("pola", [])
        ]

        return cls(
            filled_pause=canon_set(fillers.get("filled_pause", [])),
            filled_pause_ambiguous=canon_set(fillers.get("filled_pause_ambiguous", [])),
            discourse_marker=canon_set(
                [w for w in fillers.get("discourse_marker", []) if " " not in w]
            ),
            editing_term=canon_set(
                [w for w in fillers.get("editing_term", []) if " " not in w]
            ),
            editing_term_strong=(
                canon_set(
                    [w for w in fillers["editing_term_kuat"] if " " not in w]
                )
                if "editing_term_kuat" in fillers
                else set(DEFAULT_EDITING_TERM_STRONG)
            ),
            multiword=multiword,
            inherent_redup=canon_set(redup.get("inheren", [])),
            lexicalized_redup=canon_set(
                redup.get("lexicalized", []) + redup.get("berima", [])
            ),
            function_words=canon_set(redup.get("kata_fungsi", [])),
            function_words_reduplicable=canon_set(
                redup.get("kata_fungsi_boleh_reduplikasi", [])
            ),
            affix_patterns=patterns,
            gap_redup_max=float(thresholds.get("gap_reduplikasi_maks_detik", 0.08)),
            gap_disfluency_min=float(thresholds.get("gap_disfluensi_min_detik", 0.18)),
            max_multiword=max([1] + [len(k) for k in multiword]),
        )

    # -------------------------------------------------------------- lookup

    def category(self, word: str) -> str | None:
        """Filler category for one word, or None."""
        w = canonical(word)
        if w in self.filled_pause:
            return CAT_FP
        if w in self.filled_pause_ambiguous:
            return CAT_FP_AMB
        if w in self.editing_term:
            return CAT_EDIT
        if w in self.discourse_marker:
            return CAT_DM
        return None

    def is_filler(self, word: str) -> bool:
        return self.category(word) is not None

    def is_strong_editing_term(self, word: str) -> bool:
        """Editing term whose membership alone settles the decision.

        For weak ones, membership only means the word *can* mark a repair.
        """
        return canonical(word) in self.editing_term_strong

    def is_function_word(self, word: str) -> bool:
        return canonical(word) in self.function_words

    def match_multiword(self, words: Sequence[str], i: int) -> tuple[int, str] | None:
        """Longest filler phrase starting at `i`, as (length, category)."""
        seq = [canonical(w) for w in words]
        for n in range(min(self.max_multiword, len(seq) - i), 1, -1):
            key = tuple(seq[i : i + n])
            cat = self.multiword.get(key)
            if cat is not None:
                return n, cat
        return None

    # -------------------------------------------------------- reduplikasi

    def is_known_reduplication(self, word: str) -> bool:
        """Hyphenated form known to be grammatical reduplication."""
        w = canonical(word)
        if w in self.inherent_redup or w in self.lexicalized_redup:
            return True
        if w in self.function_words_reduplicable:
            return True
        return any(p.match(w) for p in self.affix_patterns)

    def is_reduplicable(self, base: str) -> bool:
        """Whether `base` can be productively reduplicated.

        Function words (saya, yang, di) practically never are, so a doubled
        one is almost certainly a disfluent repetition.
        """
        b = canonical(base)
        if b in self.function_words_reduplicable:
            return True
        if b in self.function_words:
            return False
        if self.is_filler(b):
            return False
        return len(b) >= 3
