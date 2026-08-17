"""Tell real reduplication from a repeated word.

    anak-anak      -> real word, means "kids"  (keep it)
    saya saya mau  -> the speaker stumbled     (cut the first one)

We add up clues from three places: the word list, the word type, and the
gap between the two words. Each answer keeps its clues so we can check it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .lexicon import Lexicon, canonical
from .schema import Utterance

REDUPLICATION = "reduplication"
DISFLUENCY = "disfluency"
AMBIGUOUS = "ambiguous"

#: Batas putusan pada skor bukti yang sudah dijumlahkan.
DECISION_MARGIN = 0.75

#: Every weight the classifier applies, in one place.
#:
#: `classify_adjacent_repeat` must read from here rather than spell a number
#: inline, and the spec table in the notebook is rendered by `weight_table`
#: rather than typed by hand. Both halves are enforced by
#: `tests/test_reduplication.py`: a hand-typed table drifts silently -- it
#: stays correct line by line while a newly added rule is simply missing.
WEIGHTS: dict[str, float] = {
    "known_reduplication": 3.0,
    "already_doubled": -3.0,
    "function_word": -3.0,
    "open_class": 0.5,
    "filler_repeated": -2.5,
    "gap_short": 1.5,
    "gap_long": -1.5,
    "length_clipped": -0.8,
    "length_equal": 0.4,
    "run_three_or_more": -2.0,
}

#: Ambang rasio panjang kedua salinan (dipakai kaidah `length_*`).
RATIO_CLIPPED_MAX = 0.6
RATIO_EQUAL_MIN = 0.8
RATIO_EQUAL_MAX = 1.25

#: Panjang rentetan yang sudah dianggap bukan reduplikasi.
RUN_DISFLUENT_MIN = 3


@dataclass(frozen=True)
class WeightRule:
    """One scoring rule, as the reader of the notebook sees it.

    `rule` is Indonesian on purpose: it is printed, and printed text is read
    by the examiner. Identifiers stay English.
    """

    key: str
    family: str
    rule: str
    #: Kaidah zona abu-abu tidak punya bobot sendiri; ia menginterpolasi
    #: lurus di antara dua bobot lain.
    interpolates: tuple[str, str] | None = None

    def weight_text(self) -> str:
        if self.interpolates is not None:
            lo, hi = (self._fmt(WEIGHTS[k]) for k in self.interpolates)
            return f"{lo} s/d {hi}"
        return self._fmt(WEIGHTS[self.key])

    @staticmethod
    def _fmt(value: float) -> str:
        # Koma desimal, dan hanya pada angkanya -- pemisah rentangnya jangan
        # ikut tergantikan.
        return f"{value:+.1f}".replace(".", ",")


#: Urutan baris tabel spesifikasi. Tiap kunci `WEIGHTS` wajib muncul di sini.
RULES: tuple[WeightRule, ...] = (
    WeightRule(
        "known_reduplication",
        "Leksikal",
        "`anak-anak` terdaftar reduplikasi gramatikal",
    ),
    WeightRule(
        "already_doubled",
        "Leksikal",
        "bentuk yang sudah bereduplikasi diulang lagi",
    ),
    WeightRule(
        "filler_repeated",
        "Leksikal",
        "filler diulang (`eee eee`) -- filler tidak punya reduplikasi",
    ),
    WeightRule(
        "function_word",
        "Morfosintaktis",
        "kata fungsi (`saya`, `yang`, `di`) tidak produktif direduplikasi",
    ),
    WeightRule("open_class", "Morfosintaktis", "kelas terbuka"),
    WeightRule(
        "gap_short",
        "Prosodi",
        "jeda <= {gap_redup_max} dtk (satu satuan ucap)",
    ),
    WeightRule("gap_long", "Prosodi", "jeda >= {gap_disfluency_min} dtk"),
    WeightRule(
        "gap_between",
        "Prosodi",
        "jeda di antara {gap_redup_max} dan {gap_disfluency_min} dtk, "
        "diinterpolasi lurus",
        interpolates=("gap_short", "gap_long"),
    ),
    WeightRule(
        "length_clipped",
        "Prosodi",
        "salinan pertama terpotong (rasio panjang < {ratio_clipped_max})",
    ),
    WeightRule(
        "length_equal",
        "Prosodi",
        "kedua salinan sama panjang (rasio {ratio_equal_min}-{ratio_equal_max})",
    ),
    WeightRule(
        "run_three_or_more",
        "Struktural",
        "muncul {run_disfluent_min}x berturut-turut atau lebih",
    ),
)


def _id_number(value: float) -> str:
    """Angka dengan koma desimal, buat teks yang dibaca orang Indonesia."""
    return f"{value:g}".replace(".", ",")


def weight_table(lex: Lexicon) -> str:
    """Render the weight table as Markdown, straight from the constants.

    The gap thresholds come from the lexicon file, so a change there shows
    up in the table too.
    """
    fields = {
        "gap_redup_max": _id_number(lex.gap_redup_max),
        "gap_disfluency_min": _id_number(lex.gap_disfluency_min),
        "ratio_clipped_max": _id_number(RATIO_CLIPPED_MAX),
        "ratio_equal_min": _id_number(RATIO_EQUAL_MIN),
        "ratio_equal_max": _id_number(RATIO_EQUAL_MAX),
        "run_disfluent_min": _id_number(RUN_DISFLUENT_MIN),
    }
    rows = [
        f"| {r.family} | {r.rule.format(**fields)} | {r.weight_text()} |"
        for r in RULES
    ]
    margin = _id_number(DECISION_MARGIN)
    return "\n".join(
        [
            "| Sumber | Kaidah | Bobot |",
            "|---|---|---:|",
            *rows,
            "",
            f"Zona `|skor| <= {margin}` dianggap ambigu dan putusan bawaan "
            "dipertahankan.",
        ]
    )


@dataclass
class RepeatEvent:
    """One occurrence of two identical adjacent tokens."""

    i: int  # indeks salinan pertama
    j: int  # indeks salinan kedua
    base: str
    gap: float
    kind: str
    score: float
    reasons: list[str] = field(default_factory=list)
    #: Panjang rentetan penuh, disimpan supaya putusan bisa dihitung ulang
    #: persis (mis. menguji apa jadinya tanpa bukti prosodi).
    run_length: int = 2

    @property
    def confidence(self) -> float:
        return min(1.0, abs(self.score) / 3.0)

    def to_dict(self) -> dict:
        return {
            "token_range": [self.i, self.j + 1],
            "base": self.base,
            "gap": round(self.gap, 3),
            "kind": self.kind,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "reasons": self.reasons,
        }


# --------------------------------------------------------------------------
# Bentuk bertanda hubung dalam satu token
# --------------------------------------------------------------------------


def split_reduplication(word: str) -> tuple[str, str] | None:
    """Split 'anak-anak' into ('anak', 'anak'), or None if it is not that."""
    w = canonical(word)
    if w.count("-") != 1:
        return None
    left, right = w.split("-")
    if not left or not right:
        return None
    return left, right


def is_hyphenated_reduplication(word: str, lex: Lexicon) -> bool:
    """One hyphenated word that is real reduplication, so we keep it."""
    parts = split_reduplication(word)
    if parts is None:
        return False
    if lex.is_known_reduplication(word):
        return True
    left, right = parts
    # Reduplikasi penuh produktif: kedua paruh identik dan bukan kata fungsi.
    if left == right:
        return lex.is_reduplicable(left)
    # Reduplikasi berafiks: berlari-lari, tarik-menarik, mobil-mobilan.
    if right.endswith(left) or left.endswith(right):
        return True
    if right.startswith(left) or left.startswith(right):
        return True
    return False


def is_partial_word(
    word: str,
    next_word: str | None,
    gap_after: float,
    lex: Lexicon,
) -> bool:
    """Find a cut-off word ('sa-' before 'sapi').

    We accept two clues: a hyphen at the end, or a start of the next word
    said right after it with almost no gap.
    """
    w = canonical(word)
    if not w:
        return False
    if word.rstrip().endswith("-") and len(w.rstrip("-")) >= 1:
        return True
    if next_word is None:
        return False
    nxt = canonical(next_word)
    stem = w.rstrip("-")
    if len(stem) < 2 or stem == nxt:
        return False
    if not nxt.startswith(stem):
        return False
    # Fragmen sejati harus lebih pendek dan bukan kata utuh yang sah.
    if lex.is_function_word(stem) or lex.is_filler(stem):
        return False
    return gap_after <= 0.35


# --------------------------------------------------------------------------
# Putusan untuk dua token identik berdampingan
# --------------------------------------------------------------------------


def classify_adjacent_repeat(
    base: str,
    gap: float,
    lex: Lexicon,
    dur_first: float | None = None,
    dur_second: float | None = None,
    run_length: int = 2,
) -> tuple[str, float, list[str]]:
    """Decide if two same words are reduplication or a stumble.

    Returns (kind, score, reasons). A plus score leans to reduplication,
    a minus score leans to a stumble.
    """
    b = canonical(base)
    score = 0.0
    reasons: list[str] = []

    # -- Petunjuk dari daftar kata -----------------------------------------
    joined = f"{b}-{b}"
    if lex.is_known_reduplication(joined):
        score += WEIGHTS["known_reduplication"]
        reasons.append(f"word list: '{joined}' is a real reduplicated word")

    # Bentuk yang sudah bereduplikasi tidak digandakan lagi.
    if "-" in b and is_hyphenated_reduplication(b, lex):
        score += WEIGHTS["already_doubled"]
        reasons.append(
            f"word shape: '{b}' is already doubled, and Indonesian does not "
            "double it twice"
        )

    # -- Petunjuk dari jenis kata ------------------------------------------
    if b in lex.function_words and joined not in lex.function_words_reduplicable:
        score += WEIGHTS["function_word"]
        reasons.append(f"word type: '{b}' is a function word, it is not doubled")
    elif lex.is_reduplicable(b):
        score += WEIGHTS["open_class"]
        reasons.append(f"word type: '{b}' is a normal word, it can be doubled")

    if lex.is_filler(b):
        # Filler tidak punya reduplikasi gramatikal, jadi pengulangannya
        # selalu hesitasi, serapat apa pun jedanya.
        score += WEIGHTS["filler_repeated"]
        reasons.append(f"word list: '{b}' is a filler, so saying it twice is hesitation")

    # -- Petunjuk dari jeda ------------------------------------------------
    if gap <= lex.gap_redup_max:
        score += WEIGHTS["gap_short"]
        reasons.append(f"gap: {gap:.3f}s <= {lex.gap_redup_max}s, said as one word")
    elif gap >= lex.gap_disfluency_min:
        score += WEIGHTS["gap_long"]
        reasons.append(f"gap: {gap:.3f}s >= {lex.gap_disfluency_min}s, speaker stopped")
    else:
        # Zona abu-abu: interpolasi linear supaya putusan tidak terjun bebas.
        near, far = WEIGHTS["gap_short"], WEIGHTS["gap_long"]
        span = lex.gap_disfluency_min - lex.gap_redup_max
        frac = (gap - lex.gap_redup_max) / span if span > 0 else 0.5
        contrib = near + (far - near) * frac
        score += contrib
        reasons.append(f"gap: {gap:.3f}s is in between, worth {contrib:+.2f}")

    # -- Petunjuk dari panjang kedua salinan -------------------------------
    if dur_first and dur_second and dur_first > 0 and dur_second > 0:
        ratio = dur_first / dur_second
        if ratio < RATIO_CLIPPED_MAX:
            score += WEIGHTS["length_clipped"]
            reasons.append(
                f"length: the first copy is much shorter (ratio {ratio:.2f}), "
                "which looks like a cut-off try"
            )
        elif RATIO_EQUAL_MIN <= ratio <= RATIO_EQUAL_MAX:
            score += WEIGHTS["length_equal"]
            reasons.append(f"length: both copies are as long (ratio {ratio:.2f})")

    # -- Petunjuk dari berapa kali kata itu muncul -------------------------
    if run_length >= RUN_DISFLUENT_MIN:
        score += WEIGHTS["run_three_or_more"]
        reasons.append(
            f"count: the word comes {run_length} times in a row, and real "
            "reduplication only doubles once"
        )

    if score > DECISION_MARGIN:
        kind = REDUPLICATION
    elif score < -DECISION_MARGIN:
        kind = DISFLUENCY
    else:
        kind = AMBIGUOUS
    return kind, score, reasons


def find_repeat_events(utt: Utterance, lex: Lexicon) -> list[RepeatEvent]:
    """Find every pair of same words next to each other, with its answer.

    For runs longer than two ('saya saya saya'), each pair is scored with
    the full run length as context.
    """
    events: list[RepeatEvent] = []
    toks = utt.tokens
    n = len(toks)
    i = 0
    while i < n - 1:
        cur = canonical(toks[i].text)
        if not cur or cur != canonical(toks[i + 1].text):
            i += 1
            continue

        # Ukur panjang rentetan penuh sebelum menilai tiap pasangan.
        run_end = i + 1
        while run_end + 1 < n and canonical(toks[run_end + 1].text) == cur:
            run_end += 1
        run_length = run_end - i + 1

        for k in range(i, run_end):
            kind, score, reasons = classify_adjacent_repeat(
                base=cur,
                gap=utt.gap_before(k + 1),
                lex=lex,
                dur_first=toks[k].duration,
                dur_second=toks[k + 1].duration,
                run_length=run_length,
            )
            events.append(
                RepeatEvent(
                    i=k,
                    j=k + 1,
                    base=cur,
                    gap=utt.gap_before(k + 1),
                    kind=kind,
                    score=score,
                    reasons=reasons,
                    run_length=run_length,
                )
            )
        i = run_end + 1
    return events
