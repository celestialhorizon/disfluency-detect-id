"""Detektor berbasis aturan: garis dasar sekaligus alat diagnosis.

Dua sistem disediakan dan keduanya dipakai dalam eksperimen:

`NaiveDetector`
    Meniru pendekatan yang lazim pada Bahasa Inggris: setiap dua kata
    identik yang berdampingan dianggap repetisi disfluen. Sistem ini
    ada bukan sebagai usulan, melainkan sebagai bukti terukur bahwa
    memindahkan asumsi Bahasa Inggris ke Bahasa Indonesia merusak
    reduplikasi gramatikal.

`RuleBasedDetector`
    Menambahkan penjagaan konteks leksikal, pemisahan reduplikasi
    berbasis bukti, deteksi fragmen kata, dan penelusuran ralat lewat
    pencarian rough copy.

Keduanya tidak memerlukan data latih sehingga dapat dijalankan pada
korpus mana pun sejak hari pertama.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lexicon import CAT_DM, CAT_EDIT, CAT_FP, CAT_FP_AMB, Lexicon, canonical
from .reduplication import (
    DISFLUENCY,
    find_repeat_events,
    is_hyphenated_reduplication,
    is_partial_word,
)
from .schema import DM, FP, O, PW, REP, RPR, Utterance

#: Penanda ralat yang cukup kuat untuk memicu penelusuran reparandum walau
#: tidak ditemukan rough copy kini berasal dari leksikon
#: (`Lexicon.editing_term_strong`), bukan dari konstanta modul ini. Selama
#: pengetahuan itu tertanam di sini, modul lain -- audit mutu transkrip,
#: misalnya -- tidak bisa melihatnya dan akhirnya menghitung setiap 'bukan'
#: sebagai penanda ralat.

#: Kata yang membuat 'gitu'/'gini' di sebelah kirinya bermakna penuh.
DEMONSTRATIVE_TAIL = frozenset({"saja", "aja", "juga", "doang"})

#: Panjang jendela maksimum saat mencari reparandum di kiri penanda ralat.
MAX_REPARANDUM = 4


# --------------------------------------------------------------------------
# Utilitas bersama
# --------------------------------------------------------------------------


def _rough_copy_match(left: list[str], right: list[str]) -> bool:
    """Apakah dua jendela merupakan salinan kasar satu sama lain.

    Ralat lisan hampir selalu mengulang kerangka frasa yang dibatalkan
    ('ke pasar' -> 'ke toko'). Kecocokan pada posisi yang sama, baik
    persis maupun lewat awalan bersama, sudah menjadi penanda kuat.
    """
    for a, b in zip(left, right):
        if a == b:
            return True
        if len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4]:
            return True
    return False


def _tag_multiword_fillers(utt: Utterance, lex: Lexicon, labels: list[str]) -> None:
    """Tandai frasa penanda wacana ('apa ya', 'gitu loh') secara rakus dari kiri.

    Frasa penanda ralat ('maksud saya') sengaja dilewati di sini. Menandainya
    langsung sebagai RPR akan melewati verifikasi struktur ralat, padahal
    'maksud saya' sama seringnya muncul sebagai ujaran bermakna. Frasa
    tersebut ditangani `find_repair_windows` yang menuntut adanya reparandum.
    """
    words = [t.text for t in utt.tokens]
    i = 0
    n = len(words)
    while i < n:
        hit = lex.match_multiword(words, i)
        if hit is None:
            i += 1
            continue
        length, cat = hit
        if cat != CAT_DM or any(labels[k] != O for k in range(i, i + length)):
            i += 1
            continue
        for k in range(i, i + length):
            labels[k] = DM
        i += length


def filler_positions(utt: Utterance, lex: Lexicon) -> set[int]:
    """Indeks token yang tertangkap leksikon filler, tanpa penjagaan konteks.

    Dipakai sebagai pemblokir ringan oleh `find_repair_windows`: penanda
    ralat yang didahului filler bukanlah ralat, melainkan bagian dari
    rangkaian hesitasi.
    """
    labels = [O] * len(utt)
    _tag_multiword_fillers(utt, lex, labels)
    for i, tok in enumerate(utt.tokens):
        if labels[i] == O and lex.category(tok.text) in (CAT_FP, CAT_FP_AMB, CAT_DM):
            labels[i] = FP
    return {i for i, lab in enumerate(labels) if lab != O}


def find_repair_windows(
    utt: Utterance,
    lex: Lexicon,
    blocked: set[int] | None = None,
) -> list[tuple[int, int]]:
    """Temukan rentang ralat sebagai pasangan (awal_reparandum, indeks_penanda).

    Reparandum dicari dengan membandingkan jendela di kiri penanda ralat
    terhadap jendela di kanannya: ralat lisan hampir selalu mengulang
    kerangka frasa yang dibatalkan. Jendela terpanjang yang menghasilkan
    kecocokan dipilih. Bila tidak ada yang cocok, hanya penanda kuat yang
    memicu asumsi reparandum sepanjang satu kata.

    Fungsi ini dipakai bersama oleh detektor aturan dan ekstraktor fitur
    agar keduanya melihat struktur ralat yang persis sama.
    """
    blocked = blocked if blocked is not None else filler_positions(utt, lex)
    words = [canonical(t.text) for t in utt.tokens]
    n = len(words)
    out: list[tuple[int, int]] = []

    i = 0
    while i < n:
        span, strong = _editing_term_at(words, i, lex)
        if span == 0 or i == 0 or i + span >= n:
            i += 1
            continue

        last = i + span - 1  # indeks token terakhir penanda ralat
        best: int | None = None
        limit = min(MAX_REPARANDUM, i, n - (i + span))
        for m in range(limit, 0, -1):
            left = words[i - m : i]
            right = words[i + span : i + span + m]
            if _rough_copy_match(left, right):
                best = m
                break

        if best is None:
            # Tanpa salinan kasar, hanya penanda kuat yang dianggap ralat,
            # dan hanya bila kata sebelumnya benar-benar materi ujaran.
            if not strong or (i - 1) in blocked:
                i += 1
                continue
            best = 1

        out.append((i - best, last))
        i = last + 1
    return out


def _editing_term_at(
    words: list[str], i: int, lex: Lexicon
) -> tuple[int, bool]:
    """Panjang penanda ralat yang mulai di `i`, dan apakah ia penanda kuat.

    Mengembalikan (0, False) bila tidak ada penanda ralat di posisi itu.
    Frasa diperiksa lebih dulu agar 'maksud saya' tidak terpecah menjadi
    dua token yang masing-masing tidak berarti apa-apa.
    """
    hit = lex.match_multiword(words, i)
    if hit is not None and hit[1] == CAT_EDIT:
        return hit[0], True

    w = words[i]
    if lex.category(w) not in (CAT_EDIT, CAT_FP_AMB):
        return 0, False
    if w not in lex.editing_term and not lex.is_strong_editing_term(w):
        return 0, False
    return 1, lex.is_strong_editing_term(w)


def _detect_repairs(utt: Utterance, lex: Lexicon, labels: list[str]) -> None:
    """Tandai reparandum dan penanda ralat sebagai RPR."""
    blocked = {i for i, lab in enumerate(labels) if lab in (FP, DM)}
    for start, edit_i in find_repair_windows(utt, lex, blocked):
        for k in range(start, edit_i + 1):
            labels[k] = RPR


# --------------------------------------------------------------------------
# Garis dasar naif
# --------------------------------------------------------------------------


@dataclass
class NaiveDetector:
    """Filler leksikal + repetisi tanpa kesadaran reduplikasi."""

    lex: Lexicon
    name: str = "naif"

    def predict(self, utt: Utterance) -> list[str]:
        labels = [O] * len(utt)
        _tag_multiword_fillers(utt, self.lex, labels)

        for i, tok in enumerate(utt.tokens):
            if labels[i] != O:
                continue
            cat = self.lex.category(tok.text)
            if cat in (CAT_FP, CAT_FP_AMB):
                labels[i] = FP
            elif cat in (CAT_DM, CAT_EDIT):
                labels[i] = DM

        # Setiap pasangan identik dianggap repetisi -- asumsi Bahasa Inggris.
        for i in range(len(utt) - 1):
            if canonical(utt.tokens[i].text) == canonical(utt.tokens[i + 1].text):
                labels[i] = REP
        return labels


# --------------------------------------------------------------------------
# Detektor berbasis aturan penuh
# --------------------------------------------------------------------------


@dataclass
class RuleBasedDetector:
    """Aturan sadar-konteks dengan pemisahan reduplikasi berbasis bukti."""

    lex: Lexicon
    name: str = "aturan"
    #: Putusan AMBIGUOUS diperlakukan sebagai reduplikasi (dipertahankan).
    #: Default konservatif ini melindungi makna: memotong reduplikasi
    #: merusak kalimat, sedangkan menyisakan satu repetisi hanya
    #: menyisakan sedikit kekasaran.
    ambiguous_is_reduplication: bool = True

    def predict(self, utt: Utterance) -> list[str]:
        labels = [O] * len(utt)
        _tag_multiword_fillers(utt, self.lex, labels)
        self._tag_single_fillers(utt, labels)
        self._tag_partial_words(utt, labels)
        self._tag_repetitions(utt, labels)
        _detect_repairs(utt, self.lex, labels)
        return labels

    # ------------------------------------------------------------------

    def _tag_single_fillers(self, utt: Utterance, labels: list[str]) -> None:
        words = [canonical(t.text) for t in utt.tokens]
        n = len(words)

        for i, w in enumerate(words):
            if labels[i] != O:
                continue
            cat = self.lex.category(w)
            if cat is None:
                continue

            if cat == CAT_FP:
                labels[i] = FP
                continue
            if cat == CAT_FP_AMB:
                # 'eh' dan kerabatnya diputuskan belakangan oleh detektor
                # ralat; di sini diperlakukan sebagai jeda terisi biasa.
                labels[i] = FP
                continue
            if cat == CAT_EDIT:
                continue  # ditangani `_detect_repairs`
            if cat == CAT_DM and self._dm_is_filler(words, labels, i, n):
                labels[i] = DM

    def _dm_is_filler(
        self, words: list[str], labels: list[str], i: int, n: int
    ) -> bool:
        """Penjagaan konteks agar penanda wacana bermakna tidak ikut terpotong.

        Sebagian besar kandidat DM Bahasa Indonesia bersifat homonim: kata
        yang sama bisa kosong makna atau penuh makna tergantung posisi dan
        tetangganya. Tanpa penjagaan ini, sistem memotong kata bermakna.
        """
        w = words[i]
        nxt = words[i + 1] if i < n - 1 else None
        prev_is_filler = i > 0 and labels[i - 1] in (FP, DM)
        at_start = i == 0 or all(labels[k] in (FP, DM) for k in range(i))

        # Pembuka giliran bicara: kosong makna hanya di awal ujaran.
        if w in {"jadi", "jadinya", "terus", "trus", "sebenarnya", "sebenernya", "sebetulnya"}:
            return at_start or prev_is_filler

        # 'ya' kosong makna hanya bila berdempet filler lain; berdiri
        # sendiri di awal ujaran ia adalah jawaban afirmatif.
        if w in {"ya", "yah", "iya"}:
            return prev_is_filler or (nxt is not None and self.lex.is_filler(nxt))

        # 'apa' penuh makna sebagai kata tanya; kosong makna saat penutur
        # sedang mencari kata, yang ditandai filler di sebelahnya.
        if w in {"apa", "gimana"}:
            return prev_is_filler or (nxt is not None and self.lex.is_filler(nxt))

        # 'gitu saja', 'gini aja' = demonstratif bermakna.
        if w in {"gitu", "gini"}:
            return not (nxt in DEMONSTRATIVE_TAIL)

        return True

    # ------------------------------------------------------------------

    def _tag_partial_words(self, utt: Utterance, labels: list[str]) -> None:
        n = len(utt)
        for i, tok in enumerate(utt.tokens):
            if labels[i] != O:
                continue
            nxt = utt.tokens[i + 1].text if i < n - 1 else None
            if is_partial_word(tok.text, nxt, utt.gap_after(i), self.lex):
                labels[i] = PW

    # ------------------------------------------------------------------

    def _tag_repetitions(self, utt: Utterance, labels: list[str]) -> None:
        for ev in find_repeat_events(utt, self.lex):
            if labels[ev.i] not in (O, DM, FP):
                continue
            # Token bertanda hubung yang sudah jelas reduplikasi gramatikal
            # tidak pernah dianggap repetisi walau muncul dua kali.
            if ev.kind == DISFLUENCY:
                labels[ev.i] = REP
            elif not self.ambiguous_is_reduplication and ev.kind != DISFLUENCY:
                if not is_hyphenated_reduplication(utt.tokens[ev.i].text, self.lex):
                    labels[ev.i] = REP
