"""Fixture bersama untuk seluruh berkas uji.

Keberadaan berkas ini juga yang membuat direktori `prototype/` masuk ke
sys.path, sehingga `import disfluency_id` bekerja tanpa pemasangan paket.
"""

import random

import pytest

from disfluency_id.corpus import TimingProfile, build_corpus, simulate_timings
from disfluency_id.lexicon import Lexicon
from disfluency_id.schema import O, Utterance


@pytest.fixture(scope="session")
def lex() -> Lexicon:
    return Lexicon.load()


@pytest.fixture(scope="session")
def corpus():
    return build_corpus(seed=20260812)


@pytest.fixture
def make_utt():
    """Bangun ujaran dari daftar (kata, label, jeda_sebelum) dengan waktu eksplisit.

    Jeda dikendalikan penuh agar uji terhadap bukti prosodi bersifat
    deterministik, tidak bergantung pada simulator.
    """

    def _make(spec, uid="uji-0001", dur=0.30):
        tokens = []
        t = 0.5
        for item in spec:
            if len(item) == 3:
                text, label, gap = item
            else:
                text, label = item
                gap = 0.04
            t += gap
            from disfluency_id.schema import Token

            tokens.append(Token(text=text, start=round(t, 3), end=round(t + dur, 3), label=label))
            t += dur
        return Utterance(uid=uid, tokens=tokens, source="uji")

    return _make


@pytest.fixture
def plain_utt():
    """Ujaran dari teks polos dengan penanda waktu bawaan simulator."""

    def _make(text, uid="uji-polos", seed=7):
        words = [(w, O) for w in text.split()]
        rng = random.Random(seed)
        return Utterance(
            uid=uid,
            tokens=simulate_timings(words, rng, TimingProfile()),
            source="uji",
        )

    return _make
