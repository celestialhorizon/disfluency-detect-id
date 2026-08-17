"""Uji ekstraksi fitur dan penanda sekuens perceptron."""

from disfluency_id.features import FeatureConfig, bucket, featurize
from disfluency_id.model import DisfluencyTagger, StructuredPerceptron
from disfluency_id.schema import DISFLUENT_LABELS, O


# --------------------------------------------------------------------------
# Fitur
# --------------------------------------------------------------------------


def test_bucket_monotonic():
    edges = (0.1, 0.2, 0.3)
    assert bucket(0.05, edges) == "b0"
    assert bucket(0.15, edges) == "b1"
    assert bucket(0.99, edges) == "b3"


def test_featurize_returns_one_row_per_token(lex, plain_utt):
    utt = plain_utt("kami membandingkan tiga model dengan konfigurasi sama")
    feats = featurize(utt, lex)
    assert len(feats) == len(utt)
    assert all(f for f in feats)


def test_feature_groups_can_be_switched_off(lex, plain_utt):
    utt = plain_utt("anak anak sekarang lebih mandiri")
    full = featurize(utt, lex, FeatureConfig(True, True, True))
    lex_only = featurize(utt, lex, FeatureConfig(True, False, False))

    assert all(len(a) > len(b) for a, b in zip(full, lex_only))
    assert not any(f.startswith("gapb=") for row in lex_only for f in row)
    assert not any(f.startswith("eqprev=") for row in lex_only for f in row)


def test_prosody_features_present_when_enabled(lex, plain_utt):
    utt = plain_utt("saya mau bertanya")
    rows = featurize(utt, lex, FeatureConfig(False, True, False))
    assert any(f.startswith("gapb=") for f in rows[1])


def test_reduplication_verdict_reaches_the_features(lex, make_utt):
    utt = make_utt([("saya", O, 0.05), ("saya", O, 0.30), ("mau", O, 0.05)])
    rows = featurize(utt, lex, FeatureConfig(True, True, True))
    assert "verdict=disfluency" in rows[0]


def test_repair_structure_reaches_the_features(lex, make_utt):
    utt = make_utt(
        [
            ("saya", O, 0.05),
            ("ke", O, 0.05),
            ("pasar", O, 0.05),
            ("eh", O, 0.15),
            ("ke", O, 0.10),
            ("toko", O, 0.05),
        ]
    )
    rows = featurize(utt, lex, FeatureConfig(True, False, False))
    assert "in_reparandum=True" in rows[1]
    assert "is_editterm=True" in rows[3]


def test_feature_config_name():
    assert FeatureConfig(True, True, True).name == "lex+pros+redup"
    assert FeatureConfig(True, False, False).name == "lex"


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def test_decode_on_empty_sequence():
    assert StructuredPerceptron().decode([]) == []


def test_untrained_model_produces_valid_labels(lex, plain_utt):
    utt = plain_utt("kami membandingkan tiga model")
    preds = DisfluencyTagger(lex).predict(utt)
    assert len(preds) == len(utt)
    assert set(preds) <= DISFLUENT_LABELS | {O}


def test_training_reduces_error(lex, corpus):
    tagger = DisfluencyTagger(lex).fit(corpus[:80], epochs=8, seed=3)
    history = tagger.train_history
    assert len(history) == 8
    assert history[-1] < history[0], history


def test_model_fits_its_training_data_reasonably(lex, corpus):
    """Model harus mampu mencocokkan data latihnya sendiri.

    Ini bukan ukuran generalisasi, melainkan pemeriksaan bahwa pelatihan
    benar-benar bekerja: model yang tidak bisa menghafal data latihnya
    berarti ada cacat pada pembaruan bobot atau dekode.
    """
    train = corpus[:100]
    tagger = DisfluencyTagger(lex).fit(train, epochs=15, seed=3)
    correct = total = 0
    for utt in train:
        for pred, gold in zip(tagger.predict(utt), utt.labels):
            correct += pred == gold
            total += 1
    assert correct / total > 0.90


def test_predictions_are_deterministic(lex, corpus):
    tagger = DisfluencyTagger(lex).fit(corpus[:60], epochs=5, seed=3)
    utt = corpus[70]
    assert tagger.predict(utt) == tagger.predict(utt)


def test_same_seed_gives_same_model(lex, corpus):
    a = DisfluencyTagger(lex).fit(corpus[:60], epochs=5, seed=3)
    b = DisfluencyTagger(lex).fit(corpus[:60], epochs=5, seed=3)
    utt = corpus[70]
    assert a.predict(utt) == b.predict(utt)


def test_save_and_load_round_trip(lex, corpus, tmp_path):
    tagger = DisfluencyTagger(lex).fit(corpus[:60], epochs=5, seed=3)
    path = tmp_path / "model.json"
    tagger.save(path)

    restored = DisfluencyTagger.load(path, lex=lex)
    assert restored.cfg == tagger.cfg
    for utt in corpus[60:80]:
        assert restored.predict(utt) == tagger.predict(utt)


def test_averaging_runs_once(lex, corpus):
    tagger = DisfluencyTagger(lex).fit(corpus[:40], epochs=3, seed=3)
    assert tagger.model.averaged
