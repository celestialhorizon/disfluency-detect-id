"""Uji impor transkrip ASR menjadi objek Utterance."""

import json

import pytest

from disfluency_id.ingest import check_dependencies, from_whisper_json


def _write(tmp_path, payload, name="asr.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_reads_openai_whisper_segment_format(tmp_path):
    path = _write(
        tmp_path,
        {
            "segments": [
                {
                    "words": [
                        {"word": " saya", "start": 0.0, "end": 0.4},
                        {"word": " mau", "start": 0.45, "end": 0.7},
                    ]
                }
            ]
        },
    )
    utts = from_whisper_json(path)
    assert len(utts) == 1
    assert [t.text for t in utts[0]] == ["saya", "mau"]


def test_reads_flat_words_format(tmp_path):
    path = _write(
        tmp_path,
        {"words": [{"word": "halo", "start": 0.0, "end": 0.3}]},
    )
    assert from_whisper_json(path)[0].text == "halo"


def test_reads_bare_list_format(tmp_path):
    path = _write(tmp_path, [{"text": "halo", "start": 0.0, "end": 0.3}])
    assert from_whisper_json(path)[0].text == "halo"


def test_splits_utterances_on_long_silence(tmp_path):
    path = _write(
        tmp_path,
        {
            "words": [
                {"word": "satu", "start": 0.0, "end": 0.3},
                {"word": "dua", "start": 0.35, "end": 0.6},
                {"word": "tiga", "start": 2.5, "end": 2.8},
            ]
        },
    )
    utts = from_whisper_json(path, max_gap=0.8)
    assert len(utts) == 2
    assert [t.text for t in utts[0]] == ["satu", "dua"]
    assert [t.text for t in utts[1]] == ["tiga"]


def test_hesitation_pause_does_not_split_an_utterance(tmp_path):
    """Jeda hesitasi di tengah kalimat justru objek penelitian ini;
    ia tidak boleh dipotong menjadi dua ujaran."""
    path = _write(
        tmp_path,
        {
            "words": [
                {"word": "saya", "start": 0.0, "end": 0.3},
                {"word": "eee", "start": 0.7, "end": 1.2},
                {"word": "setuju", "start": 1.3, "end": 1.7},
            ]
        },
    )
    assert len(from_whisper_json(path, max_gap=0.8)) == 1


def test_imported_tokens_are_unlabelled(tmp_path):
    """ASR tidak menghasilkan label disfluensi; itu tugas anotator."""
    path = _write(tmp_path, {"words": [{"word": "halo", "start": 0.0, "end": 0.3}]})
    assert all(t.label == "O" for t in from_whisper_json(path)[0])


def test_timings_survive_the_import(tmp_path):
    path = _write(
        tmp_path,
        {"words": [{"word": "halo", "start": 1.25, "end": 1.80}]},
    )
    tok = from_whisper_json(path)[0].tokens[0]
    assert tok.start == pytest.approx(1.25)
    assert tok.end == pytest.approx(1.80)


def test_empty_or_wordless_payload_raises(tmp_path):
    path = _write(tmp_path, {"segments": [{"text": "tanpa penanda waktu"}]})
    with pytest.raises(ValueError, match="word_timestamps"):
        from_whisper_json(path)


def test_blank_tokens_are_dropped(tmp_path):
    path = _write(
        tmp_path,
        {
            "words": [
                {"word": "  ", "start": 0.0, "end": 0.1},
                {"word": "halo", "start": 0.1, "end": 0.4},
            ]
        },
    )
    assert [t.text for t in from_whisper_json(path)[0]] == ["halo"]


def test_local_media_missing_file_fails_before_loading_whisper(tmp_path):
    """Salah ketik nama berkas harus ketahuan seketika, bukan setelah
    menunggu model ASR dimuat."""
    from disfluency_id.ingest import ingest_media

    with pytest.raises(FileNotFoundError, match="tidak ditemukan"):
        ingest_media(tmp_path / "tidak-ada.mp4")


def test_local_media_uses_the_file_name_as_source(tmp_path, monkeypatch):
    from disfluency_id import ingest as ing

    media = tmp_path / "podcast-sampel.mp4"
    media.write_bytes(b"bukan mp4 sungguhan")
    js = _write(tmp_path, {"words": [{"word": "halo", "start": 0.0, "end": 0.3}]})

    monkeypatch.setattr(ing, "transcribe", lambda *a, **k: js)
    utts = ing.ingest_media(media)

    assert utts[0].source == "podcast-sampel.mp4"
    assert all(t.label == "O" for t in utts[0])


def test_hyphen_split_reduplication_is_rejoined(tmp_path):
    """Whisper memecah 'kira-kira' jadi ' kira' + '-kira'; korpus harus
    menerimanya kembali sebagai satu kata."""
    path = _write(
        tmp_path,
        {
            "words": [
                {"word": " kira", "start": 1.00, "end": 1.30},
                {"word": "-kira", "start": 1.30, "end": 1.60},
                {"word": " saja", "start": 1.70, "end": 2.00},
            ]
        },
    )
    toks = from_whisper_json(path)[0].tokens
    assert [t.text for t in toks] == ["kira-kira", "saja"]


def test_rejoined_token_spans_both_halves(tmp_path):
    path = _write(
        tmp_path,
        {
            "words": [
                {"word": " sama", "start": 2.00, "end": 2.40},
                {"word": "-sama", "start": 2.40, "end": 2.90},
            ]
        },
    )
    tok = from_whisper_json(path)[0].tokens[0]
    assert tok.start == pytest.approx(2.00)
    assert tok.end == pytest.approx(2.90)


def test_rejoining_removes_the_fake_zero_gap(tmp_path):
    """Batas semu di tengah kata jangan sampai terhitung sebagai jeda
    antar-kata; angka itu tidak pernah diukur dari audio."""
    from disfluency_id.asr_audit import audit

    payload = {
        "words": [
            {"word": " kira", "start": 0.00, "end": 0.30},
            {"word": "-kira", "start": 0.30, "end": 0.60},
        ]
    }
    path = _write(tmp_path, payload)
    assert audit(from_whisper_json(path)).gaps.total == 0
    assert audit(from_whisper_json(path, merge_hyphens=False)).gaps.zero == 1


def test_rejoining_can_be_switched_off_for_ablation(tmp_path):
    path = _write(
        tmp_path,
        {
            "words": [
                {"word": " kira", "start": 1.00, "end": 1.30},
                {"word": "-kira", "start": 1.30, "end": 1.60},
            ]
        },
    )
    toks = from_whisper_json(path, merge_hyphens=False)[0].tokens
    assert [t.text for t in toks] == ["kira", "-kira"]


def test_trailing_hyphen_is_left_alone(tmp_path):
    """Tanda hubung di akhir adalah lambang kata terpotong pada pedoman
    anotasi; menyatukannya menghapus fenomena yang diteliti."""
    path = _write(
        tmp_path,
        {
            "words": [
                {"word": " sa-", "start": 0.00, "end": 0.20},
                {"word": " sapi", "start": 0.25, "end": 0.60},
            ]
        },
    )
    assert [t.text for t in from_whisper_json(path)[0].tokens] == ["sa-", "sapi"]


def test_lone_hyphen_token_is_not_merged(tmp_path):
    """Tanda hubung telanjang tidak membawa huruf apa pun untuk disambung."""
    from disfluency_id.ingest import merge_hyphen_continuations
    from disfluency_id.schema import Token

    toks = [Token("kira", 0.0, 0.3), Token("-", 0.3, 0.35), Token("kira", 0.35, 0.65)]
    assert [t.text for t in merge_hyphen_continuations(toks)] == ["kira", "-", "kira"]


def test_merging_an_empty_or_single_token_list_is_safe():
    from disfluency_id.ingest import merge_hyphen_continuations
    from disfluency_id.schema import Token

    assert merge_hyphen_continuations([]) == []
    assert len(merge_hyphen_continuations([Token("halo", 0.0, 0.3)])) == 1


def test_dependency_check_reports_every_optional_library():
    status = check_dependencies()
    assert set(status) == {"transformers", "torch"}
    assert all(isinstance(v, bool) for v in status.values())


def test_ingest_offers_no_download_path():
    """Satu-satunya jalur unduh ada di media.py, yang menuntut provenans.

    Uji ini mengikat klaim itu: begitu `ingest` punya pintu unduh sendiri lagi,
    syarat "provenans diisi sebelum berkas diproses" berhenti berlaku dengan
    sendirinya dan kembali jadi aturan yang harus diingat.
    """
    import disfluency_id.ingest as modul

    assert not hasattr(modul, "download_audio")
    assert not hasattr(modul, "ingest")
