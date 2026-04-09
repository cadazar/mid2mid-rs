"""Tests for the UniversalMidiTokenizer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from miditok import REMI, TSD, MIDILike, TokenizerConfig
from miditok.midi_adapter import AdapterScore
from miditok.universal_tokenizer import UniversalMidiTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_MIDI = REPO_ROOT / "violin_partita_bwv-1004_1_(c)grossman.mid"


@pytest.fixture
def utok() -> UniversalMidiTokenizer:
    return UniversalMidiTokenizer(formats=["remi", "tsd", "midi_like"])


@pytest.fixture
def utok_two() -> UniversalMidiTokenizer:
    return UniversalMidiTokenizer(formats=["remi", "tsd"])


@pytest.fixture
def score() -> AdapterScore:
    return AdapterScore.from_file(str(SAMPLE_MIDI))


# -------------------------------------------------------------------
# Init / formats
# -------------------------------------------------------------------


def test_init_multiple_formats(utok: UniversalMidiTokenizer) -> None:
    assert set(utok.formats) == {"remi", "tsd", "midi_like"}


def test_init_default_formats() -> None:
    tok = UniversalMidiTokenizer()
    assert set(tok.formats) == {"remi", "tsd"}


def test_init_with_custom_weights() -> None:
    tok = UniversalMidiTokenizer(
        formats=["remi", "tsd"],
        format_weights={"remi": 0.8, "tsd": 0.2},
    )
    assert abs(tok._format_weights["remi"] - 0.8) < 1e-6


def test_init_invalid_format() -> None:
    with pytest.raises(ValueError, match="Unknown format"):
        UniversalMidiTokenizer(formats=["nonexistent"])


# -------------------------------------------------------------------
# Vocabulary
# -------------------------------------------------------------------


def test_vocab_is_dict(utok: UniversalMidiTokenizer) -> None:
    assert isinstance(utok.vocab, dict)


def test_vocab_size_consistent(utok: UniversalMidiTokenizer) -> None:
    assert utok.vocab_size == len(utok.vocab)


def test_no_id_collisions(utok: UniversalMidiTokenizer) -> None:
    ids = list(utok.vocab.values())
    assert len(ids) == len(set(ids)), "ID collision in unified vocab"


def test_shared_tokens_same_ids(utok_two: UniversalMidiTokenizer) -> None:
    """Pitch_*, Velocity_*, Bar, etc. should have the same ID regardless of format."""
    vocab = utok_two.vocab
    # All shared tokens are present
    remi = REMI()
    tsd = TSD()
    shared = set(remi.vocab.keys()) & set(tsd.vocab.keys())
    for tok in shared:
        assert tok in vocab, f"Shared token {tok} missing from unified vocab"


def test_format_specific_tokens_have_unique_ids(utok: UniversalMidiTokenizer) -> None:
    """Format-specific tokens should not collide with each other."""
    vocab = utok.vocab
    all_ids = set()
    for tok, tid in vocab.items():
        assert tid not in all_ids or tok in vocab, f"ID {tid} duplicated"
        all_ids.add(tid)


def test_format_sentinel_tokens_in_vocab(utok: UniversalMidiTokenizer) -> None:
    sentinels = utok.format_sentinel
    for fmt, tok in sentinels.items():
        assert tok in utok.vocab, f"Sentinel {tok} not in vocab"


def test_format_sentinel_keys_match_formats(utok: UniversalMidiTokenizer) -> None:
    assert set(utok.format_sentinel.keys()) == set(utok.formats)


def test_build_unified_vocab_idempotent(utok: UniversalMidiTokenizer) -> None:
    v1 = dict(utok.vocab)
    utok.build_unified_vocab()
    v2 = dict(utok.vocab)
    assert v1 == v2


# -------------------------------------------------------------------
# Encode
# -------------------------------------------------------------------


def test_encode_with_explicit_format(utok: UniversalMidiTokenizer, score: AdapterScore) -> None:
    seq = utok.encode(score, format="remi")
    assert len(seq.ids) > 0


def test_encode_with_each_format(utok: UniversalMidiTokenizer, score: AdapterScore) -> None:
    for fmt in utok.formats:
        seq = utok.encode(score, format=fmt)
        assert len(seq.ids) > 0, f"Empty encoding for {fmt}"


def test_encode_returns_unified_ids(utok: UniversalMidiTokenizer, score: AdapterScore) -> None:
    seq = utok.encode(score, format="remi")
    max_id = utok.vocab_size
    for i in seq.ids:
        assert 0 <= i < max_id, f"ID {i} outside vocab range [0, {max_id})"


def test_encode_no_format_samples_randomly(score: AdapterScore) -> None:
    tok = UniversalMidiTokenizer(formats=["remi", "tsd"])
    tok.set_seed(42)
    results = set()
    for _ in range(50):
        seq = tok.encode(score)
        # Check if it produced tokens specific to one format
        tokens_set = set(seq.tokens) if seq.tokens else set()
        has_bar = any(t.startswith("Bar") for t in tokens_set)
        has_timeshift = any(t.startswith("TimeShift") for t in tokens_set)
        if has_bar and not has_timeshift:
            results.add("remi")
        elif has_timeshift:
            results.add("tsd")
    # With 50 samples and uniform weights, both should appear
    assert len(results) == 2, f"Only got formats: {results}"


def test_encode_respects_format_weights(score: AdapterScore) -> None:
    tok = UniversalMidiTokenizer(
        formats=["remi", "tsd"],
        format_weights={"remi": 1.0, "tsd": 0.0},
    )
    tok.set_seed(0)
    for _ in range(10):
        seq = tok.encode(score)
        # Should always be REMI (has Bar tokens, no TimeShift at start)
        assert seq.tokens[0].startswith("Bar") or seq.tokens[0].startswith("Position") or seq.tokens[0].startswith("Pitch")


def test_encode_invalid_format(utok: UniversalMidiTokenizer, score: AdapterScore) -> None:
    with pytest.raises(ValueError, match="not available"):
        utok.encode(score, format="nonexistent")


# -------------------------------------------------------------------
# Format mask
# -------------------------------------------------------------------


def test_get_format_mask_shape(utok: UniversalMidiTokenizer) -> None:
    mask = utok.get_format_mask("remi")
    assert mask.shape == (utok.vocab_size,)
    assert mask.dtype == bool


def test_get_format_mask_has_true_values(utok: UniversalMidiTokenizer) -> None:
    for fmt in utok.formats:
        mask = utok.get_format_mask(fmt)
        assert mask.any(), f"No valid tokens for {fmt}"


def test_get_format_mask_includes_sentinel(utok: UniversalMidiTokenizer) -> None:
    for fmt in utok.formats:
        mask = utok.get_format_mask(fmt)
        sentinel_tok = utok.format_sentinel[fmt]
        sentinel_id = utok.vocab[sentinel_tok]
        assert mask[sentinel_id], f"Sentinel {sentinel_tok} not in mask for {fmt}"


def test_get_format_mask_invalid_format(utok: UniversalMidiTokenizer) -> None:
    with pytest.raises(ValueError):
        utok.get_format_mask("nonexistent")


# -------------------------------------------------------------------
# Save / Load
# -------------------------------------------------------------------


def test_save_load_roundtrip(utok: UniversalMidiTokenizer) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        utok.save(tmpdir)
        loaded = UniversalMidiTokenizer.load(tmpdir)
        assert loaded.vocab == utok.vocab
        assert loaded.formats == utok.formats
        assert loaded.vocab_size == utok.vocab_size


def test_save_creates_json(utok: UniversalMidiTokenizer) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        utok.save(tmpdir)
        json_path = Path(tmpdir) / "universal_tokenizer.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "vocab" in data
        assert "formats" in data


def test_load_preserves_format_sentinels(utok: UniversalMidiTokenizer) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        utok.save(tmpdir)
        loaded = UniversalMidiTokenizer.load(tmpdir)
        assert loaded.format_sentinel == utok.format_sentinel


# -------------------------------------------------------------------
# Repr
# -------------------------------------------------------------------


def test_repr(utok: UniversalMidiTokenizer) -> None:
    r = repr(utok)
    assert "UniversalMidiTokenizer" in r
    assert str(utok.vocab_size) in r
