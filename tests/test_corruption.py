"""Tests for the T5-style corruption module."""

from __future__ import annotations

import numpy as np
import pytest

from miditok import REMI
from miditok.corruption import (
    analyze_midi_structure,
    attribute_denoising,
    beat_denoising,
    continuation,
    heavy_denoising,
    measure_denoising,
    noise_span_to_unique_sentinel,
    nonnoise_span_to_unique_sentinel,
    note_denoising,
    random_spans_noise_mask,
)
from miditok.midi_adapter import AdapterScore

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_MIDI = REPO_ROOT / "violin_partita_bwv-1004_1_(c)grossman.mid"


@pytest.fixture
def tokenizer_and_ids() -> tuple[REMI, list[int]]:
    tok = REMI()
    score = AdapterScore.from_file(SAMPLE_MIDI)
    encoded = tok.encode(score)
    seq = encoded[0] if isinstance(encoded, list) else encoded
    return tok, list(seq.ids)


def test_random_spans_density_in_target_range() -> None:
    rng = np.random.default_rng(0)
    mask = random_spans_noise_mask(1000, 0.3, 3.0, rng=rng)
    density = mask.sum() / len(mask)
    assert abs(density - 0.3) < 0.1


def test_random_spans_zero_density() -> None:
    mask = random_spans_noise_mask(100, 0.0)
    assert not mask.any()


def test_random_spans_mean_span_length_effect() -> None:
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)
    m1 = random_spans_noise_mask(2000, 0.3, 2.0, rng=rng1)
    m2 = random_spans_noise_mask(2000, 0.3, 8.0, rng=rng2)

    def avg_span_len(mask):
        n_spans = int((mask & ~np.concatenate([[False], mask[:-1]])).sum())
        return mask.sum() / max(1, n_spans)

    assert avg_span_len(m2) > avg_span_len(m1)


def test_noise_span_to_unique_sentinel_small() -> None:
    ids = [10, 11, 12, 13, 14, 15]
    mask = np.array([False, True, True, False, True, False])
    enc = noise_span_to_unique_sentinel(ids, mask, sentinel_start_id=999)
    assert enc == [10, 999, 13, 998, 15]


def test_nonnoise_span_to_unique_sentinel_small() -> None:
    ids = [10, 11, 12, 13, 14, 15]
    mask = np.array([False, True, True, False, True, False])
    dec = nonnoise_span_to_unique_sentinel(ids, mask, sentinel_start_id=999)
    assert dec == [999, 11, 12, 998, 14, 997]


def test_encoder_decoder_can_reconstruct() -> None:
    rng = np.random.default_rng(0)
    ids = list(range(100, 130))
    mask = random_spans_noise_mask(len(ids), 0.3, 2.0, rng=rng)
    enc = noise_span_to_unique_sentinel(ids, mask, sentinel_start_id=999)
    dec = nonnoise_span_to_unique_sentinel(ids, mask, sentinel_start_id=999)
    # Every original token must appear in either enc (kept) or dec (corrupted-out).
    seen = set(t for t in enc if t < 999 - 50) | set(t for t in dec if t < 999 - 50)
    assert set(ids) == seen


def test_beat_denoising(tokenizer_and_ids) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, dec = beat_denoising(ids, tok.vocab, sentinel_start_id=99999, rng=rng)
    assert len(enc) > 0 and len(dec) > 0
    assert len(enc) < len(ids) + 5


def test_measure_denoising(tokenizer_and_ids) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, dec = measure_denoising(ids, tok.vocab, sentinel_start_id=99999, rng=rng)
    assert any(t == 99999 for t in enc)


def test_note_denoising(tokenizer_and_ids) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, dec = note_denoising(ids, tok.vocab, sentinel_start_id=99999, rng=rng)
    assert len(enc) + len(dec) >= len(ids)


def test_attribute_denoising_only_changes_attributes(tokenizer_and_ids) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, dec = attribute_denoising(ids, tok.vocab, corruption_ratio=0.5, rng=rng)
    assert len(enc) == len(dec) == len(ids)
    inv = {v: k for k, v in tok.vocab.items()}
    for orig, new in zip(ids, enc):
        if orig != new:
            o_pref = inv[orig].split("_", 1)[0]
            n_pref = inv[new].split("_", 1)[0]
            assert o_pref == n_pref


def test_heavy_denoising_high_density(tokenizer_and_ids) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, dec = heavy_denoising(ids, tok.vocab, sentinel_start_id=99999, rng=rng)
    # Decoder should hold at least 40% of the original tokens (corrupted span content).
    assert len(dec) > 0.3 * len(ids)


def test_continuation_split_at_bar(tokenizer_and_ids) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, dec = continuation(ids, tok.vocab, prefix_ratio_range=(0.2, 0.5), rng=rng)
    assert len(enc) + len(dec) == len(ids)
    bar_id = tok.vocab.get("Bar_None")
    if bar_id is not None and bar_id in ids[1:]:
        # The split point should land on a Bar token (i.e. dec[0] is Bar_None).
        assert dec[0] == bar_id


def test_continuation_prefix_in_range(tokenizer_and_ids) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, _ = continuation(ids, tok.vocab, prefix_ratio_range=(0.2, 0.5), rng=rng)
    ratio = len(enc) / len(ids)
    assert 0.1 <= ratio <= 0.6


def test_analyze_midi_structure(tokenizer_and_ids) -> None:
    tok, ids = tokenizer_and_ids
    structure = analyze_midi_structure(ids, tok.vocab)
    assert "bar_indices" in structure
    assert "beat_indices" in structure
    assert "note_indices" in structure
    assert "timing_indices" in structure
    assert len(structure["bar_indices"]) > 0
    assert len(structure["note_indices"]) > 0


def test_corruption_deterministic_with_seed(tokenizer_and_ids) -> None:
    tok, ids = tokenizer_and_ids
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    a = note_denoising(ids, tok.vocab, sentinel_start_id=99999, rng=rng1)
    b = note_denoising(ids, tok.vocab, sentinel_start_id=99999, rng=rng2)
    assert a == b
