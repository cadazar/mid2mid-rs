"""Tests for the T5-style corruption module (increasing sentinel convention)."""

from __future__ import annotations

import numpy as np
import pytest

from miditok import REMI
from miditok.corruption import (
    analyze_midi_structure,
    apply_span_corruption,
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

# Sentinel IDs for testing: index i → token id 9000 + i
SENTINEL_IDS = [9000 + i for i in range(4096)]


@pytest.fixture
def tokenizer_and_ids() -> tuple[REMI, list[int]]:
    tok = REMI()
    score = AdapterScore.from_file(str(SAMPLE_MIDI))
    encoded = tok.encode(score)
    seq = encoded[0] if isinstance(encoded, list) else encoded
    return tok, list(seq.ids)


# -------------------------------------------------------------------
# random_spans_noise_mask
# -------------------------------------------------------------------


def test_random_spans_density_in_target_range() -> None:
    rng = np.random.default_rng(0)
    mask = random_spans_noise_mask(1000, 0.3, 3.0, rng=rng)
    density = mask.sum() / len(mask)
    assert abs(density - 0.3) < 0.1


def test_random_spans_zero_density() -> None:
    mask = random_spans_noise_mask(100, 0.0)
    assert not mask.any()


def test_random_spans_single_token() -> None:
    mask = random_spans_noise_mask(1, 0.5)
    assert not mask.any()  # length=1 → no corruption


def test_random_spans_length_zero() -> None:
    mask = random_spans_noise_mask(0, 0.5)
    assert len(mask) == 0


def test_random_spans_mean_span_length_effect() -> None:
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)
    m1 = random_spans_noise_mask(2000, 0.3, 2.0, rng=rng1)
    m2 = random_spans_noise_mask(2000, 0.3, 8.0, rng=rng2)

    def avg_span_len(mask: np.ndarray) -> float:
        n_spans = int((mask & ~np.concatenate([[False], mask[:-1]])).sum())
        return float(mask.sum()) / max(1, n_spans)

    assert avg_span_len(m2) > avg_span_len(m1)


def test_random_spans_high_density() -> None:
    rng = np.random.default_rng(0)
    mask = random_spans_noise_mask(100, 0.9, 3.0, rng=rng)
    density = mask.sum() / len(mask)
    assert density > 0.7


def test_random_spans_deterministic_with_seed() -> None:
    m1 = random_spans_noise_mask(500, 0.3, 3.0, rng=np.random.default_rng(42))
    m2 = random_spans_noise_mask(500, 0.3, 3.0, rng=np.random.default_rng(42))
    assert np.array_equal(m1, m2)


# -------------------------------------------------------------------
# noise_span_to_unique_sentinel / nonnoise_span_to_unique_sentinel
# -------------------------------------------------------------------


def test_noise_span_increasing_sentinels() -> None:
    ids = [10, 11, 12, 13, 14, 15]
    mask = np.array([False, True, True, False, True, False])
    enc, next_idx = noise_span_to_unique_sentinel(ids, mask, SENTINEL_IDS, start_sentinel_idx=0)
    # First span → sentinel_ids[0] = 9000, second span → sentinel_ids[1] = 9001
    assert enc == [10, 9000, 13, 9001, 15]
    assert next_idx == 2


def test_nonnoise_span_increasing_sentinels() -> None:
    ids = [10, 11, 12, 13, 14, 15]
    mask = np.array([False, True, True, False, True, False])
    dec, next_idx = nonnoise_span_to_unique_sentinel(ids, mask, SENTINEL_IDS, start_sentinel_idx=0)
    # Inverted: non-noise spans are [10], [13], [15] → replaced with sentinels
    assert dec == [9000, 11, 12, 9001, 14, 9002]
    assert next_idx == 3


def test_sentinel_ids_are_increasing() -> None:
    """Verify sentinels are assigned in increasing order (extra_id_0, extra_id_1, ...)."""
    rng = np.random.default_rng(0)
    ids = list(range(100))
    mask = random_spans_noise_mask(100, 0.3, 3.0, rng=rng)
    enc, _ = noise_span_to_unique_sentinel(ids, mask, SENTINEL_IDS, start_sentinel_idx=0)
    sentinel_vals = [t for t in enc if t >= 9000]
    # Should be increasing
    for i in range(1, len(sentinel_vals)):
        assert sentinel_vals[i] > sentinel_vals[i - 1]


def test_start_sentinel_idx_chaining() -> None:
    """Chaining: start_sentinel_idx from first call feeds into second."""
    ids1 = [10, 11, 12]
    mask1 = np.array([False, True, False])
    _, next1 = noise_span_to_unique_sentinel(ids1, mask1, SENTINEL_IDS, start_sentinel_idx=0)
    assert next1 == 1

    ids2 = [20, 21, 22]
    mask2 = np.array([True, False, True])
    enc2, next2 = noise_span_to_unique_sentinel(ids2, mask2, SENTINEL_IDS, start_sentinel_idx=next1)
    # Should use sentinel_ids[1] = 9001 and sentinel_ids[2] = 9002
    assert 9001 in enc2
    assert next2 == 3


def test_encoder_decoder_can_reconstruct() -> None:
    rng = np.random.default_rng(0)
    ids = list(range(100, 130))
    mask = random_spans_noise_mask(len(ids), 0.3, 2.0, rng=rng)
    enc, _ = noise_span_to_unique_sentinel(ids, mask, SENTINEL_IDS, start_sentinel_idx=0)
    dec, _ = nonnoise_span_to_unique_sentinel(ids, mask, SENTINEL_IDS, start_sentinel_idx=0)
    # Every original token must appear in either enc or dec
    sentinel_set = set(SENTINEL_IDS)
    seen = set(t for t in enc if t not in sentinel_set) | set(t for t in dec if t not in sentinel_set)
    assert set(ids) == seen


def test_empty_sequence() -> None:
    enc, next_idx = noise_span_to_unique_sentinel([], np.array([], dtype=bool), SENTINEL_IDS)
    assert enc == []
    assert next_idx == 0


# -------------------------------------------------------------------
# apply_span_corruption
# -------------------------------------------------------------------


def test_apply_span_corruption_returns_three_tuple() -> None:
    rng = np.random.default_rng(0)
    ids = list(range(50))
    enc, dec, next_idx = apply_span_corruption(ids, 0.3, 3.0, SENTINEL_IDS, rng=rng)
    assert isinstance(enc, list)
    assert isinstance(dec, list)
    assert isinstance(next_idx, (int, np.integer))
    assert next_idx > 0


def test_apply_span_corruption_sentinels_match() -> None:
    rng = np.random.default_rng(0)
    ids = list(range(50))
    enc, dec, _ = apply_span_corruption(ids, 0.3, 3.0, SENTINEL_IDS, rng=rng)
    sentinel_set = set(SENTINEL_IDS)
    enc_sentinels = sorted(t for t in enc if t in sentinel_set)
    dec_sentinels = sorted(t for t in dec if t in sentinel_set)
    # Encoder and decoder should use the same sentinel IDs
    assert enc_sentinels == dec_sentinels


# -------------------------------------------------------------------
# Music-aware strategies
# -------------------------------------------------------------------


def test_beat_denoising(tokenizer_and_ids: tuple) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, dec, next_idx = beat_denoising(ids, tok.vocab, SENTINEL_IDS, rng=rng)
    assert len(enc) > 0 and len(dec) > 0
    assert next_idx > 0


def test_measure_denoising(tokenizer_and_ids: tuple) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, dec, next_idx = measure_denoising(ids, tok.vocab, SENTINEL_IDS, rng=rng)
    assert any(t in SENTINEL_IDS for t in enc)
    assert next_idx > 0


def test_note_denoising(tokenizer_and_ids: tuple) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, dec, next_idx = note_denoising(ids, tok.vocab, SENTINEL_IDS, rng=rng)
    assert len(enc) + len(dec) >= len(ids)
    assert next_idx > 0


def test_heavy_denoising_high_density(tokenizer_and_ids: tuple) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, dec, next_idx = heavy_denoising(ids, tok.vocab, SENTINEL_IDS, rng=rng)
    assert len(dec) > 0.3 * len(ids)
    assert next_idx > 0


def test_attribute_denoising_only_changes_attributes(tokenizer_and_ids: tuple) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, dec, next_idx = attribute_denoising(
        ids, tok.vocab, SENTINEL_IDS, corruption_ratio=0.5, rng=rng,
    )
    assert len(enc) == len(dec) == len(ids)
    assert next_idx == 0  # No sentinels used
    inv = {v: k for k, v in tok.vocab.items()}
    for orig, new in zip(ids, enc):
        if orig != new:
            o_pref = inv[orig].split("_", 1)[0]
            n_pref = inv[new].split("_", 1)[0]
            assert o_pref == n_pref


def test_attribute_denoising_no_sentinels(tokenizer_and_ids: tuple) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, dec, next_idx = attribute_denoising(ids, tok.vocab, SENTINEL_IDS, rng=rng)
    sentinel_set = set(SENTINEL_IDS)
    assert not any(t in sentinel_set for t in enc)
    assert not any(t in sentinel_set for t in dec)
    assert next_idx == 0


def test_continuation_split(tokenizer_and_ids: tuple) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, dec, next_idx = continuation(
        ids, tok.vocab, SENTINEL_IDS, prefix_ratio_range=(0.2, 0.5), rng=rng,
    )
    assert len(enc) + len(dec) == len(ids)
    assert next_idx == 0  # No sentinels used


def test_continuation_prefix_in_range(tokenizer_and_ids: tuple) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, _, _ = continuation(
        ids, tok.vocab, SENTINEL_IDS, prefix_ratio_range=(0.2, 0.5), rng=rng,
    )
    ratio = len(enc) / len(ids)
    assert 0.1 <= ratio <= 0.6


def test_continuation_no_sentinels(tokenizer_and_ids: tuple) -> None:
    tok, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    enc, dec, _ = continuation(ids, tok.vocab, SENTINEL_IDS, rng=rng)
    sentinel_set = set(SENTINEL_IDS)
    assert not any(t in sentinel_set for t in enc)
    assert not any(t in sentinel_set for t in dec)


# -------------------------------------------------------------------
# analyze_midi_structure
# -------------------------------------------------------------------


def test_analyze_midi_structure(tokenizer_and_ids: tuple) -> None:
    tok, ids = tokenizer_and_ids
    structure = analyze_midi_structure(ids, tok.vocab)
    assert "bar_indices" in structure
    assert "beat_indices" in structure
    assert "note_indices" in structure
    assert "timing_indices" in structure
    assert len(structure["bar_indices"]) > 0
    assert len(structure["note_indices"]) > 0


def test_analyze_midi_structure_empty() -> None:
    structure = analyze_midi_structure([], {})
    assert all(len(v) == 0 for v in structure.values())


# -------------------------------------------------------------------
# Determinism
# -------------------------------------------------------------------


def test_corruption_deterministic_with_seed(tokenizer_and_ids: tuple) -> None:
    tok, ids = tokenizer_and_ids
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    a = note_denoising(ids, tok.vocab, SENTINEL_IDS, rng=rng1)
    b = note_denoising(ids, tok.vocab, SENTINEL_IDS, rng=rng2)
    assert a == b


# -------------------------------------------------------------------
# Stress tests
# -------------------------------------------------------------------


def test_4096_token_sequence_sentinel_count() -> None:
    """Test with large sequences to ensure we can handle many sentinels."""
    rng = np.random.default_rng(0)
    ids = list(range(4096))
    mask = random_spans_noise_mask(4096, 0.5, 2.0, rng=rng)
    enc, next_idx = noise_span_to_unique_sentinel(ids, mask, SENTINEL_IDS, start_sentinel_idx=0)
    # Should use a significant number of sentinels
    assert next_idx > 100
    # All sentinel IDs should be valid
    sentinel_set = set(SENTINEL_IDS)
    for t in enc:
        if t in sentinel_set:
            idx = SENTINEL_IDS.index(t)
            assert idx < next_idx


def test_many_chained_calls() -> None:
    """Chain multiple corruption calls and verify sentinel continuity."""
    rng = np.random.default_rng(0)
    global_next = 0
    for _ in range(10):
        ids = list(range(50))
        mask = random_spans_noise_mask(50, 0.2, 2.0, rng=rng)
        _, next_idx = noise_span_to_unique_sentinel(
            ids, mask, SENTINEL_IDS, start_sentinel_idx=global_next,
        )
        assert next_idx >= global_next
        global_next = next_idx
    assert global_next > 10
