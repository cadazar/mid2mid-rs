"""T5-style sentinel span corruption for MIDI token sequences.

All functions take and return plain Python lists of integer token IDs and use
NumPy as the only numeric dependency.  Sentinel tokens are assigned in
**increasing** order: first corrupted span → ``sentinel_token_ids[start_sentinel_idx]``,
second → ``sentinel_token_ids[start_sentinel_idx + 1]``, etc.

Every corruption strategy returns a 3-tuple::

    (encoder_ids, decoder_ids, next_sentinel_idx)

so that multiple corruption calls can be chained across segments.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Core span-mask primitives
# ---------------------------------------------------------------------------


def _random_segmentation(
    num_items: int,
    num_segments: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Partition ``num_items`` into ``num_segments`` non-empty integer parts."""
    if num_segments <= 0:
        return np.array([], dtype=np.int64)
    if num_segments == 1:
        return np.array([num_items], dtype=np.int64)
    if num_items < num_segments:
        out = np.ones(num_segments, dtype=np.int64)
        out[-1] += max(0, num_items - num_segments)
        return out
    positions = np.arange(num_items - 1, dtype=np.int64)
    is_boundary = np.zeros(num_items - 1, dtype=bool)
    is_boundary[: num_segments - 1] = True
    rng.shuffle(is_boundary)
    boundary_indices = np.flatnonzero(is_boundary) + 1
    starts = np.concatenate([[0], boundary_indices])
    ends = np.concatenate([boundary_indices, [num_items]])
    return ends - starts


def random_spans_noise_mask(
    length: int,
    noise_density: float,
    mean_noise_span_length: float = 3.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a boolean noise mask following the T5 random span scheme.

    Algorithm:
    1. ``num_noise_tokens = round(length * noise_density)``, clamped to ``[1, length-1]``
    2. ``num_noise_spans = max(1, round(num_noise_tokens / mean_noise_span_length))``
    3. Randomly segment both noise and non-noise tokens into ``num_noise_spans`` parts
    4. Interleave starting with non-noise: ``[nonnoise_0, noise_0, nonnoise_1, noise_1, ...]``
    5. Odd-indexed spans are noise
    """
    if rng is None:
        rng = np.random.default_rng()
    if length <= 1 or noise_density <= 0.0:
        return np.zeros(length, dtype=bool)
    num_noise_tokens = int(round(length * noise_density))
    num_noise_tokens = max(1, min(length - 1, num_noise_tokens))
    num_noise_spans = max(1, int(round(num_noise_tokens / mean_noise_span_length)))
    num_noise_spans = min(num_noise_spans, num_noise_tokens)
    num_nonnoise_tokens = length - num_noise_tokens
    num_noise_spans = min(num_noise_spans, num_nonnoise_tokens)
    num_noise_spans = max(1, num_noise_spans)

    noise_lengths = _random_segmentation(num_noise_tokens, num_noise_spans, rng)
    nonnoise_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans, rng)

    interleaved = np.empty(2 * num_noise_spans, dtype=np.int64)
    interleaved[0::2] = nonnoise_lengths
    interleaved[1::2] = noise_lengths
    mask = np.zeros(length, dtype=bool)
    cursor = 0
    for i, span_len in enumerate(interleaved):
        if i % 2 == 1 and span_len > 0:
            mask[cursor : cursor + span_len] = True
        cursor += span_len
    return mask


def noise_span_to_unique_sentinel(
    token_ids: Sequence[int],
    noise_mask: np.ndarray,
    sentinel_token_ids: list[int],
    start_sentinel_idx: int = 0,
) -> tuple[list[int], int]:
    """Replace each noise span with an increasing sentinel token.

    Returns ``(compressed_ids, next_sentinel_idx)``.
    """
    arr = np.asarray(token_ids, dtype=np.int64)
    mask = np.asarray(noise_mask, dtype=bool)
    if arr.size == 0:
        return [], start_sentinel_idx
    prev_mask = np.concatenate([[False], mask[:-1]])
    span_starts = mask & ~prev_mask
    out: list[int] = []
    sentinel_idx = start_sentinel_idx
    for i in range(arr.size):
        if span_starts[i]:
            out.append(sentinel_token_ids[sentinel_idx])
            sentinel_idx += 1
        elif mask[i]:
            continue
        else:
            out.append(int(arr[i]))
    return out, sentinel_idx


def nonnoise_span_to_unique_sentinel(
    token_ids: Sequence[int],
    noise_mask: np.ndarray,
    sentinel_token_ids: list[int],
    start_sentinel_idx: int = 0,
) -> tuple[list[int], int]:
    """Same as ``noise_span_to_unique_sentinel`` but with the inverted mask.

    Used to build decoder targets.
    """
    inv = ~np.asarray(noise_mask, dtype=bool)
    return noise_span_to_unique_sentinel(token_ids, inv, sentinel_token_ids, start_sentinel_idx)


def apply_span_corruption(
    token_ids: Sequence[int],
    noise_density: float,
    mean_span_length: float,
    sentinel_token_ids: list[int],
    start_sentinel_idx: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], list[int], int]:
    """Convenience: mask + encoder/decoder sentinel replacement.

    Returns ``(encoder_ids, decoder_ids, next_sentinel_idx)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    mask = random_spans_noise_mask(len(token_ids), noise_density, mean_span_length, rng=rng)
    enc, next_idx = noise_span_to_unique_sentinel(token_ids, mask, sentinel_token_ids, start_sentinel_idx)
    dec, _ = nonnoise_span_to_unique_sentinel(token_ids, mask, sentinel_token_ids, start_sentinel_idx)
    return enc, dec, next_idx


# ---------------------------------------------------------------------------
# MIDI structure analysis
# ---------------------------------------------------------------------------


def analyze_midi_structure(
    token_ids: Sequence[int],
    vocab: Mapping[str, int],
) -> dict[str, list[int]]:
    """Return positional indices of bars/beats/notes/timing tokens."""
    inv = {int(v): k for k, v in vocab.items()}
    bars: list[int] = []
    beats: list[int] = []
    notes: list[int] = []
    timing: list[int] = []
    for i, tid in enumerate(token_ids):
        tok = inv.get(int(tid))
        if tok is None:
            continue
        if tok.startswith("Bar"):
            bars.append(i)
        if tok == "Position_0":
            beats.append(i)
        if tok.startswith("Pitch_") or tok.startswith("PitchDrum_"):
            notes.append(i)
        if tok.startswith("Position_") or tok.startswith("TimeShift_"):
            timing.append(i)
    return {
        "bar_indices": bars,
        "beat_indices": beats,
        "note_indices": notes,
        "timing_indices": timing,
    }


# ---------------------------------------------------------------------------
# Music-aware corruption strategies
# ---------------------------------------------------------------------------


def _ensure_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _align_mask_to_boundaries(
    mask: np.ndarray, boundaries: list[int], length: int,
) -> np.ndarray:
    """Snap each contiguous noise span to the nearest boundary positions."""
    if not boundaries:
        return mask
    bset = sorted(set(boundaries))
    barr = np.array(bset, dtype=np.int64)
    out = np.zeros(length, dtype=bool)
    in_span = False
    span_start = 0
    for i in range(length):
        if mask[i] and not in_span:
            in_span = True
            span_start = i
        elif (not mask[i]) and in_span:
            in_span = False
            idx = int(np.searchsorted(barr, span_start))
            anchor = bset[idx] if idx < len(bset) else bset[-1]
            anchor = max(0, min(length - 1, anchor))
            span_len = i - span_start
            out[anchor : min(length, anchor + span_len)] = True
    if in_span:
        idx = int(np.searchsorted(barr, span_start))
        anchor = bset[idx] if idx < len(bset) else bset[-1]
        span_len = length - span_start
        out[anchor : min(length, anchor + span_len)] = True
    return out


def beat_denoising(
    token_ids: Sequence[int],
    vocab: Mapping[str, int],
    sentinel_token_ids: list[int],
    noise_density: float = 0.15,
    mean_noise_span_length: float = 2.0,
    start_sentinel_idx: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], list[int], int]:
    """Mask spans aligned to beat positions."""
    rng = _ensure_rng(rng)
    structure = analyze_midi_structure(token_ids, vocab)
    raw_mask = random_spans_noise_mask(
        len(token_ids), noise_density, mean_noise_span_length, rng=rng,
    )
    aligned = _align_mask_to_boundaries(
        raw_mask, structure["beat_indices"] or structure["timing_indices"],
        len(token_ids),
    )
    enc, next_idx = noise_span_to_unique_sentinel(token_ids, aligned, sentinel_token_ids, start_sentinel_idx)
    dec, _ = nonnoise_span_to_unique_sentinel(token_ids, aligned, sentinel_token_ids, start_sentinel_idx)
    return enc, dec, next_idx


def measure_denoising(
    token_ids: Sequence[int],
    vocab: Mapping[str, int],
    sentinel_token_ids: list[int],
    noise_density: float = 0.30,
    mean_noise_span_length: float = 2.0,
    start_sentinel_idx: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], list[int], int]:
    """Mask spans aligned to bar boundaries."""
    rng = _ensure_rng(rng)
    structure = analyze_midi_structure(token_ids, vocab)
    raw_mask = random_spans_noise_mask(
        len(token_ids), noise_density, mean_noise_span_length, rng=rng,
    )
    aligned = _align_mask_to_boundaries(raw_mask, structure["bar_indices"], len(token_ids))
    enc, next_idx = noise_span_to_unique_sentinel(token_ids, aligned, sentinel_token_ids, start_sentinel_idx)
    dec, _ = nonnoise_span_to_unique_sentinel(token_ids, aligned, sentinel_token_ids, start_sentinel_idx)
    return enc, dec, next_idx


def note_denoising(
    token_ids: Sequence[int],
    vocab: Mapping[str, int],
    sentinel_token_ids: list[int],
    noise_density: float = 0.15,
    mean_noise_span_length: float = 3.0,
    start_sentinel_idx: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], list[int], int]:
    """Standard T5 span corruption on MIDI tokens."""
    rng = _ensure_rng(rng)
    mask = random_spans_noise_mask(
        len(token_ids), noise_density, mean_noise_span_length, rng=rng,
    )
    enc, next_idx = noise_span_to_unique_sentinel(token_ids, mask, sentinel_token_ids, start_sentinel_idx)
    dec, _ = nonnoise_span_to_unique_sentinel(token_ids, mask, sentinel_token_ids, start_sentinel_idx)
    return enc, dec, next_idx


def heavy_denoising(
    token_ids: Sequence[int],
    vocab: Mapping[str, int],
    sentinel_token_ids: list[int],
    noise_density: float = 0.50,
    mean_noise_span_length: float = 8.0,
    start_sentinel_idx: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], list[int], int]:
    """Heavy corruption: 50% density with long spans."""
    rng = _ensure_rng(rng)
    mask = random_spans_noise_mask(
        len(token_ids), noise_density, mean_noise_span_length, rng=rng,
    )
    enc, next_idx = noise_span_to_unique_sentinel(token_ids, mask, sentinel_token_ids, start_sentinel_idx)
    dec, _ = nonnoise_span_to_unique_sentinel(token_ids, mask, sentinel_token_ids, start_sentinel_idx)
    return enc, dec, next_idx


def attribute_denoising(
    token_ids: Sequence[int],
    vocab: Mapping[str, int],
    sentinel_token_ids: list[int],
    corruption_ratio: float = 0.20,
    start_sentinel_idx: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], list[int], int]:
    """Replace velocity/duration/pitch tokens with random alternates of the same kind.

    No sentinels are used — returns corrupted as encoder, original as decoder,
    and ``next_sentinel_idx`` unchanged.
    """
    rng = _ensure_rng(rng)
    inv = {int(v): k for k, v in vocab.items()}
    by_prefix: dict[str, list[int]] = {}
    for tok, tid in vocab.items():
        if "_" in tok:
            prefix = tok.split("_", 1)[0]
            if prefix in ("Velocity", "Duration", "Pitch"):
                by_prefix.setdefault(prefix, []).append(int(tid))
    decoder_ids = list(int(t) for t in token_ids)
    encoder_ids = list(decoder_ids)
    candidates = [
        i for i, t in enumerate(encoder_ids)
        if (inv.get(t, "").split("_", 1)[0] in by_prefix)
    ]
    n_corrupt = int(round(len(candidates) * corruption_ratio))
    if n_corrupt > 0 and candidates:
        chosen = rng.choice(len(candidates), size=min(n_corrupt, len(candidates)), replace=False)
        for c in chosen:
            i = candidates[int(c)]
            prefix = inv[encoder_ids[i]].split("_", 1)[0]
            options = by_prefix[prefix]
            new_id = int(rng.choice(options))
            encoder_ids[i] = new_id
    return encoder_ids, decoder_ids, start_sentinel_idx


def continuation(
    token_ids: Sequence[int],
    vocab: Mapping[str, int],
    sentinel_token_ids: list[int] | None = None,
    prefix_ratio_range: tuple[float, float] = (0.2, 0.5),
    start_sentinel_idx: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], list[int], int]:
    """Split into prefix/continuation.

    Randomly chooses split granularity: bar boundary, beat boundary, or
    arbitrary token position.  No sentinels are used.

    Returns ``(prefix_ids, continuation_ids, next_sentinel_idx_unchanged)``.
    The returned dict-compatible metadata about the split (prefix_measures, etc.)
    is computed by the caller from the structure analysis.
    """
    rng = _ensure_rng(rng)
    structure = analyze_midi_structure(token_ids, vocab)
    bars = structure["bar_indices"]
    beats = structure["beat_indices"] or structure["timing_indices"]
    n = len(token_ids)
    if n < 4:
        cut = max(1, n // 2)
        return list(token_ids[:cut]), list(token_ids[cut:]), start_sentinel_idx

    lo = max(1, int(round(n * prefix_ratio_range[0])))
    hi = max(lo + 1, int(round(n * prefix_ratio_range[1])))

    # Choose split granularity: 0=bar, 1=beat, 2=arbitrary
    granularity = int(rng.integers(0, 3))

    if granularity == 0 and bars:
        eligible = [b for b in bars if lo <= b <= hi]
        if eligible:
            cut = int(rng.choice(eligible))
            return list(token_ids[:cut]), list(token_ids[cut:]), start_sentinel_idx
    if granularity <= 1 and beats:
        eligible = [b for b in beats if lo <= b <= hi]
        if eligible:
            cut = int(rng.choice(eligible))
            return list(token_ids[:cut]), list(token_ids[cut:]), start_sentinel_idx

    # Fallback: bar boundary if available, else arbitrary
    if bars:
        eligible = [b for b in bars if lo <= b <= hi]
        if eligible:
            cut = int(rng.choice(eligible))
            return list(token_ids[:cut]), list(token_ids[cut:]), start_sentinel_idx

    cut = int(rng.integers(lo, min(hi, n)))
    return list(token_ids[:cut]), list(token_ids[cut:]), start_sentinel_idx
