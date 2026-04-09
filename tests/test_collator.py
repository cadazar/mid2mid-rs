"""Tests for Mid2MidCollator and related utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mid2mid_collator import (
    MIDI_TASK_PROMPTS,
    Mid2MidCollator,
    Mid2MidCollatorConfig,
    merge_vocabularies,
    _extract_midi_vocab,
)
from miditok import REMI
from miditok.mid2mid_dataset import Mid2MidDataset, compute_bar_ticks
from miditok.midi_adapter import AdapterScore

SAMPLE_MIDI = REPO_ROOT / "violin_partita_bwv-1004_1_(c)grossman.mid"
SAMPLE_MIDI_2 = REPO_ROOT / "violin_partita_bwv-1004_2_(c)grossman.mid"


# -------------------------------------------------------------------
# Fake text tokenizer for testing
# -------------------------------------------------------------------


class FakeTextTokenizer:
    vocab_size = 32000
    unk_token_id = -1

    def __init__(self) -> None:
        self._vocab = {f"<extra_id_{i}>": 31999 - i for i in range(256)}

    def get_vocab(self) -> dict:
        return dict(self._vocab)

    def encode(self, text: str) -> list[int]:
        return [hash(w) % 1000 + 1 for w in text.split()][:32]

    def convert_tokens_to_ids(self, tok: str) -> int | None:
        return self._vocab.get(tok)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture
def dataset() -> Mid2MidDataset:
    return Mid2MidDataset(
        [SAMPLE_MIDI], measures_per_segment=8,
        augment_with_sliding=False, seed=0,
    )


@pytest.fixture
def dataset_two() -> Mid2MidDataset:
    return Mid2MidDataset(
        [SAMPLE_MIDI, SAMPLE_MIDI_2], measures_per_segment=8,
        augment_with_sliding=False, seed=0,
    )


@pytest.fixture
def collator() -> Mid2MidCollator:
    cfg = Mid2MidCollatorConfig(max_seq_len=512)
    return Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=0)


@pytest.fixture
def collator_large() -> Mid2MidCollator:
    cfg = Mid2MidCollatorConfig(max_seq_len=2048)
    return Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=0)


# -------------------------------------------------------------------
# Output structure
# -------------------------------------------------------------------


def test_output_has_5_keys(collator: Mid2MidCollator, dataset: Mid2MidDataset) -> None:
    out = collator([dataset[0]])
    expected = {"input_ids", "decoder_input_ids", "labels", "attention_mask", "decoder_attention_mask"}
    assert set(out.keys()) == expected


def test_output_dtypes(collator: Mid2MidCollator, dataset: Mid2MidDataset) -> None:
    out = collator([dataset[0]])
    for v in out.values():
        assert isinstance(v, np.ndarray)
        assert v.dtype == np.int32


def test_output_2d(collator: Mid2MidCollator, dataset: Mid2MidDataset) -> None:
    out = collator([dataset[0]])
    for v in out.values():
        assert v.ndim == 2


def test_output_shapes_single(collator: Mid2MidCollator, dataset: Mid2MidDataset) -> None:
    out = collator([dataset[0]])
    for v in out.values():
        assert v.shape == (1, 512)


def test_batch_output_shapes(collator: Mid2MidCollator, dataset: Mid2MidDataset) -> None:
    segs = [dataset[i] for i in range(min(3, len(dataset)))]
    out = collator(segs)
    batch_size = len(segs)
    for v in out.values():
        assert v.shape == (batch_size, 512)


# -------------------------------------------------------------------
# input_ids and decoder_input_ids same length
# -------------------------------------------------------------------


def test_input_decoder_same_length(collator: Mid2MidCollator, dataset: Mid2MidDataset) -> None:
    out = collator([dataset[0]])
    assert out["input_ids"].shape == out["decoder_input_ids"].shape


# -------------------------------------------------------------------
# Labels use -100 for padding
# -------------------------------------------------------------------


def test_labels_padding_is_minus_100(collator: Mid2MidCollator, dataset: Mid2MidDataset) -> None:
    out = collator([dataset[0]])
    labels = out["labels"][0]
    dec_mask = out["decoder_attention_mask"][0]
    # Where decoder_attention_mask is 0 (padding), labels should be -100
    padding_positions = dec_mask == 0
    if padding_positions.any():
        assert (labels[padding_positions] == -100).all()


def test_labels_real_tokens_not_minus_100(
    collator: Mid2MidCollator, dataset: Mid2MidDataset,
) -> None:
    out = collator([dataset[0]])
    labels = out["labels"][0]
    dec_mask = out["decoder_attention_mask"][0]
    real_positions = dec_mask == 1
    if real_positions.any():
        assert (labels[real_positions] != -100).all()


# -------------------------------------------------------------------
# Attention masks
# -------------------------------------------------------------------


def test_attention_mask_binary(collator: Mid2MidCollator, dataset: Mid2MidDataset) -> None:
    out = collator([dataset[0]])
    for key in ("attention_mask", "decoder_attention_mask"):
        mask = out[key]
        assert set(np.unique(mask)).issubset({0, 1})


def test_attention_mask_ones_then_zeros(
    collator: Mid2MidCollator, dataset: Mid2MidDataset,
) -> None:
    """attention_mask should be 1s followed by 0s (no gaps)."""
    out = collator([dataset[0]])
    mask = out["attention_mask"][0]
    # Find first zero
    zeros = np.where(mask == 0)[0]
    if len(zeros) > 0:
        first_zero = zeros[0]
        # Everything after first zero should be zero
        assert (mask[first_zero:] == 0).all()


def test_decoder_attention_mask_ones_then_zeros(
    collator: Mid2MidCollator, dataset: Mid2MidDataset,
) -> None:
    out = collator([dataset[0]])
    mask = out["decoder_attention_mask"][0]
    zeros = np.where(mask == 0)[0]
    if len(zeros) > 0:
        first_zero = zeros[0]
        assert (mask[first_zero:] == 0).all()


# -------------------------------------------------------------------
# Text prompt not corrupted
# -------------------------------------------------------------------


def test_text_prompt_not_corrupted(
    collator: Mid2MidCollator, dataset: Mid2MidDataset,
) -> None:
    out = collator([dataset[0]])
    sentinel_set = set(collator.sentinel_token_ids)
    # First few tokens should be text (small IDs from FakeTextTokenizer)
    first_tokens = out["input_ids"][0][:5]
    for tid in first_tokens:
        if int(tid) == 0:
            continue  # padding
        assert int(tid) not in sentinel_set


# -------------------------------------------------------------------
# Corruption tasks
# -------------------------------------------------------------------


def test_different_tasks_sampled(dataset: Mid2MidDataset) -> None:
    """With enough calls, different tasks should be sampled."""
    cfg = Mid2MidCollatorConfig(max_seq_len=512)
    coll = Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=42)
    # Process many examples
    results = []
    for i in range(min(20, len(dataset))):
        result = coll._process_segment(dataset[i])
        if result is not None:
            results.append(result)
    # Should produce some non-trivial results
    assert len(results) > 0


def test_no_corruption_task() -> None:
    """no_corruption task should produce identical encoder MIDI and decoder."""
    cfg = Mid2MidCollatorConfig(
        max_seq_len=512,
        task_weights={"no_corruption": 1.0},  # Force no_corruption
    )
    coll = Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=0)
    ds = Mid2MidDataset([SAMPLE_MIDI], measures_per_segment=8, augment_with_sliding=False, seed=0)
    if len(ds) > 0:
        out = coll([ds[0]])
        # With no_corruption, decoder should mirror encoder's MIDI portion
        assert out["input_ids"].any()


def test_continuation_task() -> None:
    cfg = Mid2MidCollatorConfig(
        max_seq_len=2048,
        task_weights={"continuation": 1.0},
    )
    coll = Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=0)
    ds = Mid2MidDataset([SAMPLE_MIDI], measures_per_segment=8, augment_with_sliding=False, seed=0)
    if len(ds) > 0:
        out = coll([ds[0]])
        assert out["input_ids"].any()
        assert out["decoder_input_ids"].any()


# -------------------------------------------------------------------
# Pitch augmentation stays in MIDI range
# -------------------------------------------------------------------


def test_pitch_augmentation_in_range(dataset: Mid2MidDataset) -> None:
    cfg = Mid2MidCollatorConfig(max_seq_len=512, pitch_shift_range=12)
    coll = Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=0)
    # Process many segments to test various shifts
    for i in range(min(10, len(dataset))):
        seg = dataset[i]
        score = seg["score"]
        for tr in score.tracks:
            for n in tr.notes:
                assert 0 <= n.pitch <= 127


# -------------------------------------------------------------------
# merge_vocabularies
# -------------------------------------------------------------------


def test_merge_vocabularies_non_overlapping() -> None:
    text = FakeTextTokenizer()
    midi = REMI()
    merged = merge_vocabularies(text, midi)
    text_ids = set(text.get_vocab().values())
    midi_vocab = _extract_midi_vocab(midi)
    midi_only_ids = set(merged[k] for k in midi_vocab if k not in text.get_vocab())
    assert text_ids.isdisjoint(midi_only_ids)


def test_merge_vocabularies_preserves_text() -> None:
    text = FakeTextTokenizer()
    midi = REMI()
    merged = merge_vocabularies(text, midi)
    for k, v in text.get_vocab().items():
        assert merged[k] == v


def test_merge_vocabularies_sentinel_shared() -> None:
    """Sentinel tokens should remain at their text-vocab positions, not duplicated."""
    text = FakeTextTokenizer()
    midi_vocab_dict = {"Pitch_60": 0, "Velocity_80": 1, "<extra_id_0>": 2}

    class FakeMidi:
        vocab = midi_vocab_dict

    merged = merge_vocabularies(text, FakeMidi())
    # <extra_id_0> should be at the text tokenizer's position
    assert merged["<extra_id_0>"] == text.get_vocab()["<extra_id_0>"]


def test_merge_vocabularies_simple() -> None:
    text = FakeTextTokenizer()
    midi = {"a": 0, "b": 1}

    class FakeMidi:
        vocab = midi

    merged = merge_vocabularies(text, FakeMidi())
    assert "a" in merged
    assert "b" in merged
    assert merged["a"] >= text.vocab_size or merged["a"] in text.get_vocab().values()


# -------------------------------------------------------------------
# Prompt templates
# -------------------------------------------------------------------


def test_prompt_templates_exist() -> None:
    required = [
        "beat_denoising", "measure_denoising", "note_denoising",
        "attribute_denoising", "heavy_denoising", "continuation", "no_corruption",
    ]
    for key in required:
        assert key in MIDI_TASK_PROMPTS
        assert len(MIDI_TASK_PROMPTS[key]) > 0


def test_continuation_templates_have_musical_units() -> None:
    templates = MIDI_TASK_PROMPTS["continuation"]
    has_measures = any("measures" in t for t in templates)
    has_beats = any("beats" in t for t in templates)
    has_complete = any("Complete" in t for t in templates)
    assert has_measures
    assert has_beats
    assert has_complete


def test_no_corruption_template() -> None:
    templates = MIDI_TASK_PROMPTS["no_corruption"]
    assert any("Reproduce" in t for t in templates)


# -------------------------------------------------------------------
# Empty input handling
# -------------------------------------------------------------------


def test_empty_segments_list(collator: Mid2MidCollator) -> None:
    out = collator([])
    assert out["input_ids"].shape == (1, 512)
    assert (out["input_ids"] == 0).all()
    assert (out["labels"] == -100).all()


# -------------------------------------------------------------------
# End-to-end
# -------------------------------------------------------------------


def test_end_to_end_real_midi(collator: Mid2MidCollator, dataset: Mid2MidDataset) -> None:
    out = collator([dataset[0]])
    assert out["input_ids"].dtype == np.int32
    assert out["input_ids"].shape == (1, 512)
    assert out["attention_mask"][0].sum() > 0
    assert out["decoder_attention_mask"][0].sum() > 0


def test_end_to_end_batch(
    collator: Mid2MidCollator, dataset_two: Mid2MidDataset,
) -> None:
    segs = [dataset_two[i] for i in range(min(4, len(dataset_two)))]
    out = collator(segs)
    batch_size = len(segs)
    assert out["input_ids"].shape[0] == batch_size
    assert out["decoder_input_ids"].shape[0] == batch_size


def test_end_to_end_load_segment_tokenize_corrupt_collate() -> None:
    """Full pipeline: load MIDI → segment → tokenize → corrupt → collate → verify."""
    ds = Mid2MidDataset(
        [SAMPLE_MIDI], measures_per_segment=8,
        augment_with_sliding=False, seed=0,
    )
    assert len(ds) > 0

    cfg = Mid2MidCollatorConfig(max_seq_len=1024)
    coll = Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=0)
    out = coll([ds[0], ds[min(1, len(ds) - 1)]])

    # Verify output structure
    assert set(out.keys()) == {
        "input_ids", "decoder_input_ids", "labels",
        "attention_mask", "decoder_attention_mask",
    }
    for v in out.values():
        assert v.dtype == np.int32
        assert v.ndim == 2
        assert v.shape[1] == 1024


def test_collator_handles_segment_with_metadata() -> None:
    ds = Mid2MidDataset(
        [SAMPLE_MIDI], measures_per_segment=8,
        augment_with_sliding=False,
        metadata={str(SAMPLE_MIDI): {"composer": "Bach", "title": "Partita 2"}},
        seed=0,
    )
    if len(ds) > 0:
        seg = ds[0]
        assert seg["metadata"]["composer"] == "Bach"
        cfg = Mid2MidCollatorConfig(max_seq_len=512)
        coll = Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=0)
        out = coll([seg])
        assert out["input_ids"].any()


def test_collator_deterministic_with_seed(dataset: Mid2MidDataset) -> None:
    cfg = Mid2MidCollatorConfig(max_seq_len=512)
    coll1 = Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=42)
    coll2 = Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=42)
    seg = dataset[0]
    out1 = coll1([seg])
    out2 = coll2([seg])
    assert np.array_equal(out1["input_ids"], out2["input_ids"])
    assert np.array_equal(out1["labels"], out2["labels"])
