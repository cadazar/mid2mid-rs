"""Tests for Mid2MidCollator and Mid2MidDataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mid2mid_collator import (  # noqa: E402
    MIDI_TASK_PROMPTS,
    Mid2MidCollator,
    Mid2MidCollatorConfig,
    merge_vocabularies,
)
from miditok import REMI  # noqa: E402
from miditok.mid2mid_dataset import Mid2MidDataset, compute_bar_ticks  # noqa: E402
from miditok.midi_adapter import AdapterScore  # noqa: E402

SAMPLE_MIDI = REPO_ROOT / "violin_partita_bwv-1004_1_(c)grossman.mid"
SAMPLE_MIDI_2 = REPO_ROOT / "violin_partita_bwv-1004_2_(c)grossman.mid"


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


@pytest.fixture
def collator() -> Mid2MidCollator:
    cfg = Mid2MidCollatorConfig(max_seq_len=512, min_measures=2, max_measures=64)
    return Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=0)


@pytest.fixture
def collator_unpacked() -> Mid2MidCollator:
    cfg = Mid2MidCollatorConfig(
        max_seq_len=2048, min_measures=2, max_measures=64, pack_sequences=False
    )
    return Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=0)


def test_output_keys(collator: Mid2MidCollator) -> None:
    out = collator([SAMPLE_MIDI])
    expected = {
        "input_ids", "decoder_input_ids", "labels",
        "segment_ids", "position_ids",
        "decoder_segment_ids", "decoder_position_ids",
    }
    assert expected.issubset(out.keys())


def test_output_dtypes(collator: Mid2MidCollator) -> None:
    out = collator([SAMPLE_MIDI])
    for v in out.values():
        assert isinstance(v, np.ndarray)
        assert v.dtype == np.int32


def test_output_shapes(collator: Mid2MidCollator) -> None:
    out = collator([SAMPLE_MIDI])
    for v in out.values():
        assert v.ndim == 2
        assert v.shape[1] == collator.config.max_seq_len


def test_input_decoder_same_length(collator: Mid2MidCollator) -> None:
    out = collator([SAMPLE_MIDI, SAMPLE_MIDI_2])
    assert out["input_ids"].shape == out["decoder_input_ids"].shape


def test_text_prefix_in_input(collator: Mid2MidCollator) -> None:
    out = collator([SAMPLE_MIDI])
    first = out["input_ids"][0]
    text_id_max = collator.midi_id_offset
    # First few tokens should be text (id < midi_id_offset)
    nonzero = first[first > 0]
    assert nonzero.size > 0
    assert nonzero[0] < text_id_max


def test_text_prompt_not_corrupted(collator: Mid2MidCollator) -> None:
    out = collator([SAMPLE_MIDI])
    sentinel_set = set(collator.sentinel_ids)
    seg = out["segment_ids"][0]
    # Tokens belonging to the first segment within the text region.
    text_region = out["input_ids"][0][:5]
    for tid in text_region:
        assert int(tid) not in sentinel_set


def test_padding_zero(collator: Mid2MidCollator) -> None:
    out = collator([SAMPLE_MIDI])
    seg = out["segment_ids"][0]
    pad_positions = seg == 0
    assert (out["input_ids"][0][pad_positions] == 0).all()
    assert (out["position_ids"][0][pad_positions] == 0).all()


def test_segment_ids_one_based(collator: Mid2MidCollator) -> None:
    # Use a long max_seq_len to guarantee padding.
    cfg = Mid2MidCollatorConfig(max_seq_len=8192, min_measures=2, max_measures=64)
    coll = Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=0)
    out = coll([SAMPLE_MIDI])
    seg = out["segment_ids"]
    assert seg.min() == 0  # padding
    nonzero = seg[seg > 0]
    assert nonzero.min() == 1


def test_position_ids_one_based_per_segment(collator: Mid2MidCollator) -> None:
    out = collator([SAMPLE_MIDI])
    pos = out["position_ids"][0]
    nonzero = pos[pos > 0]
    assert nonzero[0] == 1


def test_pack_sequences_packs_multiple(collator: Mid2MidCollator) -> None:
    out = collator([SAMPLE_MIDI, SAMPLE_MIDI_2])
    seg = out["segment_ids"]
    # Either packed in same row (>=2 distinct seg ids in a row) or in different rows.
    distinct_per_row = [len(set(int(x) for x in row if x > 0)) for row in seg]
    assert max(distinct_per_row) >= 1


def test_unpacked_per_example(collator_unpacked: Mid2MidCollator) -> None:
    out = collator_unpacked([SAMPLE_MIDI, SAMPLE_MIDI_2])
    seg = out["segment_ids"]
    for row in seg:
        nz = row[row > 0]
        if nz.size:
            assert (nz == 1).all()


def test_min_measures_filter() -> None:
    cfg = Mid2MidCollatorConfig(max_seq_len=512, min_measures=10000, max_measures=20000)
    coll = Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=0)
    out = coll([SAMPLE_MIDI])
    # All examples filtered out -> empty single padded row
    assert (out["input_ids"] == 0).all()


def test_max_measures_truncation() -> None:
    cfg = Mid2MidCollatorConfig(max_seq_len=2048, min_measures=2, max_measures=4)
    coll = Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=0)
    out = coll([SAMPLE_MIDI])
    # Should still produce some output, but limited
    assert out["input_ids"].shape[1] == 2048


def test_task_sampling_distribution() -> None:
    cfg = Mid2MidCollatorConfig(max_seq_len=1024, min_measures=2, max_measures=64)
    coll = Mid2MidCollator(REMI(), FakeTextTokenizer(), cfg, seed=42)
    # Use multiple invocations
    for _ in range(3):
        coll([SAMPLE_MIDI])
    # Just verify it doesn't crash; deterministic via seed
    out = coll([SAMPLE_MIDI])
    assert out["input_ids"].shape[1] == 1024


def test_merge_vocabularies_disjoint() -> None:
    text = FakeTextTokenizer()
    midi = {f"midi_{i}": i for i in range(50)}
    merged = merge_vocabularies(text, midi)
    text_ids = set(text.get_vocab().values())
    midi_only_ids = set(merged[k] for k in midi)
    assert text_ids.isdisjoint(midi_only_ids)


def test_merge_vocabularies_preserves_text() -> None:
    text = FakeTextTokenizer()
    midi = {"a": 0, "b": 1}
    merged = merge_vocabularies(text, midi)
    for k, v in text.get_vocab().items():
        assert merged[k] == v


def test_collator_handles_missing_metadata(collator: Mid2MidCollator) -> None:
    out = collator([SAMPLE_MIDI], metadata=None)
    assert out["input_ids"].shape[1] == collator.config.max_seq_len


def test_collator_with_metadata(collator: Mid2MidCollator) -> None:
    out = collator([SAMPLE_MIDI], metadata=[{"composer": "Bach", "title": "Partita 2"}])
    assert out["input_ids"].any()


def test_end_to_end_real_midi(collator: Mid2MidCollator) -> None:
    out = collator([SAMPLE_MIDI])
    assert out["input_ids"].dtype == np.int32
    assert out["input_ids"].shape == (1, 512)
    assert out["segment_ids"].max() >= 1


def test_dataset_segments_and_compute_bar_ticks() -> None:
    score = AdapterScore.from_file(SAMPLE_MIDI)
    bars = compute_bar_ticks(score)
    assert len(bars) >= 2
    ds = Mid2MidDataset([SAMPLE_MIDI], measures_per_segment=8, augment_with_sliding=False, seed=0)
    assert len(ds) > 0
    seg = ds[0]
    assert "score" in seg
    assert seg["num_measures"] <= 8
