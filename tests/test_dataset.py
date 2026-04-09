"""Tests for Mid2MidDataset and compute_bar_ticks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from miditok.midi_adapter import (
    AdapterNote,
    AdapterScore,
    AdapterTempo,
    AdapterTimeSignature,
    AdapterTrack,
    NoteList,
    TempoList,
    TimeSignatureList,
)
from miditok.mid2mid_dataset import Mid2MidDataset, compute_bar_ticks

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_MIDI = REPO_ROOT / "violin_partita_bwv-1004_1_(c)grossman.mid"
SAMPLE_MIDI_2 = REPO_ROOT / "violin_partita_bwv-1004_2_(c)grossman.mid"
SAMPLE_MIDI_3 = REPO_ROOT / "violin_partita_bwv-1004_5_(c)grossman.mid"
EMPTY_MIDI = REPO_ROOT / "tests" / "MIDIs_one_track" / "empty.mid"
ONE_TRACK_DIR = REPO_ROOT / "tests" / "MIDIs_one_track"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_score(
    tpq: int = 480,
    time_sig: tuple[int, int] = (4, 4),
    num_notes: int = 32,
    note_dur: int = 120,
    note_spacing: int = 120,
) -> AdapterScore:
    """Create a synthetic score for testing."""
    score = AdapterScore(ticks_per_quarter=tpq)
    ts = AdapterTimeSignature(0, time_sig[0], time_sig[1])
    score.time_signatures = TimeSignatureList([ts])
    score.tempos = TempoList([AdapterTempo(0, 120.0)])
    track = AdapterTrack(program=0, is_drum=False, name="Piano")
    for i in range(num_notes):
        track.notes.append(AdapterNote(i * note_spacing, note_dur, 60 + (i % 12), 80))
    score._tracks = [track]
    return score


# -------------------------------------------------------------------
# compute_bar_ticks
# -------------------------------------------------------------------


def test_bar_ticks_4_4() -> None:
    score = _make_score(tpq=480, time_sig=(4, 4), num_notes=100, note_spacing=120)
    bars = compute_bar_ticks(score)
    assert bars[0] == 0
    # 4/4 at 480 tpq → bar length = 4 * 480 = 1920
    for i in range(1, min(5, len(bars))):
        assert bars[i] - bars[i - 1] == 1920


def test_bar_ticks_3_4() -> None:
    score = _make_score(tpq=480, time_sig=(3, 4), num_notes=100, note_spacing=120)
    bars = compute_bar_ticks(score)
    assert bars[0] == 0
    # 3/4 at 480 tpq → bar length = 3 * 480 = 1440
    for i in range(1, min(5, len(bars))):
        assert bars[i] - bars[i - 1] == 1440


def test_bar_ticks_6_8() -> None:
    score = _make_score(tpq=480, time_sig=(6, 8), num_notes=50, note_spacing=120)
    bars = compute_bar_ticks(score)
    assert bars[0] == 0
    # 6/8 at 480 tpq → bar length = 6 * (4/8) * 480 = 1440
    for i in range(1, min(5, len(bars))):
        assert bars[i] - bars[i - 1] == 1440


def test_bar_ticks_changing_time_signature() -> None:
    score = _make_score(tpq=480, num_notes=100, note_spacing=120)
    score.time_signatures = TimeSignatureList([
        AdapterTimeSignature(0, 4, 4),
        AdapterTimeSignature(3840, 3, 4),  # after 2 bars of 4/4
    ])
    bars = compute_bar_ticks(score)
    assert bars[0] == 0
    # First bar: 1920
    assert bars[1] == 1920
    # After tick 3840 → 3/4 bars of length 1440
    idx_3840 = next(i for i, b in enumerate(bars) if b >= 3840)
    if idx_3840 + 1 < len(bars):
        assert bars[idx_3840 + 1] - bars[idx_3840] == 1440


def test_bar_ticks_no_time_signature() -> None:
    """Default 4/4 when no time signatures present."""
    score = _make_score(tpq=480, num_notes=20, note_spacing=120)
    score.time_signatures = TimeSignatureList()
    bars = compute_bar_ticks(score)
    assert bars[0] == 0
    for i in range(1, min(3, len(bars))):
        assert bars[i] - bars[i - 1] == 1920  # default 4/4


def test_bar_ticks_real_midi() -> None:
    score = AdapterScore.from_file(str(SAMPLE_MIDI))
    bars = compute_bar_ticks(score)
    assert len(bars) >= 2
    assert bars[0] == 0


# -------------------------------------------------------------------
# Mid2MidDataset: fixed segments
# -------------------------------------------------------------------


def test_dataset_fixed_segments() -> None:
    ds = Mid2MidDataset(
        [SAMPLE_MIDI], measures_per_segment=8,
        augment_with_sliding=False, variable_measure_length=False, seed=0,
    )
    assert len(ds) > 0
    for seg in ds.segments:
        assert seg["num_measures"] <= 8
        assert seg["num_measures"] >= ds.min_measures
        assert seg["segment_type"] == "fixed"


def test_dataset_correct_bar_ranges() -> None:
    ds = Mid2MidDataset(
        [SAMPLE_MIDI], measures_per_segment=8,
        augment_with_sliding=False, seed=0,
    )
    for seg in ds.segments:
        assert seg["end_bar"] > seg["start_bar"]
        assert seg["num_measures"] == seg["end_bar"] - seg["start_bar"]


def test_dataset_segments_shifted_to_tick_zero() -> None:
    ds = Mid2MidDataset(
        [SAMPLE_MIDI], measures_per_segment=8,
        augment_with_sliding=False, seed=0,
    )
    for seg in ds.segments:
        score = seg["score"]
        # All note times should be >= 0
        for tr in score.tracks:
            for n in tr.notes:
                assert n.time >= 0


def test_dataset_getitem() -> None:
    ds = Mid2MidDataset([SAMPLE_MIDI], augment_with_sliding=False, seed=0)
    seg = ds[0]
    assert "score" in seg
    assert "piece_id" in seg
    assert "start_bar" in seg
    assert "end_bar" in seg
    assert "segment_type" in seg
    assert "source_file" in seg
    assert "num_measures" in seg


def test_dataset_len() -> None:
    ds = Mid2MidDataset([SAMPLE_MIDI], augment_with_sliding=False, seed=0)
    assert len(ds) > 0


# -------------------------------------------------------------------
# Sliding window
# -------------------------------------------------------------------


def test_sliding_window_produces_more_segments() -> None:
    ds_no_slide = Mid2MidDataset(
        [SAMPLE_MIDI], measures_per_segment=8,
        augment_with_sliding=False, seed=0,
    )
    ds_slide = Mid2MidDataset(
        [SAMPLE_MIDI], measures_per_segment=8,
        augment_with_sliding=True, sliding_window_stride=2, seed=0,
    )
    assert len(ds_slide) >= len(ds_no_slide)


def test_sliding_window_no_fixed_overlap() -> None:
    ds = Mid2MidDataset(
        [SAMPLE_MIDI], measures_per_segment=8,
        augment_with_sliding=True, sliding_window_stride=2, seed=0,
    )
    fixed_starts = {seg["start_bar"] for seg in ds.segments if seg["segment_type"] == "fixed"}
    sliding_starts = {seg["start_bar"] for seg in ds.segments if seg["segment_type"] == "sliding"}
    assert fixed_starts.isdisjoint(sliding_starts)


# -------------------------------------------------------------------
# Variable length
# -------------------------------------------------------------------


def test_variable_length_segments() -> None:
    ds = Mid2MidDataset(
        [SAMPLE_MIDI], measures_per_segment=16,
        augment_with_sliding=False, variable_measure_length=True,
        min_measures=4, seed=42,
    )
    var_segs = [s for s in ds.segments if s["segment_type"] == "variable"]
    assert len(var_segs) > 0
    for seg in var_segs:
        assert seg["num_measures"] >= ds.min_measures
        assert seg["num_measures"] <= ds.measures_per_segment


# -------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------


def test_files_with_few_bars_skipped() -> None:
    ds = Mid2MidDataset(
        [SAMPLE_MIDI], measures_per_segment=8,
        min_measures=99999,  # Skip everything
        augment_with_sliding=False, seed=0,
    )
    assert len(ds) == 0
    assert len(ds.skipped_files) > 0


def test_empty_score_handling() -> None:
    ds = Mid2MidDataset(
        [EMPTY_MIDI], measures_per_segment=8,
        augment_with_sliding=False, seed=0,
    )
    # Should either skip or produce no segments
    assert len(ds) == 0


def test_multiple_files() -> None:
    ds = Mid2MidDataset(
        [SAMPLE_MIDI, SAMPLE_MIDI_2], measures_per_segment=8,
        augment_with_sliding=False, seed=0,
    )
    piece_ids = {seg["piece_id"] for seg in ds.segments}
    assert len(piece_ids) == 2


# -------------------------------------------------------------------
# clip() and shift_time()
# -------------------------------------------------------------------


def test_clip_produces_valid_score() -> None:
    score = AdapterScore.from_file(str(SAMPLE_MIDI))
    bars = compute_bar_ticks(score)
    if len(bars) >= 5:
        clipped = score.clip(bars[1], bars[4])
        assert clipped.note_num() > 0
        for tr in clipped.tracks:
            for n in tr.notes:
                assert n.time >= 0


def test_shift_time_produces_valid_score() -> None:
    score = AdapterScore.from_file(str(SAMPLE_MIDI))
    shifted = score.shift_time(-100)
    for tr in shifted.tracks:
        for n in tr.notes:
            assert n.time >= 0


def test_shift_time_positive() -> None:
    score = _make_score(num_notes=5, note_spacing=100)
    shifted = score.shift_time(500)
    for tr in shifted.tracks:
        for n in tr.notes:
            assert n.time >= 500


def test_shift_time_negative_clamps() -> None:
    score = _make_score(num_notes=5, note_spacing=100)
    shifted = score.shift_time(-10000)
    for tr in shifted.tracks:
        for n in tr.notes:
            assert n.time >= 0


def test_clip_and_shift_roundtrip() -> None:
    score = AdapterScore.from_file(str(SAMPLE_MIDI))
    bars = compute_bar_ticks(score)
    if len(bars) >= 3:
        start = bars[1]
        end = bars[2]
        # clip already shifts to 0
        clipped = score.clip(start, end)
        # shift_time(0) should be a no-op
        shifted = clipped.shift_time(0)
        assert clipped.note_num() == shifted.note_num()
