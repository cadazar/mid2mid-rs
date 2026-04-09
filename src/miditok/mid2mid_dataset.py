"""Mid2MidDataset: pre-segments MIDI files into measure-aligned chunks."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .midi_adapter import AdapterScore


def compute_bar_ticks(score: AdapterScore) -> list[int]:
    """Return bar boundary tick positions for ``score``.

    The boundaries are computed by walking the score's time signatures, taking
    each one's duration in ticks (``num * (4/denom) * ticks_per_quarter``) and
    laying out bars from one time-signature change to the next.  The list ends
    one bar past ``score.end()``.
    """
    tpq = score.ticks_per_quarter
    end_tick = score.end()
    tss = list(score.time_signatures)
    if not tss:
        # Default 4/4 throughout
        bar_len = 4 * tpq
        bars = [0]
        while bars[-1] < end_tick:
            bars.append(bars[-1] + bar_len)
        return bars
    tss = sorted(tss, key=lambda t: t.time)
    bars: list[int] = []
    for i, ts in enumerate(tss):
        ts_start = ts.time
        bar_len = int(round(ts.numerator * (4 / ts.denominator) * tpq))
        if bar_len <= 0:
            continue
        next_start = tss[i + 1].time if i + 1 < len(tss) else max(end_tick + bar_len, ts_start + bar_len)
        if not bars:
            cur = 0
            while cur < ts_start:
                bars.append(cur)
                cur += bar_len
        cur = ts_start if not bars else bars[-1]
        if cur < ts_start:
            cur = ts_start
        while cur < next_start:
            if not bars or bars[-1] != cur:
                bars.append(cur)
            cur += bar_len
    if not bars:
        bars = [0]
    while bars[-1] < end_tick:
        last_bar_len = bars[-1] - bars[-2] if len(bars) >= 2 else 4 * tpq
        bars.append(bars[-1] + max(1, last_bar_len))
    return bars


class Mid2MidDataset:
    """Pre-segment MIDI files into measure-aligned chunks."""

    def __init__(
        self,
        midi_files: list[str | Path],
        measures_per_segment: int = 16,
        min_measures: int = 4,
        augment_with_sliding: bool = True,
        sliding_window_stride: int = 2,
        variable_measure_length: bool = False,
        metadata: dict[str, dict] | None = None,
        seed: int | None = None,
    ) -> None:
        self.measures_per_segment = measures_per_segment
        self.min_measures = min_measures
        self.augment_with_sliding = augment_with_sliding
        self.sliding_window_stride = sliding_window_stride
        self.variable_measure_length = variable_measure_length
        self.metadata = metadata or {}
        self.skipped_files: list[str] = []
        self.failed_files: list[tuple[str, str]] = []
        self.segments: list[dict[str, Any]] = []
        self._rng = random.Random(seed)
        for piece_id, path in enumerate(midi_files):
            self._process_file(piece_id, Path(path))

    def _process_file(self, piece_id: int, path: Path) -> None:
        try:
            score = AdapterScore.from_file(str(path))
        except Exception as exc:  # noqa: BLE001
            self.failed_files.append((str(path), str(exc)))
            return
        bar_ticks = compute_bar_ticks(score)
        n_bars = max(0, len(bar_ticks) - 1)
        if n_bars < self.min_measures:
            self.skipped_files.append(str(path))
            return
        meta = self.metadata.get(str(path), {})

        boundary_starts: set[int] = set()

        # Fixed segments
        start = 0
        while start < n_bars:
            end = min(start + self.measures_per_segment, n_bars)
            if end - start < self.min_measures:
                break
            seg = self._build_segment(score, bar_ticks, piece_id, start, end, "fixed", path, meta)
            if seg is not None:
                self.segments.append(seg)
                boundary_starts.add(start)
            start = end
        # Tail handling
        if start < n_bars and (n_bars - start) >= self.min_measures:
            seg = self._build_segment(score, bar_ticks, piece_id, start, n_bars, "fixed", path, meta)
            if seg is not None:
                self.segments.append(seg)
                boundary_starts.add(start)

        # Sliding window augmentation
        if self.augment_with_sliding and n_bars > self.measures_per_segment:
            offset = self.sliding_window_stride
            while offset + self.min_measures <= n_bars:
                if offset in boundary_starts:
                    offset += self.sliding_window_stride
                    continue
                end = min(offset + self.measures_per_segment, n_bars)
                if end - offset < self.min_measures:
                    break
                seg = self._build_segment(score, bar_ticks, piece_id, offset, end, "sliding", path, meta)
                if seg is not None:
                    self.segments.append(seg)
                offset += self.sliding_window_stride

        # Variable measure length augmentation
        if self.variable_measure_length:
            n_extra = max(1, n_bars // self.measures_per_segment)
            for _ in range(n_extra):
                if n_bars <= self.min_measures:
                    break
                length = self._rng.randint(self.min_measures, max(self.min_measures, self.measures_per_segment))
                start_b = self._rng.randint(0, max(0, n_bars - length))
                end_b = min(start_b + length, n_bars)
                seg = self._build_segment(score, bar_ticks, piece_id, start_b, end_b, "variable", path, meta)
                if seg is not None:
                    self.segments.append(seg)

    def _build_segment(
        self,
        score: AdapterScore,
        bar_ticks: list[int],
        piece_id: int,
        start_bar: int,
        end_bar: int,
        segment_type: str,
        path: Path,
        meta: dict,
    ) -> dict[str, Any] | None:
        if end_bar <= start_bar:
            return None
        start_tick = bar_ticks[start_bar]
        end_tick = bar_ticks[end_bar] if end_bar < len(bar_ticks) else score.end() + 1
        clipped = score.clip(start_tick, end_tick)
        if clipped.note_num() == 0:
            return None
        return {
            "score": clipped,
            "piece_id": piece_id,
            "start_bar": start_bar,
            "end_bar": end_bar,
            "segment_type": segment_type,
            "source_file": str(path),
            "num_measures": end_bar - start_bar,
            "metadata": dict(meta),
        }

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.segments[idx]
