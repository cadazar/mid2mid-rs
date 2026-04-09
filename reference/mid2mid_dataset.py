"""
Mid2Mid Dataset classes and tokenizer factory.

Extracted from train_mid2mid.py for modular use in mid2mid-rs integration.
"""

import os
import random
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple

from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset

from miditok import (
    CPWord, PerTok, REMI, MMM, MIDILike, TSD,
    TokenizerConfig, MusicTokenizer, TokSequence,
)
from miditok.utils.utils import (
    get_bars_ticks, get_num_notes_per_bar,
    merge_same_program_tracks, merge_tracks
)
from miditok.utils.split import split_score_per_ticks, split_score_per_beats
from symusic import Score

from decoding_strategies import (
    PerTokStrategy,
    ConstrainedDecodingStrategy,
    TypeGraphStrategy
)
from adaptive_sampling_strategy import PianoBartStyleStrategy
from mid2mid_config import Mid2MidConfig


def create_tokenizers(decoder_type: str = "pertok",
                      restore_from: Optional[str] = None,
                      minimal_decoder: bool = False,
                      pianobart_style: bool = False
                      ) -> Tuple[MusicTokenizer, MusicTokenizer, Optional[ConstrainedDecodingStrategy]]:
    """Return (encoder_tok, decoder_tok, strategy) according to `decoder_type`.
    Supports: pertok, remi, mmm, mmm-remi, mmm-tsd, mmm-midilike, midilike, tsd
    """
    decoder_type = decoder_type.lower()
    tsr = {
        1:  [1],
        2:  [1, 2, 3, 4, 6],
        4:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 26, 96],
        8:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 23, 56, 57],
        16: [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 24, 25, 49],
        32: [1, 3, 5, 6, 7, 17, 25, 26],
    }
    beat_res = {(0,16): 24}
    enc_config = TokenizerConfig(
        beat_res=beat_res,
        beat_res_rest=beat_res,
        use_chords=True,
        use_programs=True,
        num_velocities=32,
        use_time_signatures=True,
        time_signature_range=tsr,
        use_rests=True,
        use_tempos=True,
        use_sustain_pedals=True,
        use_pitch_bends=True,
        one_token_stream_for_programs=False,
        use_pitch_intervals=True,
        ticks_per_quarter=480,
    )
    dec_config = TokenizerConfig(
        beat_res=beat_res,
        beat_res_rest=beat_res,
        use_velocities=not minimal_decoder,
        use_programs=not minimal_decoder,
        use_pitch_intervals=True,
        ticks_per_quarter=480,
        use_time_signatures=True if decoder_type=='pertok' else (not minimal_decoder),
        use_tempos=not minimal_decoder,
        time_signature_range=tsr,
        use_microtiming=not minimal_decoder,
        max_microtiming_shift=0.125,
        num_microtiming_bins=32,
        one_token_stream_for_programs=True,
    )

    if restore_from is not None:
        restore_from = os.path.join(restore_from, "encoder_tokenizer.json")
    encoder_tokenizer = CPWord(enc_config, params=restore_from)
    strategy: Optional[ConstrainedDecodingStrategy] = None

    decoder_type = decoder_type.lower()

    if restore_from is not None:
        restore_from = os.path.join(os.path.dirname(restore_from), "decoder_tokenizer.json")
    if decoder_type == "pertok":
        decoder_tokenizer = PerTok(dec_config, restore_from)
        strategy = PerTokStrategy(decoder_tokenizer)
    elif decoder_type == "remi":
        dec_config.additional_params = {
            'max_bar_embedding': None
        }
        decoder_tokenizer = REMI(dec_config, restore_from)
        strategy = TypeGraphStrategy(decoder_tokenizer)
    elif decoder_type.startswith("mmm"):
        if decoder_type == "mmm":
            base_tok = "REMI"
        else:
            parts = decoder_type.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid MMM format: {decoder_type}. Use 'mmm' or 'mmm-<base>'")
            base_tok = parts[1].upper()
            if base_tok not in ["REMI", "TSD", "MIDILIKE"]:
                raise ValueError(f"Invalid MMM base tokenizer: {base_tok}")
            if base_tok == "MIDILIKE": base_tok = "MIDILike"

        dec_config.additional_params = {
            'base_tokenizer': base_tok
        }
        decoder_tokenizer = MMM(dec_config, restore_from)
        strategy = TypeGraphStrategy(decoder_tokenizer)
    elif decoder_type == "midilike":
        decoder_tokenizer = MIDILike(dec_config, restore_from)
        strategy = TypeGraphStrategy(decoder_tokenizer)
    elif decoder_type == "tsd":
        decoder_tokenizer = TSD(dec_config, restore_from)
        strategy = TypeGraphStrategy(decoder_tokenizer)
    else:
        raise ValueError(f"Unsupported decoder_tokenizer: {decoder_type}")

    if pianobart_style and strategy is not None:
        strategy = PianoBartStyleStrategy(decoder_tokenizer, base_strategy=strategy)

    return encoder_tokenizer, decoder_tokenizer, strategy


class Mid2MidDataset(Dataset):
    def __init__(
        self,
        midi_files: Union[List[str], List[Path], str, Path],
        encoder_tokenizer: MusicTokenizer,
        decoder_tokenizer: MusicTokenizer,
        measures_per_segment: int = 4,
        augment_with_sliding: bool = True,
        sliding_window_stride: int = 2,
        variable_measure_length: bool = False,
        min_measures: int = 8,
        solo_only: bool = False
    ):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.measures_per_segment = measures_per_segment
        self.augment_with_sliding = augment_with_sliding
        self.sliding_window_stride = sliding_window_stride
        self.variable_measure_length = variable_measure_length
        self.min_measures = min_measures
        self.solo_only = solo_only

        self.segments = []
        self.piece_to_slices = {}
        self.idx2piece = {}
        self.skipped_files = []
        self.failed_files = []
        self._seg_idx_count = 0

        midi_files = [midi_files] if isinstance(midi_files, (str, Path)) else midi_files

        for midi_file in tqdm(midi_files, desc=f"Processing {len(midi_files)} MIDI files..."):
            try:
                file_segments = self._create_segments_from_file(midi_file)
                self.segments.extend(file_segments)
            except Exception as e:
                if isinstance(e, TypeError):
                    pass
                else:
                    self.failed_files.append((midi_file, str(e)))

        self._sampling_pools = {piece_id: slice_indices.copy() for piece_id, slice_indices in self.piece_to_slices.items()}
        for pool in self._sampling_pools.values():
            random.shuffle(pool)

    def _create_segments_from_file(self, midi_file: Union[str, Path]) -> List[Dict]:
        """Create segments storing symusic Score objects.

        Note that this merges all tracks designating the same program. Multitrack modeling
        is still possible, given that the input MIDI assigns different programs for each track."""

        score = Score(midi_file)
        merge_same_program_tracks(score.tracks)
        if self.solo_only and len(score.tracks) > 1:
            self.skipped_files.append({
                'file': str(midi_file),
                'reason': f'multi-track ({len(score.tracks)} tracks)',
                'programs': [t.program for t in score.tracks]
            })
            return

        bar_ticks = get_bars_ticks(score, only_notes_onsets=True)
        n_bars = len(bar_ticks) - 1

        if n_bars < self.min_measures:
            raise ValueError(f"Not enough measures: {n_bars} < {self.min_measures}")

        segments = []
        piece_id = hash(str(midi_file))
        self.piece_to_slices[piece_id] = []
        self.idx2piece[len(self.idx2piece)] = piece_id

        if n_bars < self.measures_per_segment:
            segment_score = score.copy()
            segments.append({
                'symusic_score': segment_score,
                'piece_id': piece_id,
                'start_bar': 0,
                'end_bar': n_bars,
                'is_first': True,
                'is_last': True,
                'segment_type': 'fixed',
                'source_file': str(midi_file),
                'num_tracks': len(segment_score.tracks),
                'num_measures': n_bars
            })
            self.piece_to_slices[piece_id].append(self._seg_idx_count)
            self._seg_idx_count += 1
        else:
            for start_bar_idx in range(0, n_bars - self.measures_per_segment + 1, self.measures_per_segment):
                end_bar_idx = start_bar_idx + self.measures_per_segment

                start_tick = bar_ticks[start_bar_idx]
                end_tick = bar_ticks[end_bar_idx] if not end_bar_idx == n_bars else bar_ticks[-1]
                segment_score = score.copy().clip(start_tick, end_tick, clip_end=False).shift_time(-start_tick)
                if segment_score.note_num() == 0:
                    continue

                segments.append({
                    'symusic_score': segment_score,
                    'piece_id': piece_id,
                    'start_bar': start_bar_idx,
                    'end_bar': end_bar_idx,
                    'is_first': start_bar_idx == 0,
                    'is_last': end_bar_idx == n_bars,
                    'segment_type': 'fixed',
                    'source_file': str(midi_file),
                    'num_tracks': len(segment_score.tracks),
                    'num_measures': end_bar_idx - start_bar_idx
                })
                self.piece_to_slices[piece_id].append(self._seg_idx_count)
                self._seg_idx_count += 1

            last_processed_bar = (n_bars // self.measures_per_segment) * self.measures_per_segment
            remaining_bars = n_bars - last_processed_bar
            if remaining_bars >= self.min_measures:
                start_bar_idx = last_processed_bar
                end_bar_idx = n_bars

                start_tick = bar_ticks[start_bar_idx]
                end_tick = bar_ticks[-1]
                segment_score = score.copy().clip(start_tick, end_tick, clip_end=False).shift_time(-start_tick)

                if segment_score.note_num() > 0:
                    segments.append({
                        'symusic_score': segment_score,
                        'piece_id': piece_id,
                        'start_bar': start_bar_idx,
                        'end_bar': end_bar_idx,
                        'is_first': start_bar_idx == 0,
                        'is_last': True,
                        'segment_type': 'fixed',
                        'source_file': str(midi_file),
                        'num_tracks': len(segment_score.tracks),
                        'num_measures': remaining_bars
                    })
                    self.piece_to_slices[piece_id].append(self._seg_idx_count)
                    self._seg_idx_count += 1

        if self.augment_with_sliding and n_bars >= self.min_measures + self.sliding_window_stride:
            stride = self.sliding_window_stride
            window_size = min(self.measures_per_segment, n_bars)
            for start_bar_idx in range(stride, n_bars - self.min_measures + 1, stride):
                if n_bars >= self.measures_per_segment and start_bar_idx % self.measures_per_segment == 0:
                    continue

                end_bar_idx = min(start_bar_idx + window_size, n_bars)

                start_tick = bar_ticks[start_bar_idx]
                end_tick = bar_ticks[end_bar_idx] if end_bar_idx < len(bar_ticks) else bar_ticks[-1]
                segment_score = score.copy().clip(start_tick, end_tick, clip_end=False).shift_time(-start_tick)

                if segment_score.note_num() == 0:
                    continue

                segments.append({
                    'symusic_score': segment_score,
                    'piece_id': piece_id,
                    'start_bar': start_bar_idx,
                    'end_bar': end_bar_idx,
                    'is_first': start_bar_idx == 0,
                    'is_last': end_bar_idx == n_bars,
                    'segment_type': 'sliding',
                    'source_file': str(midi_file),
                    'num_tracks': len(segment_score.tracks),
                    'num_measures': end_bar_idx - start_bar_idx
                })
                self.piece_to_slices[piece_id].append(self._seg_idx_count)
                self._seg_idx_count += 1

        if self.variable_measure_length and n_bars > self.min_measures:
            for _ in range(sum([1 for s in segments if s['segment_type'] == 'fixed'])):
                max_start = n_bars - self.min_measures
                if max_start <= 0:
                    continue

                start_bar_idx = random.randint(0, max_start)
                max_len = min(self.measures_per_segment, n_bars - start_bar_idx)
                segment_length = random.randint(self.min_measures, max_len)
                end_bar_idx = start_bar_idx + segment_length

                start_tick = bar_ticks[start_bar_idx]
                end_tick = bar_ticks[end_bar_idx] if end_bar_idx < n_bars else bar_ticks[-1]
                segment_score = score.copy().clip(start_tick, end_tick, clip_end=False).shift_time(-start_tick)

                if segment_score.note_num() == 0:
                    continue

                segments.append({
                    'symusic_score': segment_score,
                    'piece_id': piece_id,
                    'start_bar': start_bar_idx,
                    'end_bar': end_bar_idx,
                    'is_first': start_bar_idx == 0,
                    'is_last': end_bar_idx == n_bars,
                    'segment_type': 'variable',
                    'source_file': str(midi_file),
                    'num_tracks': len(segment_score.tracks),
                    'num_measures': segment_length
                })
                self.piece_to_slices[piece_id].append(self._seg_idx_count)
                self._seg_idx_count += 1

        return segments

    def __len__(self):
        return len(self.piece_to_slices)

    def __getitem__(self, idx) -> Dict:
        """Returns one segment for the given piece index.
        We draw without replacement from the piece's pool so that every
        segment is eventually seen.  When the pool is empty we refill it with
        a fresh shuffled copy of all of that piece's segments."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"{idx} out of range")
        piece_hash = self.idx2piece[idx]
        pool = self._sampling_pools[piece_hash]
        if len(pool) == 0:
            pool.extend(self.piece_to_slices[piece_hash])
            random.shuffle(pool)
        slice_idx = pool.pop()
        return self.segments[slice_idx]

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get dataset statistics"""
        fixed_segments = sum(1 for s in self.segments if s['segment_type'] == 'fixed')
        sliding_segments = sum(1 for s in self.segments if s['segment_type'] == 'sliding')

        sample_size = min(200, len(self.segments))
        if sample_size > 0:
            enc_lens, dec_lens = [], []
            for seg in tqdm(random.sample(self.segments, sample_size),
                            total=sample_size, desc="Grabbing dataset statistics"):
                enc_tracks = self.encoder_tokenizer(seg['symusic_score'].copy())
                dec_tracks = self.decoder_tokenizer(seg['symusic_score'].copy())
                try:
                    enc_lens.append(len(sum(enc_tracks).ids))
                except:
                    enc_lens.append(len(enc_tracks.ids))
                try:
                    dec_lens.append(len(sum(dec_tracks).ids))
                except:
                    dec_lens.append(len(dec_tracks.ids))
            avg_enc_len = sum(enc_lens) / len(enc_lens)
            avg_dec_len = sum(dec_lens) / len(dec_lens)
        else:
            avg_enc_len = avg_dec_len = 0

        unique_files = len(set(s['piece_id'] for s in self.segments))
        avg_segments_per_piece = (len(self.segments) / unique_files) if unique_files > 0 else 0
        return {
            'total_segments': len(self.segments),
            'fixed_segments': fixed_segments,
            'sliding_segments': sliding_segments,
            'unique_files': unique_files,
            'avg_segments_per_piece': avg_segments_per_piece,
            'failed_files': len(self.failed_files),
            'skipped_files': len(self.skipped_files) if self.solo_only else 0,
            'solo_only': self.solo_only,
            'avg_encoder_length': avg_enc_len,
            'avg_decoder_length': avg_dec_len,
            'measures_per_segment': self.measures_per_segment,
            'sliding_window_stride': self.sliding_window_stride
        }


class SegmentDataset(Dataset):
    """Fixed view over segments for reproducible evaluation / test splits.

    Given an existing `Mid2MidDataset` instance and an index list, this class
    exposes the underlying `segments` directly by index.  Unlike
    `Mid2MidDataset.__getitem__` it is deterministic (no piece-wise random
    sampling), so every epoch sees exactly the same order / content.
    """
    def __init__(self, base_ds: Mid2MidDataset, indices: List[int]):
        self.base_ds = base_ds
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"{idx} out of range")
        seg_idx = self.indices[idx]
        return self.base_ds.segments[seg_idx]
