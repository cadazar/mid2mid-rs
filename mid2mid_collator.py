"""Stub interface for the Mid2Mid collator.

This module defines the expected API for a multimodal text+MIDI collator
that produces encoder/decoder pairs for sequence-to-sequence pre-training.

The collator should support:
- Instruction-conditioned corruption with natural language task prompts
  that describe the specific type of corruption applied (beat-level,
  measure-level, note-level, track-level, attribute-level), teaching the
  model musical concepts through language
- Sentinel-based span corruption (T5-style, not BERT <mask>)
- Mixed text+MIDI sequences where the encoder sees a natural language
  prompt describing the corruption and score metadata, followed by
  corrupted MIDI tokens, and the decoder sees the original MIDI sequence
- NumPy array output (for JAX/Flax compatibility, no PyTorch dependency)
- Vocabulary merging with a pre-trained text tokenizer so that text and
  MIDI tokens share a single unified vocabulary
- Reuse of existing sentinel tokens from the text tokenizer (e.g.,
  <extra_id_0>..<extra_id_255>) so that the model builds a unified
  understanding of span corruption across modalities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from miditok import corruption as _corruption
from miditok.midi_adapter import AdapterScore


# corruption task types and their corresponding prompt templates.
# each corruption type maps to a list of prompt variants that describe
# what was done to the sequence, so the model learns musical concepts
# through natural language conditioning from the start of training.
MIDI_TASK_PROMPTS = {
    'beat_denoising': [
        "Several beats have been removed from this MIDI sequence. "
        "The original piece is {metadata}. Restore the missing beats:",
        "Beats have been dropped from the following MIDI excerpt. "
        "This is {metadata}. Reconstruct them:",
    ],
    'measure_denoising': [
        "Multiple measures have been removed from this MIDI sequence. "
        "The original piece is {metadata}. Restore them:",
        "The following MIDI excerpt of {metadata} has had entire measures "
        "deleted. Reconstruct the missing measures:",
    ],
    'note_denoising': [
        "Individual notes have been randomly removed from this MIDI "
        "sequence. The original piece is {metadata}. Restore them:",
        "Some notes are missing from this MIDI excerpt of {metadata}. "
        "Fill in the gaps:",
    ],
    'attribute_denoising': [
        "The velocity values in this MIDI sequence have been corrupted. "
        "The original piece is {metadata}. Restore the original dynamics:",
        "Note durations have been altered in this MIDI excerpt of "
        "{metadata}. Restore them to their original values:",
        "Pitch values have been shifted randomly in parts of this MIDI "
        "sequence. The original piece is {metadata}. Correct them:",
    ],
    'track_denoising': [
        "Tracks have been incorrectly merged in this MIDI sequence. "
        "The original piece is {metadata}. Separate and restore them:",
        "Instrument assignments have been scrambled in this MIDI excerpt "
        "of {metadata}. Restore the correct program assignments:",
    ],
    'heavy_denoising': [
        "This MIDI sequence has been heavily corrupted with large sections "
        "removed. The original piece is {metadata}. Reconstruct it:",
        "Most of the following MIDI excerpt of {metadata} has been "
        "destroyed. Restore it as completely as possible:",
    ],
    'continuation': [
        "Continue the following MIDI sequence. The piece is {metadata}:",
        "Write a continuation for this MIDI excerpt of {metadata}:",
        "Complete the following MIDI passage from {metadata}:",
    ],
}


@dataclass
class Mid2MidCollatorConfig:
    """Configuration for the Mid2Mid collator."""

    max_seq_len: int = 2048

    # sentinel tokens are reused from the text tokenizer (e.g.,
    # <extra_id_0>..<extra_id_255> in CRAFT/T5-style tokenizers).
    # if sentinel_token_pattern is set, the collator will discover
    # sentinel token IDs from the text tokenizer at init time rather
    # than creating new ones. this ensures the model uses the same
    # sentinel semantics for both text and MIDI span corruption.
    sentinel_token_pattern: str = "<extra_id_{i}>"
    max_sentinels: int = 256

    # corruption task weights: keys are task types from MIDI_TASK_PROMPTS,
    # values are sampling weights (will be normalized to sum to 1.0)
    task_weights: dict[str, float] = field(default_factory=lambda: {
        'beat_denoising': 0.15,
        'measure_denoising': 0.15,
        'note_denoising': 0.15,
        'attribute_denoising': 0.10,
        'track_denoising': 0.05,
        'heavy_denoising': 0.20,
        'continuation': 0.20,
    })

    # corruption intensity per task type
    corruption_params: dict[str, dict] = field(default_factory=lambda: {
        'beat_denoising': {
            'mask_ratio': 0.15,
            'mean_span_beats': 2.0,
            'unit': 'beat',
        },
        'measure_denoising': {
            'mask_ratio': 0.30,
            'mean_span_measures': 2.0,
            'unit': 'measure',
        },
        'note_denoising': {
            'mask_ratio': 0.15,
            'mean_span_notes': 3.0,
            'unit': 'note',
        },
        'attribute_denoising': {
            'corruption_ratio': 0.20,
            'attributes': ['velocity', 'duration', 'pitch'],
            'unit': 'attribute',
        },
        'track_denoising': {
            'merge_probability': 0.5,
            'unit': 'track',
        },
        'heavy_denoising': {
            'mask_ratio': 0.50,
            'mean_span_measures': 4.0,
            'unit': 'measure',
        },
        'continuation': {
            'prefix_ratio_range': (0.2, 0.5),
            'unit': 'sequence',
        },
    })

    # augmentation
    pitch_shift_range: int = 6
    tempo_stretch_range: tuple[float, float] = (0.8, 1.2)

    # measure filtering
    min_measures: int = 4
    max_measures: int = 64

    # batch packing
    pack_sequences: bool = True
    pad_token_id: int = 0


def merge_vocabularies(text_tokenizer, midi_vocab: dict[str, int]) -> dict[str, int]:
    """Append MIDI tokens after the text tokenizer's vocabulary."""
    base_size = int(text_tokenizer.vocab_size)
    combined: dict[str, int] = {}
    text_vocab = text_tokenizer.get_vocab() if hasattr(text_tokenizer, "get_vocab") else {}
    combined.update(text_vocab)
    next_id = max(combined.values(), default=-1) + 1
    next_id = max(next_id, base_size)
    for tok, _ in sorted(midi_vocab.items(), key=lambda kv: kv[1]):
        if tok in combined:
            continue
        combined[tok] = next_id
        next_id += 1
    return combined


class Mid2MidCollator:
    """Concrete instruction-conditioned collator for MIDI pre-training."""

    def __init__(
        self,
        midi_tokenizer,
        text_tokenizer,
        config: Mid2MidCollatorConfig,
        seed: int | None = None,
    ) -> None:
        self.midi_tokenizer = midi_tokenizer
        self.text_tokenizer = text_tokenizer
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._py_random = __import__("random").Random(seed)

        # Build merged vocab so MIDI tokens have unique IDs offset from text.
        midi_vocab = self._extract_midi_vocab(midi_tokenizer)
        self.combined_vocab = merge_vocabularies(text_tokenizer, midi_vocab)
        self.midi_id_offset = int(getattr(text_tokenizer, "vocab_size", 0))
        # Discover sentinel ids in the text vocab.
        self.sentinel_ids = self._discover_sentinels()
        self.sentinel_start_id = self.sentinel_ids[0] if self.sentinel_ids else (
            self.midi_id_offset + len(midi_vocab) + 100
        )

    @staticmethod
    def _extract_midi_vocab(midi_tokenizer) -> dict[str, int]:
        vocab = getattr(midi_tokenizer, "vocab", None)
        if isinstance(vocab, dict):
            return dict(vocab)
        if isinstance(vocab, list) and vocab and isinstance(vocab[0], dict):
            out: dict[str, int] = {}
            offset = 0
            for sub in vocab:
                for k, v in sub.items():
                    if k not in out:
                        out[k] = v + offset
                offset += len(sub)
            return out
        return {}

    def _discover_sentinels(self) -> list[int]:
        out: list[int] = []
        pattern = self.config.sentinel_token_pattern
        for i in range(self.config.max_sentinels):
            tok = pattern.format(i=i)
            if hasattr(self.text_tokenizer, "convert_tokens_to_ids"):
                tid = self.text_tokenizer.convert_tokens_to_ids(tok)
                if tid is not None and tid != getattr(self.text_tokenizer, "unk_token_id", None):
                    out.append(int(tid))
                    continue
            voc = self.combined_vocab
            if tok in voc:
                out.append(int(voc[tok]))
        return out

    def _midi_id_to_global(self, mid_id: int) -> int:
        return mid_id + self.midi_id_offset

    def _process_single(
        self,
        path: Path,
        metadata: dict | None,
    ) -> tuple[list[int], list[int]] | None:
        try:
            score = AdapterScore.from_file(str(path))
        except Exception:
            return None
        # Augmentation: pitch shift
        if self.config.pitch_shift_range > 0:
            shift = int(self._rng.integers(-self.config.pitch_shift_range, self.config.pitch_shift_range + 1))
            if shift != 0:
                for tr in score.tracks:
                    if tr.is_drum:
                        continue
                    for n in tr.notes:
                        n.pitch = max(0, min(127, n.pitch + shift))
        # Tempo stretch
        lo, hi = self.config.tempo_stretch_range
        if hi > lo:
            stretch = float(self._rng.uniform(lo, hi))
            for tempo in score.tempos:
                tempo.tempo = float(tempo.tempo * stretch)

        # Tokenize
        try:
            tokseq = self.midi_tokenizer.encode(score)
        except Exception:
            return None
        if isinstance(tokseq, list):
            if not tokseq:
                return None
            seq = tokseq[0]
        else:
            seq = tokseq
        ids = list(seq.ids) if hasattr(seq, "ids") and seq.ids else []
        tokens = list(seq.tokens) if hasattr(seq, "tokens") else []
        if not ids or not tokens:
            return None

        # Bar-token filtering
        bar_positions = [i for i, t in enumerate(tokens) if t.startswith("Bar")]
        if len(bar_positions) < self.config.min_measures:
            return None
        if len(bar_positions) > self.config.max_measures:
            cut_pos = bar_positions[self.config.max_measures]
            ids = ids[:cut_pos]
            tokens = tokens[:cut_pos]

        midi_vocab = self._extract_midi_vocab(self.midi_tokenizer)

        # Sample task
        keys = list(self.config.task_weights.keys())
        weights = np.array([self.config.task_weights[k] for k in keys], dtype=np.float64)
        weights /= weights.sum()
        task = str(self._rng.choice(keys, p=weights))

        # Apply corruption
        sentinel_start = self.sentinel_start_id
        rng = self._rng
        if task == "beat_denoising":
            enc_mid, dec_mid = _corruption.beat_denoising(ids, midi_vocab, sentinel_start_id=sentinel_start, rng=rng)
        elif task == "measure_denoising":
            enc_mid, dec_mid = _corruption.measure_denoising(ids, midi_vocab, sentinel_start_id=sentinel_start, rng=rng)
        elif task == "note_denoising":
            enc_mid, dec_mid = _corruption.note_denoising(ids, midi_vocab, sentinel_start_id=sentinel_start, rng=rng)
        elif task == "attribute_denoising":
            enc_mid, dec_mid = _corruption.attribute_denoising(ids, midi_vocab, rng=rng)
        elif task == "heavy_denoising":
            enc_mid, dec_mid = _corruption.heavy_denoising(ids, midi_vocab, sentinel_start_id=sentinel_start, rng=rng)
        elif task == "continuation":
            enc_mid, dec_mid = _corruption.continuation(ids, midi_vocab, rng=rng)
        else:
            # Default to no corruption: full reconstruction
            enc_mid, dec_mid = list(ids), list(ids)

        # Offset MIDI IDs into the global vocabulary range
        enc_mid_global = [self._midi_id_to_global(int(t)) if int(t) < self.midi_id_offset else int(t) for t in enc_mid]
        dec_mid_global = [self._midi_id_to_global(int(t)) if int(t) < self.midi_id_offset else int(t) for t in dec_mid]

        # Format prompt
        templates = MIDI_TASK_PROMPTS.get(task, MIDI_TASK_PROMPTS["note_denoising"])
        template = self._py_random.choice(templates)
        meta_str = self._format_metadata(metadata)
        prompt = template.format(metadata=meta_str)
        if hasattr(self.text_tokenizer, "encode"):
            text_ids = list(self.text_tokenizer.encode(prompt))
        else:
            text_ids = []

        encoder_ids = list(text_ids) + enc_mid_global
        decoder_ids = dec_mid_global
        return encoder_ids, decoder_ids

    @staticmethod
    def _format_metadata(metadata: dict | None) -> str:
        if not metadata:
            return "an unknown piece"
        parts = []
        if "title" in metadata:
            parts.append(str(metadata["title"]))
        if "composer" in metadata:
            parts.append(f"by {metadata['composer']}")
        if not parts:
            return "an unknown piece"
        return " ".join(parts)

    def __call__(
        self,
        midi_paths: list[Path],
        metadata: list[dict] | None = None,
    ) -> dict[str, np.ndarray]:
        """Collate a batch of MIDI files into training examples.

        For each MIDI file:
        1. Parse with midi_tokenizer into token IDs
        2. Sample a corruption task type from config.task_weights
        3. Apply the corresponding corruption strategy
        4. Sample a natural language prompt template for that task type
        5. Format the prompt with score metadata
        6. Encode: [text_prompt_tokens] + [corrupted_midi_tokens]
        7. Decode: [original_midi_tokens] (or continuation segment)

        Args:
            midi_paths: paths to .mid files
            metadata: optional per-file metadata dicts with keys like
                "composer", "title", "key", "time_signature", "era"

        Returns dict of NumPy int32 arrays of shape ``(batch, max_seq_len)``.
        """
        max_len = self.config.max_seq_len
        examples: list[tuple[list[int], list[int]]] = []
        for i, p in enumerate(midi_paths):
            md = metadata[i] if metadata is not None and i < len(metadata) else None
            res = self._process_single(Path(p), md)
            if res is None:
                continue
            enc, dec = res
            # truncate per example so packing can fit
            if len(enc) > max_len:
                enc = enc[:max_len]
            if len(dec) > max_len:
                dec = dec[:max_len]
            examples.append((enc, dec))

        if not examples:
            empty = np.zeros((1, max_len), dtype=np.int32)
            return {
                "input_ids": empty,
                "decoder_input_ids": empty.copy(),
                "labels": empty.copy(),
                "segment_ids": empty.copy(),
                "position_ids": empty.copy(),
                "decoder_segment_ids": empty.copy(),
                "decoder_position_ids": empty.copy(),
            }

        if self.config.pack_sequences:
            packed_enc: list[list[int]] = [[]]
            packed_dec: list[list[int]] = [[]]
            packed_eseg: list[list[int]] = [[]]
            packed_dseg: list[list[int]] = [[]]
            packed_epos: list[list[int]] = [[]]
            packed_dpos: list[list[int]] = [[]]
            seg_idx = 1
            for enc, dec in examples:
                # Decide whether to start a new pack: enforce that BOTH enc and
                # dec fit within max_len after appending.
                cur_enc_len = len(packed_enc[-1])
                cur_dec_len = len(packed_dec[-1])
                if cur_enc_len + len(enc) > max_len or cur_dec_len + len(dec) > max_len:
                    if cur_enc_len == 0 and cur_dec_len == 0:
                        # Empty pack — accept this oversized example by truncation
                        enc = enc[:max_len]
                        dec = dec[:max_len]
                    else:
                        packed_enc.append([])
                        packed_dec.append([])
                        packed_eseg.append([])
                        packed_dseg.append([])
                        packed_epos.append([])
                        packed_dpos.append([])
                        seg_idx = 1
                packed_enc[-1].extend(enc)
                packed_dec[-1].extend(dec)
                packed_eseg[-1].extend([seg_idx] * len(enc))
                packed_dseg[-1].extend([seg_idx] * len(dec))
                packed_epos[-1].extend(range(1, len(enc) + 1))
                packed_dpos[-1].extend(range(1, len(dec) + 1))
                seg_idx += 1
        else:
            packed_enc = [list(e) for e, _ in examples]
            packed_dec = [list(d) for _, d in examples]
            packed_eseg = [[1] * len(e) for e in packed_enc]
            packed_dseg = [[1] * len(d) for d in packed_dec]
            packed_epos = [list(range(1, len(e) + 1)) for e in packed_enc]
            packed_dpos = [list(range(1, len(d) + 1)) for d in packed_dec]

        batch = len(packed_enc)
        pad = self.config.pad_token_id

        def _pad(rows: list[list[int]], fill: int = pad) -> np.ndarray:
            arr = np.full((batch, max_len), fill, dtype=np.int32)
            for i, row in enumerate(rows):
                row = row[:max_len]
                arr[i, : len(row)] = row
            return arr

        input_ids = _pad(packed_enc)
        decoder_input_ids = _pad(packed_dec)
        segment_ids = _pad(packed_eseg, fill=0)
        position_ids = _pad(packed_epos, fill=0)
        decoder_segment_ids = _pad(packed_dseg, fill=0)
        decoder_position_ids = _pad(packed_dpos, fill=0)
        labels = decoder_input_ids.copy()

        return {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "segment_ids": segment_ids,
            "position_ids": position_ids,
            "decoder_segment_ids": decoder_segment_ids,
            "decoder_position_ids": decoder_position_ids,
        }
