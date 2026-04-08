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
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


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
    num_sentinels: int = 100

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


class Mid2MidCollator(abc.ABC):
    """Produces encoder/decoder pairs for text+MIDI pre-training.

    The collator combines a MIDI tokenizer with a text tokenizer to produce
    multimodal sequences. The encoder input contains a natural language prompt
    (tokenized with the text tokenizer) followed by corrupted MIDI tokens. The
    decoder target is the original (or continuation) MIDI sequence.

    Corruption is instruction-conditioned: each corruption type has associated
    natural language prompts that describe what was done, so the model learns
    musical concepts (beats, measures, dynamics, etc.) through the language
    signal from the very start of training. This mirrors the approach in
    task_prompts.py for text pre-training.

    All outputs are NumPy arrays for JAX/Flax compatibility.
    """

    @abc.abstractmethod
    def __init__(
        self,
        midi_tokenizer,
        text_tokenizer,
        config: Mid2MidCollatorConfig,
    ) -> None: ...

    @abc.abstractmethod
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

        Returns:
            dict with keys:
                input_ids: (batch, max_seq_len) int32
                decoder_input_ids: (batch, max_seq_len) int32
                labels: (batch, max_seq_len) int32
                attention_mask: (batch, max_seq_len) bool
                decoder_attention_mask: (batch, max_seq_len) bool
        """
        ...

    @abc.abstractmethod
    def merge_vocabularies(
        self,
        text_tokenizer,
        midi_vocab: dict[str, int],
    ) -> dict[str, int]:
        """Merge MIDI tokens into an existing text vocabulary.

        MIDI tokens (Pitch_0..127, Velocity_0..31, Duration_0..N, Bar,
        Position_0..63, Tempo_0..48, TimeSig_N/D, Program_0..127, and
        sentinel tokens s_0..s_99) are appended after the last text
        token ID.

        Returns:
            combined vocabulary mapping token strings to integer IDs
        """
        ...
