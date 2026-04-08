"""Stub interface for the Mid2Mid collator.

This module defines the expected API for a multimodal text+MIDI collator
that produces encoder/decoder pairs for sequence-to-sequence pre-training.

The collator should support:
- Multiple corruption modes (UL2-style: R=regular denoising, X=extreme
  denoising, S=sequential/continuation)
- Sentinel-based span corruption (T5-style, not BERT <mask>)
- Mixed text+MIDI sequences where the encoder sees a natural language prompt
  describing the corruption level and score metadata, followed by corrupted
  MIDI tokens, and the decoder sees the original MIDI sequence
- NumPy array output (for JAX/Flax compatibility, no PyTorch dependency)
- Vocabulary merging with a pre-trained text tokenizer
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class Mid2MidCollatorConfig:
    """Configuration for the Mid2Mid collator."""

    max_seq_len: int = 2048
    num_sentinels: int = 100

    # ul2 mode weights: must sum to 1.0
    ul2_r_weight: float = 0.4
    ul2_x_weight: float = 0.4
    ul2_s_weight: float = 0.2

    # corruption parameters per mode
    r_mask_ratio: float = 0.15
    r_mean_span_length: float = 3.0
    x_mask_ratio: float = 0.50
    x_mean_span_length: float = 8.0
    s_prefix_ratio_range: tuple[float, float] = (0.2, 0.5)

    # augmentation
    pitch_shift_range: int = 6
    tempo_stretch_range: tuple[float, float] = (0.8, 1.2)

    # prompt templates per mode
    prompt_templates: dict[str, str] = field(default_factory=lambda: {
        "R": (
            "The following MIDI sequence has light corruption. "
            "The original piece is {metadata}. "
            "Please restore it to its original state: "
        ),
        "X": (
            "The following MIDI sequence is heavily corrupted. "
            "The original piece is {metadata}. "
            "Please restore it to its original state: "
        ),
        "S": (
            "Continue the following MIDI sequence. "
            "The piece is {metadata}: "
        ),
    })


class Mid2MidCollator(abc.ABC):
    """Produces encoder/decoder pairs for text+MIDI pre-training.

    The collator combines a MIDI tokenizer with a text tokenizer to produce
    multimodal sequences. The encoder input contains a natural language prompt
    (tokenized with the text tokenizer) followed by corrupted MIDI tokens. The
    decoder target is the original (or continuation) MIDI sequence.

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

        Args:
            midi_paths: paths to .mid files
            metadata: optional per-file metadata dicts with keys like
                "composer", "title", "key", "time_signature"

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

        MIDI tokens are appended after the last text token ID.

        Returns:
            combined vocabulary mapping token strings to integer IDs
        """
        ...
