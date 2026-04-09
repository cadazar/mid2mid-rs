"""Instruction-conditioned collator for Mid2Mid pre-training.

Produces **unpacked** per-example dicts with 5 keys matching the CRAFT
training interface.  The CRAFT packing layer (``pack_documents``) handles
segment_ids / position_ids separately.

Output per example::

    {
        "input_ids":              np.array(int32, (max_seq_len,)),
        "decoder_input_ids":      np.array(int32, (max_seq_len,)),
        "labels":                 np.array(int32, (max_seq_len,)),
        "attention_mask":         np.array(int32, (max_seq_len,)),
        "decoder_attention_mask": np.array(int32, (max_seq_len,)),
    }

Batch output stacks along axis 0 → ``(batch_size, max_seq_len)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from miditok import corruption as _corruption
from miditok.corruption import analyze_midi_structure
from miditok.midi_adapter import AdapterScore

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

MIDI_TASK_PROMPTS: dict[str, list[str]] = {
    "beat_denoising": [
        "Several beats have been removed from this MIDI sequence. "
        "The original piece is {metadata}. Restore the missing beats: ",
        "Beats have been dropped from the following MIDI excerpt. "
        "This is {metadata}. Reconstruct them: ",
    ],
    "measure_denoising": [
        "Multiple measures have been removed from this MIDI sequence. "
        "The original piece is {metadata}. Restore them: ",
        "The following MIDI excerpt of {metadata} has had entire measures "
        "deleted. Reconstruct the missing measures: ",
    ],
    "note_denoising": [
        "Individual notes have been randomly removed from this MIDI "
        "sequence. The original piece is {metadata}. Restore them: ",
        "Some notes are missing from this MIDI excerpt of {metadata}. "
        "Fill in the gaps: ",
    ],
    "attribute_denoising": [
        "The velocity values in this MIDI sequence have been corrupted. "
        "The original piece is {metadata}. Restore the original dynamics: ",
        "Note durations have been altered in this MIDI excerpt of "
        "{metadata}. Restore them to their original values: ",
        "Pitch values have been shifted randomly in parts of this MIDI "
        "sequence. The original piece is {metadata}. Correct them: ",
    ],
    "heavy_denoising": [
        "This MIDI sequence has been heavily corrupted with large sections "
        "removed. The original piece is {metadata}. Reconstruct it: ",
        "Most of the following MIDI excerpt of {metadata} has been "
        "destroyed. Restore it as completely as possible: ",
    ],
    "continuation": [
        "Here are {prefix_measures} measures of {metadata}. "
        "Continue for {continuation_measures} measures: ",
        "The following {prefix_beats} beats are from {metadata}. "
        "Continue for {continuation_beats} beats: ",
        "Complete this piece to the end. The piece is {metadata}: ",
    ],
    "no_corruption": [
        "The following MIDI sequence is from {metadata}. "
        "Reproduce it exactly: ",
    ],
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Mid2MidCollatorConfig:
    """Configuration for the Mid2Mid collator."""

    max_seq_len: int = 2048
    pad_token_id: int = 0
    label_ignore_id: int = -100  # for padding positions in labels

    # Corruption task weights (normalized to sum to 1.0)
    task_weights: dict[str, float] = field(default_factory=lambda: {
        "beat_denoising": 0.15,
        "measure_denoising": 0.15,
        "note_denoising": 0.15,
        "attribute_denoising": 0.10,
        "heavy_denoising": 0.20,
        "continuation": 0.20,
        "no_corruption": 0.05,
    })

    # Augmentation
    pitch_shift_range: int = 6
    tempo_stretch_range: tuple[float, float] = (0.8, 1.2)


# ---------------------------------------------------------------------------
# Vocabulary merging
# ---------------------------------------------------------------------------


def merge_vocabularies(
    text_tokenizer: Any,
    midi_tokenizer: Any,
) -> dict[str, int]:
    """Append MIDI tokens after the text tokenizer's vocabulary.

    Sentinel tokens remain at their existing positions (shared across text
    and MIDI).
    """
    base_size = int(getattr(text_tokenizer, "vocab_size", 0))

    # Get text vocab
    if hasattr(text_tokenizer, "get_vocab"):
        text_vocab: dict[str, int] = dict(text_tokenizer.get_vocab())
    else:
        text_vocab = {}

    combined: dict[str, int] = dict(text_vocab)
    next_id = max(combined.values(), default=-1) + 1
    next_id = max(next_id, base_size)

    # Get MIDI vocab
    midi_vocab = _extract_midi_vocab(midi_tokenizer)

    # Identify sentinel tokens (they should stay at their text-vocab positions)
    sentinel_tokens = {k for k in text_vocab if k.startswith("<extra_id_")}

    for tok in sorted(midi_vocab.keys()):
        if tok in combined:
            continue
        if tok in sentinel_tokens:
            continue
        combined[tok] = next_id
        next_id += 1

    return combined


def _extract_midi_vocab(midi_tokenizer: Any) -> dict[str, int]:
    """Extract a flat vocab dict from a midi tokenizer."""
    vocab = getattr(midi_tokenizer, "vocab", None)
    if vocab is None:
        vocab = getattr(midi_tokenizer, "_unified_vocab", {})
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


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


class Mid2MidCollator:
    """Instruction-conditioned collator for MIDI seq2seq pre-training.

    Takes segment dicts (from :class:`Mid2MidDataset`) and produces batched
    NumPy int32 arrays.
    """

    def __init__(
        self,
        midi_tokenizer: Any,
        text_tokenizer: Any,
        config: Mid2MidCollatorConfig,
        seed: int | None = None,
    ) -> None:
        self.midi_tokenizer = midi_tokenizer
        self.text_tokenizer = text_tokenizer
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._py_random = __import__("random").Random(seed)

        # Build merged vocab
        self.combined_vocab = merge_vocabularies(text_tokenizer, midi_tokenizer)
        self.midi_id_offset = int(getattr(text_tokenizer, "vocab_size", 0))

        # Discover sentinel token IDs from text tokenizer
        self.sentinel_token_ids = self._discover_sentinels()

    def _discover_sentinels(self, max_sentinels: int = 4096) -> list[int]:
        """Find sentinel token IDs in the text tokenizer, ordered by index."""
        out: list[int] = []
        for i in range(max_sentinels):
            tok = f"<extra_id_{i}>"
            if hasattr(self.text_tokenizer, "convert_tokens_to_ids"):
                tid = self.text_tokenizer.convert_tokens_to_ids(tok)
                if tid is not None and tid != getattr(self.text_tokenizer, "unk_token_id", None):
                    out.append(int(tid))
                    continue
            voc = self.combined_vocab
            if tok in voc:
                out.append(int(voc[tok]))
            else:
                break
        return out

    # ------------------------------------------------------------------
    # Per-example pipeline
    # ------------------------------------------------------------------

    def _process_segment(self, segment: dict[str, Any]) -> dict[str, np.ndarray] | None:
        """Process a single segment dict into a training example."""
        score: AdapterScore = segment["score"].copy()
        metadata = segment.get("metadata", {})

        # 1. Augmentation: pitch shift
        if self.config.pitch_shift_range > 0:
            shift = int(self._rng.integers(
                -self.config.pitch_shift_range,
                self.config.pitch_shift_range + 1,
            ))
            if shift != 0:
                for tr in score.tracks:
                    if tr.is_drum:
                        continue
                    for n in tr.notes:
                        n.pitch = max(0, min(127, n.pitch + shift))

        # Augmentation: tempo stretch
        lo, hi = self.config.tempo_stretch_range
        if hi > lo:
            stretch = float(self._rng.uniform(lo, hi))
            for tempo in score.tempos:
                tempo.tempo = float(tempo.tempo * stretch)

        # 2. Tokenize
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
        midi_ids = list(seq.ids) if hasattr(seq, "ids") and seq.ids else []
        midi_tokens = list(seq.tokens) if hasattr(seq, "tokens") else []
        if not midi_ids:
            return None

        # 3. Get format sentinel token (if universal tokenizer)
        format_sentinel_id = None
        if hasattr(self.midi_tokenizer, "format_sentinel"):
            # Universal tokenizer — pick format from the sequence context
            fmt_sentinels = self.midi_tokenizer.format_sentinel
            fmt = getattr(seq, "_format", None)
            if fmt and fmt in fmt_sentinels:
                sentinel_tok = fmt_sentinels[fmt]
                vocab = getattr(self.midi_tokenizer, "vocab", {})
                if isinstance(vocab, dict):
                    format_sentinel_id = vocab.get(sentinel_tok)

        # 4. Get MIDI vocab for structure analysis
        midi_vocab = _extract_midi_vocab(self.midi_tokenizer)

        # 5. Sample corruption task
        keys = list(self.config.task_weights.keys())
        weights = np.array([self.config.task_weights[k] for k in keys], dtype=np.float64)
        weights /= weights.sum()
        task = str(self._rng.choice(keys, p=weights))

        # 6. Apply corruption
        if task == "beat_denoising":
            enc_mid, dec_mid, _ = _corruption.beat_denoising(
                midi_ids, midi_vocab, self.sentinel_token_ids, rng=self._rng)
        elif task == "measure_denoising":
            enc_mid, dec_mid, _ = _corruption.measure_denoising(
                midi_ids, midi_vocab, self.sentinel_token_ids, rng=self._rng)
        elif task == "note_denoising":
            enc_mid, dec_mid, _ = _corruption.note_denoising(
                midi_ids, midi_vocab, self.sentinel_token_ids, rng=self._rng)
        elif task == "attribute_denoising":
            enc_mid, dec_mid, _ = _corruption.attribute_denoising(
                midi_ids, midi_vocab, self.sentinel_token_ids, rng=self._rng)
        elif task == "heavy_denoising":
            enc_mid, dec_mid, _ = _corruption.heavy_denoising(
                midi_ids, midi_vocab, self.sentinel_token_ids, rng=self._rng)
        elif task == "continuation":
            enc_mid, dec_mid, _ = _corruption.continuation(
                midi_ids, midi_vocab, self.sentinel_token_ids, rng=self._rng)
        elif task == "no_corruption":
            enc_mid, dec_mid = list(midi_ids), list(midi_ids)
        else:
            enc_mid, dec_mid = list(midi_ids), list(midi_ids)

        # Offset MIDI IDs into global vocab range
        enc_mid_global = [
            self._midi_id_to_global(int(t)) if int(t) < self.midi_id_offset else int(t)
            for t in enc_mid
        ]
        dec_mid_global = [
            self._midi_id_to_global(int(t)) if int(t) < self.midi_id_offset else int(t)
            for t in dec_mid
        ]

        # 7. Build prompt
        meta_str = self._format_metadata(metadata, task, segment, midi_ids, midi_vocab)
        templates = MIDI_TASK_PROMPTS.get(task, MIDI_TASK_PROMPTS["note_denoising"])
        template = self._pick_continuation_template(task, templates, segment, midi_ids, midi_vocab)
        prompt = self._format_prompt(template, meta_str, segment, midi_ids, midi_vocab, enc_mid, dec_mid)

        # 8. Encode prompt text
        if hasattr(self.text_tokenizer, "encode"):
            text_ids = list(self.text_tokenizer.encode(prompt))
        else:
            text_ids = []

        # 9. Build encoder: [text_prompt_ids] + [format_sentinel] + [corrupted_midi_ids]
        encoder_ids: list[int] = list(text_ids)
        if format_sentinel_id is not None:
            encoder_ids.append(format_sentinel_id)
        encoder_ids.extend(enc_mid_global)

        # 10. Build decoder: [decoder_midi_ids]
        decoder_ids: list[int] = list(dec_mid_global)

        # Truncate to max_seq_len
        max_len = self.config.max_seq_len
        if len(encoder_ids) > max_len:
            encoder_ids = encoder_ids[:max_len]
        if len(decoder_ids) > max_len:
            decoder_ids = decoder_ids[:max_len]

        # Pad both to same length (max_seq_len)
        pad = self.config.pad_token_id
        ignore = self.config.label_ignore_id

        enc_arr = np.full(max_len, pad, dtype=np.int32)
        enc_arr[:len(encoder_ids)] = encoder_ids

        dec_arr = np.full(max_len, pad, dtype=np.int32)
        dec_arr[:len(decoder_ids)] = decoder_ids

        labels = np.full(max_len, ignore, dtype=np.int32)
        labels[:len(decoder_ids)] = decoder_ids

        attn_mask = np.zeros(max_len, dtype=np.int32)
        attn_mask[:len(encoder_ids)] = 1

        dec_attn_mask = np.zeros(max_len, dtype=np.int32)
        dec_attn_mask[:len(decoder_ids)] = 1

        return {
            "input_ids": enc_arr,
            "decoder_input_ids": dec_arr,
            "labels": labels,
            "attention_mask": attn_mask,
            "decoder_attention_mask": dec_attn_mask,
        }

    def _midi_id_to_global(self, mid_id: int) -> int:
        return mid_id + self.midi_id_offset

    def _pick_continuation_template(
        self,
        task: str,
        templates: list[str],
        segment: dict[str, Any],
        midi_ids: list[int],
        midi_vocab: dict[str, int],
    ) -> str:
        """Pick an appropriate template, choosing continuation variants wisely."""
        if task != "continuation":
            return self._py_random.choice(templates)

        # For continuation, choose between measure-based, beat-based, and "complete" templates
        # "Complete" only if the piece is short enough
        n = len(midi_ids)
        max_len = self.config.max_seq_len
        if n <= max_len * 0.8:
            # Piece fits — all templates valid
            return self._py_random.choice(templates)
        else:
            # Exclude "Complete this piece" template
            filtered = [t for t in templates if "Complete this piece" not in t]
            return self._py_random.choice(filtered) if filtered else templates[0]

    def _format_prompt(
        self,
        template: str,
        meta_str: str,
        segment: dict[str, Any],
        midi_ids: list[int],
        midi_vocab: dict[str, int],
        enc_ids: list[int],
        dec_ids: list[int],
    ) -> str:
        """Format a prompt template, filling in all placeholders."""
        structure = analyze_midi_structure(midi_ids, midi_vocab)
        bars = structure["bar_indices"]
        beats = structure["beat_indices"] or structure["timing_indices"]

        # Count prefix/continuation measures and beats
        prefix_len = len(enc_ids)
        prefix_bars = sum(1 for b in bars if b < prefix_len)
        total_bars = len(bars)
        continuation_bars = max(1, total_bars - prefix_bars)

        prefix_beats = sum(1 for b in beats if b < prefix_len)
        total_beats = len(beats)
        continuation_beats = max(1, total_beats - prefix_beats)

        return template.format(
            metadata=meta_str,
            prefix_measures=prefix_bars,
            continuation_measures=continuation_bars,
            prefix_beats=prefix_beats,
            continuation_beats=continuation_beats,
        )

    @staticmethod
    def _format_metadata(
        metadata: dict | None,
        task: str | None = None,
        segment: dict | None = None,
        midi_ids: list[int] | None = None,
        midi_vocab: dict | None = None,
    ) -> str:
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

    # ------------------------------------------------------------------
    # Batch collation
    # ------------------------------------------------------------------

    def __call__(self, segments: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        """Collate a list of segment dicts into a batch.

        Returns dict of NumPy int32 arrays of shape ``(batch_size, max_seq_len)``.
        """
        max_len = self.config.max_seq_len
        examples: list[dict[str, np.ndarray]] = []

        for seg in segments:
            result = self._process_segment(seg)
            if result is not None:
                examples.append(result)

        if not examples:
            empty = np.zeros((1, max_len), dtype=np.int32)
            ignore_labels = np.full((1, max_len), self.config.label_ignore_id, dtype=np.int32)
            return {
                "input_ids": empty,
                "decoder_input_ids": empty.copy(),
                "labels": ignore_labels,
                "attention_mask": empty.copy(),
                "decoder_attention_mask": empty.copy(),
            }

        batch_size = len(examples)
        return {
            key: np.stack([ex[key] for ex in examples], axis=0)
            for key in (
                "input_ids",
                "decoder_input_ids",
                "labels",
                "attention_mask",
                "decoder_attention_mask",
            )
        }
