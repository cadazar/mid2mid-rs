"""Universal MIDI Tokenizer — unified interface over multiple tokenization formats.

Provides a single vocabulary and API for encoding/decoding MIDI with any of the
supported tokenization formats (REMI, TSD, MIDILike, Structured, etc.).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .classes import TokenizerConfig, TokSequence
from .midi_adapter import AdapterScore
from .tokenizations import REMI, TSD, MIDILike, Structured

# Registry of format name → tokenizer class (flat-vocab only).
_FORMAT_REGISTRY: dict[str, type] = {
    "remi": REMI,
    "tsd": TSD,
    "midi_like": MIDILike,
    "structured": Structured,
}

# Default formats when none specified.
_DEFAULT_FORMATS = ["remi", "tsd"]


class UniversalMidiTokenizer:
    """Wrapper providing a unified interface over multiple tokenization formats.

    All formats share a single merged vocabulary where common tokens
    (Pitch_0..127, Velocity_*, Bar, Position_*, Duration_*, Tempo_*,
    TimeSig_*) map to the same IDs and format-specific tokens get unique
    IDs after the shared ones.
    """

    def __init__(
        self,
        config: TokenizerConfig | None = None,
        formats: list[str] | None = None,
        format_weights: dict[str, float] | None = None,
    ) -> None:
        self._config = config or TokenizerConfig()
        self._format_names: list[str] = formats or list(_DEFAULT_FORMATS)
        for f in self._format_names:
            if f not in _FORMAT_REGISTRY:
                raise ValueError(f"Unknown format {f!r}. Available: {list(_FORMAT_REGISTRY)}")

        # Build per-format tokenizers
        self._tokenizers: dict[str, Any] = {}
        for name in self._format_names:
            cls = _FORMAT_REGISTRY[name]
            self._tokenizers[name] = cls(self._config)

        # Sampling weights
        if format_weights is not None:
            self._format_weights = {
                f: format_weights.get(f, 0.0) for f in self._format_names
            }
        else:
            w = 1.0 / len(self._format_names)
            self._format_weights = {f: w for f in self._format_names}
        # Normalize
        total = sum(self._format_weights.values())
        if total > 0:
            self._format_weights = {f: v / total for f, v in self._format_weights.items()}

        # Build the unified vocabulary
        self._unified_vocab: dict[str, int] = {}
        self._format_token_sets: dict[str, set[str]] = {}
        self._format_sentinel_tokens: dict[str, str] = {}
        self.build_unified_vocab()

        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def build_unified_vocab(self) -> None:
        """Construct the merged vocabulary from all active formats.

        Shared tokens get one ID; format-specific tokens get unique IDs.
        Format sentinel tokens are appended last.
        """
        # Collect all tokens per format
        per_format: dict[str, dict[str, int]] = {}
        for name, tok in self._tokenizers.items():
            v = tok.vocab
            if isinstance(v, list):
                # Multi-vocab tokenizer — flatten
                flat: dict[str, int] = {}
                for sub in v:
                    for k, vid in sub.items():
                        if k not in flat:
                            flat[k] = vid
                per_format[name] = flat
            else:
                per_format[name] = dict(v)
            self._format_token_sets[name] = set(per_format[name].keys())

        # Identify shared tokens (present in ALL formats)
        all_sets = [set(v.keys()) for v in per_format.values()]
        shared = set.intersection(*all_sets) if all_sets else set()

        # Assign IDs: shared tokens first (sorted for determinism)
        next_id = 0
        self._unified_vocab = {}
        for tok in sorted(shared):
            self._unified_vocab[tok] = next_id
            next_id += 1

        # Format-specific tokens (sorted, per format in order)
        for name in self._format_names:
            specific = sorted(self._format_token_sets[name] - shared)
            for tok in specific:
                if tok not in self._unified_vocab:
                    self._unified_vocab[tok] = next_id
                    next_id += 1

        # Format sentinel tokens
        for name in self._format_names:
            sentinel = f"<{name}>"
            self._format_sentinel_tokens[name] = sentinel
            self._unified_vocab[sentinel] = next_id
            next_id += 1

        # Build per-format ID mapping: original_id -> unified_id
        self._format_id_maps: dict[str, dict[int, int]] = {}
        self._format_id_reverse: dict[str, dict[int, int]] = {}
        for name in self._format_names:
            fwd: dict[int, int] = {}
            rev: dict[int, int] = {}
            orig_vocab = per_format[name]
            for tok, orig_id in orig_vocab.items():
                unified_id = self._unified_vocab[tok]
                fwd[orig_id] = unified_id
                rev[unified_id] = orig_id
            self._format_id_maps[name] = fwd
            self._format_id_reverse[name] = rev

    @property
    def vocab(self) -> dict[str, int]:
        """Return the shared unified vocabulary."""
        return dict(self._unified_vocab)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including format sentinels."""
        return len(self._unified_vocab)

    @property
    def format_sentinel(self) -> dict[str, str]:
        """Map format names to their sentinel tokens."""
        return dict(self._format_sentinel_tokens)

    @property
    def formats(self) -> list[str]:
        """List of active format names."""
        return list(self._format_names)

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(
        self,
        score: AdapterScore,
        format: str | None = None,
    ) -> TokSequence:
        """Encode a score, optionally specifying a format.

        If *format* is ``None``, a format is sampled randomly according to
        ``format_weights``.  Returns a :class:`TokSequence` with IDs mapped
        into the unified vocabulary.
        """
        if format is None:
            names = list(self._format_weights.keys())
            weights = np.array([self._format_weights[n] for n in names])
            format = str(self._rng.choice(names, p=weights))

        if format not in self._tokenizers:
            raise ValueError(f"Format {format!r} not available")

        tok = self._tokenizers[format]
        result = tok.encode(score)
        seq = result[0] if isinstance(result, list) else result

        # Remap IDs to unified vocab
        fwd = self._format_id_maps[format]
        unified_ids = [fwd.get(int(i), int(i)) for i in seq.ids]
        unified_tokens = list(seq.tokens) if seq.tokens else []

        return TokSequence(
            tokens=unified_tokens,
            ids=unified_ids,
            events=list(seq.events) if seq.events else [],
        )

    def decode(self, tokens: TokSequence | list[int], format: str) -> AdapterScore:
        """Decode tokens using the specified format's tokenizer.

        Token IDs are mapped back from unified to format-local IDs.
        """
        if format not in self._tokenizers:
            raise ValueError(f"Format {format!r} not available")

        rev = self._format_id_reverse[format]
        if isinstance(tokens, TokSequence):
            ids = tokens.ids
        else:
            ids = tokens

        local_ids = [rev.get(int(i), int(i)) for i in ids]

        tok = self._tokenizers[format]
        # Build a TokSequence with local IDs and let the tokenizer decode
        local_seq = TokSequence(ids=local_ids)
        # Convert ids to tokens via the tokenizer's vocab
        inv_vocab = {v: k for k, v in (tok.vocab if isinstance(tok.vocab, dict) else {}).items()}
        local_seq.tokens = [inv_vocab.get(i, f"UNK_{i}") for i in local_ids]

        return tok.decode(local_seq)

    # ------------------------------------------------------------------
    # Format mask
    # ------------------------------------------------------------------

    def get_format_mask(self, format: str) -> np.ndarray:
        """Return a boolean array of shape ``(vocab_size,)`` that is True for
        token IDs valid in the given format.

        Useful for masking the softmax during inference.
        """
        if format not in self._format_token_sets:
            raise ValueError(f"Format {format!r} not available")
        mask = np.zeros(self.vocab_size, dtype=bool)
        valid_tokens = self._format_token_sets[format]
        # Also include the format's own sentinel
        sentinel = self._format_sentinel_tokens.get(format)
        for tok, tid in self._unified_vocab.items():
            if tok in valid_tokens or tok == sentinel:
                mask[tid] = True
        return mask

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the unified tokenizer config and vocab to JSON."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Serialize config, converting non-JSON-safe keys (tuples) to strings
        config_dict: dict[str, Any] = {}
        if hasattr(self._config, "to_dict"):
            raw = self._config.to_dict()
            for k, v in raw.items():
                try:
                    json.dumps({k: v})
                    config_dict[k] = v
                except (TypeError, ValueError):
                    config_dict[k] = str(v)

        data = {
            "formats": self._format_names,
            "format_weights": self._format_weights,
            "vocab": self._unified_vocab,
            "format_sentinels": self._format_sentinel_tokens,
            "config": config_dict,
        }
        with open(path / "universal_tokenizer.json", "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "UniversalMidiTokenizer":
        """Load a previously saved universal tokenizer."""
        path = Path(path)
        json_path = path / "universal_tokenizer.json"
        with open(json_path) as f:
            data = json.load(f)

        # Use a fresh default config — the saved config is for reference only
        # since TokenizerConfig has tuple-keyed dicts that don't round-trip via JSON.
        obj = cls(
            config=TokenizerConfig(),
            formats=data["formats"],
            format_weights=data.get("format_weights"),
        )
        # Overwrite the vocab with saved version for exact reproducibility
        obj._unified_vocab = {k: int(v) for k, v in data["vocab"].items()}
        obj._format_sentinel_tokens = data.get("format_sentinels", obj._format_sentinel_tokens)
        return obj

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def set_seed(self, seed: int) -> None:
        """Set the random seed for format sampling."""
        self._rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return (
            f"UniversalMidiTokenizer(formats={self._format_names}, "
            f"vocab_size={self.vocab_size})"
        )
