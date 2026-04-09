"""
Interface contract tests for mid2mid-rs MIDI pipeline integration.

These stubs define the exact interface that the Rust MIDI pipeline must
satisfy to integrate with the CRAFT training infrastructure. Each test
is skipped until the corresponding functionality is implemented.

See docs/midi_integration_interface.md for the full specification.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mid2mid-local'))

import pytest


@pytest.mark.skip(reason="requires _handle_midi_pretraining in UnifiedCRAFTCollator")
def test_midi_collator_output_keys():
    """MIDI collator must produce these exact keys."""
    required_keys = {
        "input_ids", "decoder_input_ids", "labels",
        "attention_mask", "decoder_attention_mask",
    }
    # packed output adds these
    packed_keys = {
        "segment_ids", "position_ids",
        "decoder_segment_ids", "decoder_position_ids",
    }
    # TODO: instantiate midi collator and verify output contains required_keys
    # TODO: when packing enabled, verify packed_keys are also present


@pytest.mark.skip(reason="requires _handle_midi_pretraining in UnifiedCRAFTCollator")
def test_midi_collator_output_shapes():
    """All outputs must be (batch_size, max_seq_len) int32."""
    import numpy as np
    # TODO: instantiate midi collator, process a batch, verify:
    # - all values are np.int32
    # - all shapes are (batch_size, max_length)
    # - encoder and decoder have same max_length dimension


@pytest.mark.skip(reason="requires MIDI packing support")
def test_midi_collator_segment_ids_1_based():
    """segment_ids must be 1-based (first segment = 1, padding = 0)."""
    # TODO: verify segment_ids[non_padding] >= 1
    # TODO: verify segment_ids[padding] == 0
    # TODO: verify position_ids follow same convention


@pytest.mark.skip(reason="requires _handle_midi_pretraining in UnifiedCRAFTCollator")
def test_midi_collator_encoder_decoder_same_length():
    """input_ids and decoder_input_ids must have identical seq_len dim."""
    # TODO: verify input_ids.shape[-1] == decoder_input_ids.shape[-1]


@pytest.mark.skip(reason="requires MIDI-aware sentinel masking")
def test_midi_collator_sentinel_tokens():
    """Sentinels must use <extra_id_N> with increasing indices per span."""
    # TODO: verify encoder contains <extra_id_0>, <extra_id_1>, etc.
    # TODO: verify decoder mirrors with sentinel + span content
    # TODO: verify sentinel indices increase (0, 1, 2, ...)


@pytest.mark.skip(reason="requires MIDI task prompt registration")
def test_midi_collator_prompt_not_corrupted():
    """Text prompt tokens must never be corrupted."""
    # TODO: when use_task_prompts=True, verify prompt tokens
    # appear unchanged at the start of input_ids


@pytest.mark.skip(reason="requires end-to-end MIDI pipeline")
def test_midi_collator_feeds_into_unified():
    """Output must be accepted by UnifiedCRAFTCollator's training loop."""
    # TODO: produce MIDI collator output, verify it can be:
    # 1. packed by CRAFTPackedPhase2Collator.pack_documents()
    # 2. converted to JAX arrays without error
    # 3. fed through the model forward pass shape checks


@pytest.mark.skip(reason="requires UnifiedMidiTokenizer implementation")
def test_universal_tokenizer_format_selection():
    """UnifiedMidiTokenizer must support encode(score, format='remi')."""
    # TODO: verify tokenizer accepts format parameter
    # TODO: verify different formats produce different token sequences
    # TODO: verify all formats produce valid token sequences


@pytest.mark.skip(reason="requires UnifiedMidiTokenizer implementation")
def test_universal_tokenizer_shared_vocab():
    """All formats share Bar, Position, Pitch, Velocity tokens."""
    # TODO: verify common structural tokens exist across all formats
    # TODO: verify special tokens (BOS, EOS, PAD, MASK) are shared
