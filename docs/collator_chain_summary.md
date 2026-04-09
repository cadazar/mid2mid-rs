# Collator Inheritance Chain

Reference for integrating the mid2mid-rs MIDI pipeline into the CRAFT training infrastructure.

## Inheritance Diagram

```
DataCollator (h2hcollator.py:94)
  |
BARTCollator (h2hcollator.py:1205)
  |
MultilingualCRAFTCollator (multilingual_craft_collator.py:84)
  |
Phase2MixedCollator (craft_phase2_collator.py:26)
  |
CRAFTPackedPhase2Collator (craft_packed_phase2_collator.py:20)
  |
PackedMultilingualCRAFTCollator (packed_multilingual_craft_collator.py:21)
  |
UnifiedCRAFTCollator (unified_craft_collator.py:26)
```

## Output Key Evolution

| Level | Class | Keys Added |
| ----- | ----- | ---------- |
| Base | DataCollator | `input_ids`, `decoder_input_ids`, `labels`, `attention_mask`, `decoder_attention_mask` |
| Packed | CRAFTPackedPhase2Collator | `segment_ids`, `position_ids`, `decoder_segment_ids`, `decoder_position_ids` |
| Merged (optional) | CRAFTPackedPhase2Collator | `merge_indices`, `decoder_mask` |

All arrays are `np.int32` with shape `(max_length,)` for single examples or `(batch_size, max_length)` for batches. Labels use `-100` for padding (ignored by cross-entropy loss).

## Per-Class Details

### 1. DataCollator

**File**: `h2hcollator.py:94-1202`
**Parent**: `@dataclasses.dataclass` (base)

Handles tokenization, noise application, subword feature lookup, and batching.

**`__call__` signature**:
```python
def __call__(self, examples: List[Dict], cooldown_phase=False,
             bucket_idx=None, tokenizer=None)
```

**Input**: List of dicts with `sentences` or `original_text` and optional `metadata`.

**Output**: `BatchEncoding` with 5 keys:
- `input_ids`: `[metadata] + [corrupted_text] + </s>`
- `decoder_input_ids`: `</s> + [labels][:-1]` (teacher forcing shift)
- `labels`: `[sentences joined with </s>] + </s>` (no BOS, -100 for padding)
- `attention_mask`: 1 for real tokens, 0 for padding
- `decoder_attention_mask`: same convention

**Key methods**: `_apply_deletion_noise`, `_sliding_window_slice`, `__post_init__` (epoch tracking, MeCab init)

### 2. BARTCollator

**File**: `h2hcollator.py:1205-1499`
**Parent**: `DataCollator`

Adds BART-style text infilling, sentence permutation, and T5-style sentinel masking.

**Does not override `__call__`** -- inherits from DataCollator. Parent calls `span_masking()` which routes here.

**Key methods added**:
- `permute_sentences()`: shuffles 3+ sentences randomly
- `span_masking()`: routes to morpheme or token-based masking
- `_token_based_infilling()`: BART-style single `<mask>` replacing spans
- `_token_based_sentinel_masking()`: T5-style `<extra_id_N>` per span (decreasing N per span). Returns `(encoder_ids, decoder_ids, next_sentinel_idx)`.

**Sentinel convention**: each masked span gets a unique `<extra_id_N>` token. Encoder replaces spans with sentinels; decoder is `<extra_id_0> [span0] <extra_id_1> [span1] ...`. Sentinel indices increase per span (0, 1, 2...). `start_sentinel_idx` parameter chains across calls.

### 3. MultilingualCRAFTCollator

**File**: `multilingual_craft_collator.py:84-1538`
**Parent**: `BARTCollator`

Adds multilingual support: language tokens, Hanja detection, script tokens, Han2Han transcription, token-level balancing across data sources.

**Overrides `__call__`**:
```python
def __call__(self, example: Dict|List[Dict], cooldown_phase=None,
             bucket_idx=None, tokenizer=None, return_source=False, padding=True)
```

**Input**: Single example dict or list of dicts. Examples have `original_text`, `language`, `source`, optional `metadata`.

**Output**: Same 5 keys. Token flow:
- Encoder: `[metadata] + [corrupted_content] + <lang_token> + [<script_token>]`
- Decoder: `<lang_token> + [<script_token>] + [decoder_content]`
- Labels: `[decoder_content] + <lang_token> + [<script_token>]`

**Key methods added**:
- `_instantiate_dsets()`: creates dataset iterators with token-level balancing
- `_build_hanja_token_lookup()`: vectorized Hanja detection in vocabulary
- `_classify_korean_sample()`: routes Korean samples to sub-buffers (heavy/light Hanja)
- `get_language_token()`: maps language to `<ko>`, `<en>`, `<zh>`, `<ja>`, `<vi>`, `<fr>`, `<de>`

**Han2Han transcription**: when `han2han_transcription_ratio > 0`, bidirectional Hanja-Hangul transcription at 50% probability. Tracks direction in `_training_mode`.

### 4. Phase2MixedCollator

**File**: `craft_phase2_collator.py:26-1625`
**Parent**: `MultilingualCRAFTCollator`

UL2-style mixture of denoisers with configurable mode ratios.

**Overrides `__call__`**:
```python
def __call__(self, examples: Dict|List[Dict], cooldown_phase=False,
             bucket_idx=None, tokenizer=None, return_source=False,
             use_length_sampling=True, padding=True,
             morpheme_tokenizers=None)
```

**Routing logic** (single example path):
1. Check for byte reconstruction eligibility
2. Check for temporal continuation (year metadata)
3. Sample mode from `mode_ratios`: `denoising` (R, 40%), `denoising_heavy` (X, 40%), `continuation` (S, 20%)
4. Route to handler

**Training modes tracked in `_training_mode`**:
- `denoising_sentinel`, `denoising_bart`
- `morpheme_denoising_sentinel`, `morpheme_denoising_bart`
- `denoising_heavy_sentinel`
- `continuation`
- `temporal_continuation`
- `byte_reconstruction`
- Suffixed with `_transcription_hangul_to_hanja` or `_transcription_hanja_to_hangul`

**Key methods added**:
- `_sample_mode()`, `_sample_mode_by_length()`: mode selection
- `_sample_denoiser_config()`: samples (lambda, ratio) tuples for R/X/morpheme denoisers
- `_collate_morpheme_denoising()`: language-specific morpheme tokenization + corruption
- `_collate_heavy_denoising()`: 50% corruption, token-level only
- `_collate_continuation()`: prefix LM style
- `_collate_temporal_continuation()`, `_collate_byte_reconstruction()`
- `_split_document()`: splits long docs favoring decoder (60-80%)

**UL2 denoiser configs**: `r_denoiser_configs`, `x_denoiser_configs`, `morpheme_denoiser_configs` -- each a list of (lambda, ratio) tuples for multi-config denoising.

### 5. CRAFTPackedPhase2Collator

**File**: `craft_packed_phase2_collator.py:20-593`
**Parent**: `Phase2MixedCollator`

Packs multiple documents into single sequences for training efficiency.

**Overrides `__call__`**: handles single and batch, returns packed examples with extended keys.

**Packing format**:
- Encoder: `[doc1] <lang1> [doc2] <lang2> [doc3] <lang3> [PAD...]`
- Decoder: `<lang1> [gen1] <lang2> [gen2] <lang3> [gen3] [PAD...]`
- `segment_ids`: 0=padding, 1=first doc, 2=second doc, ...
- `position_ids`: 0=padding, 1-based sequential within each document (resets per segment)

**Key methods added**:
- `pack_documents()` (line 68): first-fit-decreasing bin packing. Sorts by length, packs into bins respecting `max_length`.
- `_finalize_pack()` (line 275): pads to `max_length`, converts to `np.int32` arrays, produces the 9-key output dict.
- `_create_standalone_example()` (line 331): wraps a single document with `segment_ids=1`, `position_ids=1..N`.
- `create_packed_attention_masks()` (line 547): 4D attention masks preventing cross-document attention.

**Output**: 9 keys (base 5 + packing 4), optionally 11 with merged attention (`merge_indices`, `decoder_mask`).

### 6. PackedMultilingualCRAFTCollator

**File**: `packed_multilingual_craft_collator.py:21-381`
**Parent**: `CRAFTPackedPhase2Collator`

Moves document buffering and packing into the generator level for 100% packing efficiency.

**Does not override `__call__`** -- inherits from parent.

**Overrides `_instantiate_dsets()`**: creates infinite packed generators per source with per-source per-mode buffering. Creates `TokenBalancedIterator` for training and `DirectEvalIterator` for evaluation.

**Key behavior**: each source's generator buffers examples, calls `pack_documents()` internally, and yields only packed sequences. This ensures every yielded example is a packed sequence.

**Metadata added**: `_source`, `_source_name`, `_training_mode` on each yielded example.

### 7. UnifiedCRAFTCollator

**File**: `unified_craft_collator.py:26-854`
**Parent**: `PackedMultilingualCRAFTCollator`

Top of chain. Smart task routing with explicit handler dispatch.

**Overrides `__call__`**:
```python
def __call__(self, examples, cooldown_phase=False, bucket_idx=None,
             tokenizer=None, **kwargs)
```

**Routing logic**:
1. If already tokenized (has `input_ids` key): pass through to parent (avoids re-entry during packing)
2. If untokenized: detect task type via `_detect_task_type()`, route to handler

**Task handlers** (registered in `self.task_handlers`):
- `unsupervised_pretraining`: delegates to Phase2MixedCollator (denoising/continuation)
- `ocr_correction`: noisy->clean pairs
- `temporal_classification`: text->year
- `temporal_continuation`: year-aware continuation
- `sts`: semantic textual similarity
- `translation`: bidirectional ko<->en
- `transcription`: bidirectional hanja<->hangul
- `topic_classification`: text->category (YNAT)
- `nli`: natural language inference
- `instruction_following`: instruction-based generation
- `multiple_choice`: question->answer

**Task detection** (`_detect_task_type`, line 60):
- Explicit: `task_type` or `data_type` fields
- Inferred: `corrected` -> ocr, `sentence1+sentence2` -> sts, `source+target` -> translation, `transcribed` -> transcription
- Default: `unsupervised_pretraining`

**`_handle_supervised()`** (line 107): generic supervised handler used by most task handlers.
- Encoder: `[metadata] [input] <encoder_lang_token>`
- Decoder input: `<decoder_lang_token> [label]`
- Labels: `[label] <decoder_lang_token>`
- Returns 5-key dict (unpadded if `padding=False`, for packing later)
