# MIDI Integration Interface

Contract that the mid2mid-rs MIDI pipeline must satisfy to integrate with the CRAFT training infrastructure.

## 1. Required Output Dict

The MIDI collator must produce output dicts compatible with `UnifiedCRAFTCollator`'s packed format.

### Unpacked (5 keys)

Used by `_handle_supervised()` in `unified_craft_collator.py:107-254`:

```python
{
    'input_ids':             np.array(int32, shape=(max_length,)),
    'decoder_input_ids':     np.array(int32, shape=(max_length,)),
    'labels':                np.array(int32, shape=(max_length,)),
    'attention_mask':        np.array(int32, shape=(max_length,)),
    'decoder_attention_mask': np.array(int32, shape=(max_length,)),
}
```

- `labels` uses `-100` for padding positions (ignored by cross-entropy loss)
- `attention_mask` and `decoder_attention_mask`: 1 for real tokens, 0 for padding

### Packed (9 keys)

Produced by `_finalize_pack()` in `craft_packed_phase2_collator.py:275-329`:

```python
{
    'input_ids':             np.array(int32, shape=(max_length,)),
    'decoder_input_ids':     np.array(int32, shape=(max_length,)),
    'labels':                np.array(int32, shape=(max_length,)),
    'attention_mask':        np.array(int32, shape=(max_length,)),
    'decoder_attention_mask': np.array(int32, shape=(max_length,)),
    'segment_ids':           np.array(int32, shape=(max_length,)),
    'position_ids':          np.array(int32, shape=(max_length,)),
    'decoder_segment_ids':   np.array(int32, shape=(max_length,)),
    'decoder_position_ids':  np.array(int32, shape=(max_length,)),
}
```

### Batched

All keys become `(batch_size, max_length)` when stacked.

## 2. Sentinel Token Convention

Implemented in `BARTCollator._token_based_sentinel_masking` (`h2hcollator.py:1358-1499`).

- Each masked span gets a unique `<extra_id_N>` token
- Sentinel indices **increase** per span: first span = `<extra_id_0>`, second = `<extra_id_1>`, etc.
- 256 sentinels available (`tokenizer.NUM_SENTINEL_TOKENS`)
- Access via `tokenizer.get_sentinel_token_id(idx)`

### Encoder format

Original tokens with each masked span replaced by its sentinel:

```
[tokens] <extra_id_0> [tokens] <extra_id_1> [tokens] <lang_token>
```

### Decoder format

Sentinels followed by the masked span content:

```
<lang_token> <extra_id_0> [span_0_tokens] <extra_id_1> [span_1_tokens] ...
```

### Chaining

`start_sentinel_idx` parameter allows chaining sentinel indices across multiple calls. Returns `(encoder_ids, decoder_ids, next_sentinel_idx)`.

## 3. Packing Format

Implemented in `CRAFTPackedPhase2Collator.pack_documents` (`craft_packed_phase2_collator.py:68-204`).

### Segment IDs

- `0` = padding
- `1` = first document
- `2` = second document
- `N` = Nth document

### Position IDs

- `0` = padding
- `1, 2, 3, ...` = sequential positions within each document
- **Resets per segment** (each document starts at position 1)

### Packing algorithm

First-fit-decreasing bin packing:
1. Sort documents by length (longest first)
2. For each document, try to fit into existing bins
3. If no bin has room, create a new bin
4. Pad all bins to `max_length`

### Attention masking

Cross-document attention is prevented: decoder segment N only attends to encoder segment N. Implemented in `create_packed_attention_masks` (`craft_packed_phase2_collator.py:547-593`).

## 4. DataSourceConfig

Defined in `dynamic_data_loader.py:62-140`.

To add a MIDI data source, the `data_type` Literal at line 70 must be extended to include `'midi_pretraining'`.

### Required fields for a MIDI source

```python
DataSourceConfig(
    name='midi_kunstderfuge',
    gcs_pattern='gs://bucket/midi/*.parquet',
    weight=0.10,
    data_type='midi_pretraining',
    language='music',
    text_field='midi_tokens',
    metadata_field='filename',
)
```

### Optional fields

- `eval_pattern`: GCS path for validation data
- `test_pattern`: GCS path for test data
- `has_stratified_split`: whether pre-split train/eval/test exists

### Registration

Add the `DataSourceConfig` to the list returned by `create_craft_data_sources()` in `dynamic_data_loader.py`.

## 5. Adding a MIDI Handler

### Step 1: Register handler

In `UnifiedCRAFTCollator.__init__` (`unified_craft_collator.py:46-58`), add:

```python
self.task_handlers['midi_pretraining'] = self._handle_midi_pretraining
```

### Step 2: Task detection

`_detect_task_type` (`unified_craft_collator.py:60-101`) will route to the handler when:
- `example['task_type'] == 'midi_pretraining'`, or
- `example['data_type'] == 'midi_pretraining'`

### Step 3: Handler signature

```python
def _handle_midi_pretraining(self, examples, cooldown_phase=False,
                              bucket_idx=None, tokenizer=None,
                              return_source=False, padding=True, **kwargs):
```

The handler must return a dict with at least the 5 base keys (unpacked). The packing layer (`pack_documents`) adds segment/position IDs automatically when `enable_packing=True`.

### Step 4: Handler pattern

Most handlers in the chain follow this pattern:
1. Extract/transform data from `examples`
2. Build a normalized dict with `original_text`, `labels`, `metadata`, `language`, `_training_mode`
3. Delegate to `_handle_supervised()` for tokenization and formatting

For MIDI, the handler may need custom tokenization (not text-based), so it could directly construct the 5-key output dict instead of delegating to `_handle_supervised`.

## 6. Task Prompts

Defined in `task_prompts.py`.

### Adding MIDI prompts

Add a `'midi_pretraining'` key to the `TASK_PROMPTS` dict:

```python
TASK_PROMPTS['midi_pretraining'] = {
    'en': [
        "Reconstruct the corrupted MIDI sequence:",
        "Complete the following musical passage:",
        "Restore the missing notes in this score:",
    ],
}
```

Also add `'midi_pretraining'` to the Literal type in `sample_task_prompt()`.

### Prompt injection

Prompts are injected via `example['metadata']` field before tokenization. The prompt text becomes part of the encoder input: `[prompt] [input_tokens] <lang_token>`.

## 7. Data Flow

```
dynamic_data_loader.py          create_craft_data_sources()
       |                        returns List[DataSourceConfig]
       v
packed_multilingual_craft_collator.py   _instantiate_dsets()
       |                                creates per-source generators
       |                                buffers examples, calls pack_documents()
       v
unified_craft_collator.py       __call__()
       |                        detects task_type, routes to handler
       |                        handler returns 5-key dict (unpadded)
       v
craft_packed_phase2_collator.py pack_documents()
       |                        adds segment_ids, position_ids
       |                        pads to max_length
       v
train_craft_multilingual.py     training loop
                                converts np arrays to JAX arrays
                                feeds to model
```

### For MIDI specifically

The mid2mid-rs pipeline replaces the `symusic` + MidiTok stack. It must:

1. Parse MIDI files into token sequences
2. Apply corruption strategies (span masking with sentinels, measure dropping, etc.)
3. Produce output dicts with the 5 base keys
4. Let the packing layer handle segment_ids/position_ids

The MIDI collator output flows through the same packing and training pipeline as text data. The key constraint is matching the output dict format exactly.
