"""
Mid2Mid Data Collator with MIDI corruption strategies.

Extracted from train_mid2mid.py for modular use in mid2mid-rs integration.
"""

import random
from typing import List, Optional
import logging

import torch
import numpy as np

from miditok import MusicTokenizer, TokSequence
from miditok.utils.utils import get_bars_ticks, get_num_notes_per_bar
from miditok.utils.split import split_score_per_ticks, split_score_per_beats
from symusic import Score


logger = logging.getLogger(__name__)


class Mid2MidDataCollator:    
    def __init__(self, encoder_tokenizer: MusicTokenizer, decoder_tokenizer: MusicTokenizer, 
                 encoder_max_length=512, decoder_max_length=1024, pad_to_multiple_of=128,
                 apply_corruption=True, dynamic_padding=False,
                 corruption_mode: str = "hybrid",
                 pooling_mode: str = "beat",
                 p_corrupt_init=0.15,
                 p_corrupt_end=0.45,
                 corruption_anneal_portion=0.7,
                 extra_corruptions: Optional[str] = None,
                 override_corruption_defaults: bool = False,
                 block_curriculum: bool = True,
                 minimal_decoder: bool = False,
                 ):

        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

        if dynamic_padding:
            encoder_max_length = None
            decoder_max_length = None

        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.apply_corruption = apply_corruption
        self.dynamic_padding = dynamic_padding
        self.corruption_mode = corruption_mode.lower()
        self.pooling_mode = pooling_mode
        self.block_curriculum = block_curriculum
        self.minimal_decoder = minimal_decoder

        # delete-prob schedule parameters
        self.p_corrupt_end = p_corrupt_end
        self.p_corrupt_init = p_corrupt_init
        self.anneal_portion = corruption_anneal_portion
        self._global_step = 0     # updated from train loop
        self.total_train_steps = float('inf')

        # decide which corruption modes are legal for this granularity
        beat_modes    = {"beat_drop", "beat_permute", "note_corruption"}
        bar_modes     = {"measure_drop", "aligned_drop", "measure_permute", "element_corruption"}
        all_modes     = {"track_drop", "no_corruption", "bart_span", "token_deletion"}

        if override_corruption_defaults and extra_corruptions:
            # skip defaults entirely, use only extra_corruptions
            self.allowed_modes = set(extra_corruptions.split(","))
        else:
            # use default table based on pooling mode
            default_table = {
                "beat":      beat_modes | all_modes,
                "measure":   bar_modes  | all_modes,
                "sequence":  beat_modes | bar_modes | all_modes,
                "none":      beat_modes | bar_modes | all_modes,
            }
            self.allowed_modes = default_table[self.pooling_mode]
            # add extra corruptions to defaults (if not overriding)
            if extra_corruptions:
                self.allowed_modes |= set(extra_corruptions.split(","))

        self.n_encoder_vocabs = len(encoder_tokenizer.vocab)
        self.encoder_pad_id = [encoder_tokenizer.pad_token_id] * self.n_encoder_vocabs
        self.decoder_pad_id = decoder_tokenizer.pad_token_id
        self.decoder_has_bar_tokens = any('Bar' in token for token in decoder_tokenizer.vocab)

        # structural token ids (BOS / EOS appear in every vocab)
        self.bos_token_id = encoder_tokenizer.vocab[0]["BOS_None"] if \
            encoder_tokenizer.is_multi_voc else encoder_tokenizer["BOS_None"]
        self.eos_token_id = encoder_tokenizer.vocab[0]["EOS_None"] if \
            encoder_tokenizer.is_multi_voc else encoder_tokenizer["EOS_None"]

        # Bar tokens: CP-Word (and other multi-vocab tokenizers) store them in vocab-1
        bar_ids_list = encoder_tokenizer.token_ids_of_type("Bar", vocab_id=1)
        self.bar_ids = list(set(bar_ids_list))
        self.bar_vocab_idx = 1

        # MASK token id (needed for BART-style span masking)
        if encoder_tokenizer.is_multi_voc:
            # for CPWord, all vocabs have MASK_None at the same ID
            self.mask_token_id = encoder_tokenizer.vocab[0]["MASK_None"]
            self.mask_token_ids = [v["MASK_None"] for v in encoder_tokenizer.vocab]
        else:
            self.mask_token_id = encoder_tokenizer["MASK_None"]
            self.mask_token_ids = None

        # position tokens for beat-based corruption (CPWord stores them in vocab-1)
        position_ids_list = encoder_tokenizer.token_ids_of_type("Position", vocab_id=1)
        self.position_ids = set(position_ids_list)
        self.ignore_idx = encoder_tokenizer.vocab[1]['Ignore_None']
        self.velocity_bins = [*set(encoder_tokenizer.token_ids_of_type("Velocity", 3))]
        self.num_pitch_bins = len(set(encoder_tokenizer.token_ids_of_type("Pitch", 2)))

    def __call__(self, batch):
        encoder_tokens = []
        decoder_tokens = []
        decoder_labels = []
        decoder_beat_boundaries = []

        for item in batch:
            # ensure the score can be consumed again later
            score = item['symusic_score'].copy() # symusic uses deepcopy under the hood

            # in the case of multiple programs, encoder potentially will 
            # drop a set of them; initially keep them as several tracks
            try:
                enc = self.encoder_tokenizer(score)
            except KeyError:
                print(f'inspect this piece: {item}')
                encoder_tokens.append(torch.tensor([[v['PAD_None'] for v in self.encoder_tokenizer.vocab]], dtype=torch.long))
                decoder_tokens.append(torch.tensor([self.decoder_tokenizer.pad_token_id], dtype=torch.long))
                decoder_labels.append(torch.tensor([self.decoder_tokenizer.pad_token_id], dtype=torch.long))
                continue
            enc = [t.ids for t in enc]

            # weird bug, but sometimes scores come prepended with several 
            # empty bars: remove those
            # thankfully (?) they are just several subsequent tokens, so we
            # can just remove that first number of items from each list
            maybe_del_idx_ = (np.array(get_num_notes_per_bar(score))!=0).argmax(0).item()
            enc = [t[maybe_del_idx_:] for t in enc]
            clean_score_enc = self.encoder_tokenizer(enc).copy()   # no repeated bars
            clean_score_dec = clean_score_enc.copy()
            if len(clean_score_dec.tracks) > 1:     # shouldn't happen at this point for solo pieces
                merge_tracks(clean_score_dec)

            # now get the clean decoder-tokenized sequence
            dec = self.decoder_tokenizer(clean_score_dec.copy())
            if isinstance(dec, TokSequence): dec = dec.ids
            else: dec = dec[0].ids

            if self.decoder_has_bar_tokens:
                if dec[0] != self.decoder_tokenizer['Bar_None']:
                    if enc[0][self.bar_vocab_idx] == self.bar_ids[0]:   # should always be true, but just in case
                        dec.insert(0, self.decoder_tokenizer['Bar_None'])

            # now get measure boundaries for block sampling with curriculum
            beat_boundaries = []
            try:
                split_scores = split_score_per_beats(clean_score_dec, max_num_beats=1)
                if split_scores:
                    try:
                        beat_boundaries = [len(self.decoder_tokenizer(s).tokens) for s in split_scores]
                    except: # should already be merged
                        assert len(self.decoder_tokenizer(split_scores[0])) == 1, "Score tracks not merged for decoder"
                        beat_boundaries = [len(self.decoder_tokenizer(s)[0].tokens) for s in split_scores]
            except (IndexError, ValueError) as e:
                print(f"\nWARNING: Beat splitting failed for {item['source_file']}")
                print(f"  Segment: bars {item.get('start_bar', '?')}-{item.get('end_bar', '?')}")
                print(f"  Score: {clean_score_dec.end()} ticks, {clean_score_dec.note_num()} notes")
                print(f"  Error: {e}")
                print(f"  Full item: {item}")
                print("-" * 80)

            # fallback: if no beat boundaries, create even chunks
            if not beat_boundaries:
                total_tokens = len(dec)
                num_chunks = max(1, min(4, total_tokens // 20))
                chunk_size = max(1, total_tokens // num_chunks)
                beat_boundaries = []
                for i in range(num_chunks):
                    end = (i + 1) * chunk_size if i < num_chunks - 1 else total_tokens
                    beat_boundaries.append(end - i * chunk_size)

            decoder_beat_boundaries.append(beat_boundaries)

            if self.apply_corruption and self.training:
                enc = self._apply_track_corruption(enc.copy())

            # if the result is empty or badly shaped, fall back
            if not enc or len(enc[0]) == 0 or (
                    isinstance(enc[0][0], int)):    # realistically this shouldn't ever happen, but
                                                    # reconstruct the tokens from the orig. score
                enc = [t.ids for t in self.encoder_tokenizer(clean_score_enc.copy())]

            enc = self._maybe_merge(enc)

            # check if encoder tokens are empty after merging
            if not enc or len(enc) == 0:
                # fallback to pad tokens
                enc = [[self.encoder_pad_id[i] for i in range(self.n_encoder_vocabs)]]

            # add bos/eos tokens for piece boundaries
            if item.get('is_first', False):
                if self.encoder_tokenizer.is_multi_voc:
                    bos_enc = [self.encoder_tokenizer.vocab[i]['BOS_None'] for i in range(self.n_encoder_vocabs)]
                    enc = [bos_enc] + enc
                else:
                    enc = [[self.encoder_tokenizer.vocab['BOS_None']]] + enc
                dec = [self.decoder_tokenizer.vocab['BOS_None']] + dec
            if item.get('is_last', False):
                if self.encoder_tokenizer.is_multi_voc:
                    eos_enc = [self.encoder_tokenizer.vocab[i]['EOS_None'] for i in range(self.n_encoder_vocabs)]
                    enc = enc + [eos_enc]
                else:
                    enc = enc + [[self.encoder_tokenizer.vocab['EOS_None']]]
                dec = dec + [self.decoder_tokenizer.vocab['EOS_None']]

            # maybe truncate
            enc = enc[:self.encoder_max_length]
            dec = dec[:self.decoder_max_length]

            encoder_tokens.append(torch.tensor(enc, dtype=torch.long))
            decoder_tokens.append(torch.tensor(dec[:-1], dtype=torch.long))
            decoder_labels.append(torch.tensor(dec[1:], dtype=torch.long))

        max_enc_len = max(t.shape[0] for t in encoder_tokens)
        max_dec_len = max(t.shape[0] for t in decoder_tokens)

        if self.pad_to_multiple_of:
            max_enc_len = ((max_enc_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
            max_dec_len = ((max_dec_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        if self.encoder_max_length is not None:
            max_enc_len = min(max_enc_len, self.encoder_max_length)
        if self.decoder_max_length is not None:
            max_dec_len = min(max_dec_len, self.decoder_max_length - 1)

        encoder_padded = self._pad_encoder_tokens(encoder_tokens, max_enc_len)

        assert encoder_padded.dim() == 3 and \
            encoder_padded.shape[2] == self.n_encoder_vocabs, \
            f"bad encoder batch shape {encoder_padded.shape}"

        decoder_padded = self._pad_decoder_tokens(decoder_tokens, max_dec_len, self.decoder_pad_id)
        labels_padded = self._pad_decoder_tokens(decoder_labels, max_dec_len, -100)

        return {
            'encoder_input_ids': encoder_padded,
            'decoder_input_ids': decoder_padded,
            'decoder_blocks': decoder_beat_boundaries,
            'labels': labels_padded
        }

    def _maybe_merge(self, tracks) -> List[int]:
        """Merge several tracks post-corruption into one time-ordered sequence.

        The input tracks is a list where every element is a track represented
        as a nested list of encoder tokens.  Each token is already encoded in
        multi-vocabulary (CP-Word) format.  When more than one track is
        provided, we want to interleave them so that the tokens that belong to
        the same bar across tracks are kept together - this keeps rhythmic
        structure roughly aligned even after corruption.

        The strategy is:
        1. Split every track into bars using _split_into_bars.
        2. For the number of bars that all tracks share (the minimum across
           tracks) we concatenate the bar-tokens from every track in turn and
           finish the merged bar with one Bar token (taken from the first
           encountered Bar token for that bar).  This avoids having multiple
           consecutive Bar markers while still marking the end of the bar.
        3. Any left-over bars that only some tracks contain are appended in the
           original order (simply concatenated) - this is a best-effort fall-
           back and only affects unequal track lengths.

        If tracks already contains a single element, that element is
        returned unchanged.
        """
        if len(tracks) <= 1:
            return tracks[0] if len(tracks) == 1 else []

        # pre-split all tracks into their constituent bars
        bars_per_track = [self._split_into_bars(t, preserve_order=True) for t in tracks]

        # determine how many bars are common to every track so we can align on them
        n_bars_common = min(len(bars) for bars in bars_per_track)

        merged: List = []
        def _is_bar(tok):
            if self.bar_vocab_idx is not None:
                return isinstance(tok, (list, tuple)) and tok[self.bar_vocab_idx] in self.bar_ids
            return tok in self.bar_ids
        # interleave bars that are shared by all tracks
        for bar_idx in range(n_bars_common):
            non_bar_tokens = []
            bar_token = None  # we will keep a /single/ bar token for this bar

            for track_bars in bars_per_track:
                bar_tokens = track_bars[bar_idx]
                for tok in bar_tokens:
                    if _is_bar(tok):
                        # remember first encountered bar token so we add only one later
                        if bar_token is None:
                            bar_token = tok
                        # skip adding duplicate bar tokens
                        continue
                    non_bar_tokens.append(tok)

            # append collected tokens and exactly one bar token to terminate the bar
            merged.extend(non_bar_tokens)
            if bar_token is not None:
                merged.append(bar_token)

        # append any remaining bars from tracks that are longer than n_bars_common
        for track_bars in bars_per_track:
            for bar_idx in range(n_bars_common, len(track_bars)):
                merged.extend(track_bars[bar_idx])

        return merged

    def _ensure_non_empty_tracks(self, tracks: List[List[int]], original_tracks: List[List[int]]) -> List[List[int]]:
        """Ensure at least one track has content, otherwise return a minimal valid sequence."""
        if all(not track for track in tracks):
            # find the first non-empty original track and use its first token
            for orig_track in original_tracks:
                if orig_track:
                    return [[orig_track[0]]]
            # if all original tracks were empty, return as-is
            return tracks
        return tracks

    def _apply_bert_mask_ratio(self, masked_positions: List[int], tokens: List, mask_prob: float = 0.15) -> List:
        """Apply BERT-style masking ratios: 80% mask, 10% random, 10% unchanged.
        
        Args:
            masked_positions: Indices of tokens to potentially mask
            tokens: Full list of tokens
            mask_prob: Probability of masking (default 0.15)
        
        Returns:
            Modified tokens list with BERT-style masking applied
        """
        # first, determine which positions will actually be masked based on mask_prob
        actual_masked = []
        for pos in masked_positions:
            if random.random() < mask_prob:
                actual_masked.append(pos)

        if not actual_masked:
            return tokens

        # apply bert ratios to the selected positions
        result_tokens = tokens.copy() if isinstance(tokens[0], list) else list(tokens)

        for idx in actual_masked:
            rand = random.random()
            if rand < 0.8:  # 80% mask token
                if self.encoder_tokenizer.is_multi_voc and isinstance(result_tokens[idx], list):
                    result_tokens[idx] = self.mask_token_ids.copy()
                else:
                    result_tokens[idx] = self.mask_token_id
            elif rand < 0.9:  # 10% random token
                if self.encoder_tokenizer.is_multi_voc and isinstance(result_tokens[idx], list):
                    # random token for each vocabulary
                    random_token = []
                    for v_idx, vocab in enumerate(self.encoder_tokenizer.vocab):
                        # avoid special tokens when selecting random
                        regular_tokens = [tid for tok, tid in vocab.items() 
                                        if not any(special in tok for special in ['PAD', 'BOS', 'EOS', 'MASK'])]
                        if regular_tokens:
                            random_token.append(random.choice(regular_tokens))
                        else:
                            random_token.append(result_tokens[idx][v_idx])  # keep original if no regular tokens
                    result_tokens[idx] = random_token
                else:
                    # single vocab random token
                    vocab = self.encoder_tokenizer.vocab
                    regular_tokens = [tid for tok, tid in vocab.items() 
                                    if not any(special in tok for special in ['PAD', 'BOS', 'EOS', 'MASK'])]
                    if regular_tokens:
                        result_tokens[idx] = random.choice(regular_tokens)
            # else: 10% keep unchanged (do nothing)

        return result_tokens

    def _apply_track_corruption(self, tracks: List[List[int]]) -> List[List[int]]:
        """Apply track-aware corruption including BART-style methods."""

        if len(tracks) == 0: return tracks
        if not self.apply_corruption:
            return tracks

        # ensure we have at least one non-empty track to work with
        non_empty_tracks = [t for t in tracks if t]
        if not non_empty_tracks:
            return tracks  # all tracks are already empty, return as-is

        # keep a copy of original tracks for safeguard
        original_tracks = [track[:] for track in tracks]

        progress = min(1.0, self._global_step /
                            max(self.anneal_portion * self.total_train_steps, 1e-10))
        corrupt_prob = self.p_corrupt_init + progress * (self.p_corrupt_end - self.p_corrupt_init)

        mode = self.corruption_mode

        # no corruption - just return the tracks as-is, pure translation
        if mode == "no_corruption":
            return tracks

        # track drop - drop random tracks
        elif mode == "track_drop":
            # ensure at least one non-empty track survives
            non_empty = [t for t in tracks if t]          # filter out []
            if not non_empty:                             # everything empty – skip
                return tracks
            # use fixed 0.1 probability for track drop (acts like instrument dropout)
            # useful for "fill the missing instrument" tasks
            track_drop_prob = 0.1
            n_keep = max(1, int(len(non_empty) * (1 - track_drop_prob)))
            keep_idx = random.sample(range(len(non_empty)), n_keep)
            return [TokSequence(ids=non_empty[i]).ids for i in keep_idx]

        # measure drop - drop measures within each track
        elif mode == "measure_drop":
            corrupted_tracks = []
            for track in tracks:
                bars = self._split_into_bars(track)
                if len(bars) <= 1:
                    corrupted_tracks.append(track)
                    continue
                n_keep = max(1, int(len(bars) * (1 - corrupt_prob)))
                keep_indices = sorted(random.sample(range(len(bars)), n_keep))
                kept_bars = [TokSequence(ids=bars[i]) for i in keep_indices]
                corrupted_tracks.append(sum(kept_bars).ids)
            return self._ensure_non_empty_tracks(corrupted_tracks, original_tracks)

        # aligned drop - drop same measures across all tracks
        elif mode == "aligned_drop":
            # use minimum bar count across tracks to stay in bounds
            n_bars_common = min(len(self._split_into_bars(t)) for t in tracks)
            if n_bars_common == 0:
                return tracks  # nothing to drop
            n_drop = max(1, int(n_bars_common * corrupt_prob))
            drop_indices = set(random.sample(range(n_bars_common), n_drop))

            corrupted_tracks = []
            for track in tracks:
                bars = [TokSequence(ids=b) for b in self._split_into_bars(track)]
                # handle tracks that might be shorter than common length
                bars_in_scope = bars[:n_bars_common]
                kept_bars = [bars_in_scope[i] for i in range(n_bars_common) if i not in drop_indices]
                if not kept_bars:
                    kept_bars = [bars_in_scope[0]]  # guarantee at least one
                # append any bars beyond common length untouched
                kept_bars.extend(bars[n_bars_common:])
                corrupted_tracks.append(sum(kept_bars).ids)
            return self._ensure_non_empty_tracks(corrupted_tracks, original_tracks)

        # measure permute - shuffle measure order using phrases (4-8 measure blocks)
        elif mode == "measure_permute":
            corrupted_tracks = []

            for track in tracks:
                bars = self._split_into_bars(track, preserve_order=True)
                n_bars = len(bars)

                # skip if too few bars
                if n_bars <= 4:
                    corrupted_tracks.append(track)
                    continue

                # determine phrase size based on total bars
                # aim for 4-8 bar phrases, with at least 2 phrases
                if n_bars <= 8:
                    phrase_size = 4
                elif n_bars <= 16:
                    phrase_size = min(8, n_bars // 2)  # ensure at least 2 phrases
                else:
                    # for longer sequences, use 6-8 bar phrases
                    phrase_size = min(8, max(6, n_bars // 4))

                # group bars into phrases
                phrases = []
                for i in range(0, n_bars, phrase_size):
                    phrase_bars = bars[i:i + phrase_size]
                    if phrase_bars:  # ensure non-empty
                        phrases.append(phrase_bars)

                # only shuffle if we have at least 2 phrases
                if len(phrases) < 2:
                    corrupted_tracks.append(track)
                    continue

                # shuffle phrases, not individual bars
                shuffled_phrase_indices = list(range(len(phrases)))
                random.shuffle(shuffled_phrase_indices)

                # reconstruct track from shuffled phrases
                shuffled_bars = []
                for phrase_idx in shuffled_phrase_indices:
                    shuffled_bars.extend(phrases[phrase_idx])

                # convert back to token sequence
                shuffled_track = sum([TokSequence(ids=bar) for bar in shuffled_bars]).ids
                corrupted_tracks.append(shuffled_track)
            return self._ensure_non_empty_tracks(corrupted_tracks, original_tracks)

        # beat drop - drop random beats within measures
        elif mode == "beat_drop":
            corrupted_tracks = []

            for track in tracks:
                # beat positions in CPword are in vocab[1]
                track_tokens = []
                for i, tok in enumerate(track):
                    if isinstance(tok, (list, tuple)):
                        if tok[1] in self.position_ids:
                            if random.random() > corrupt_prob:
                                track_tokens.append(tok)
                        else:
                            track_tokens.append(tok)
                    else:
                        track_tokens.append(tok)

                if not track_tokens and track:  # only add first token if track is non-empty
                    track_tokens = [track[0]]  # keep at least one token
                corrupted_tracks.append(track_tokens)
            return self._ensure_non_empty_tracks(corrupted_tracks, original_tracks)

        # beat permute - shuffle beat order within measures
        elif mode == "beat_permute":
            corrupted_tracks = []

            for track in tracks:
                # IMPORTANT: use preserve_order=True to avoid re-sorting
                bars = self._split_into_bars(track, preserve_order=True)
                shuffled_bars = []

                # select which measures to corrupt based on corrupt_prob
                num_bars_to_corrupt = int(len(bars) * corrupt_prob)
                bars_to_corrupt = set(random.sample(range(len(bars)), min(num_bars_to_corrupt, len(bars))))

                for bar_idx, bar in enumerate(bars):
                    # only shuffle beats in selected measures
                    if bar_idx not in bars_to_corrupt:
                        shuffled_bars.append(bar)
                        continue
                        
                    beat_indices = []
                    for i, tok in enumerate(bar):
                        # check for position tokens - handle both single and multi-vocab
                        if self.encoder_tokenizer.is_multi_voc:
                            if isinstance(tok, list) and len(tok) > 1 and tok[1] in self.position_ids:
                                beat_indices.append(i)
                        else:
                            # for single vocab tokenizers, position tokens are just IDs
                            if tok in self.position_ids:
                                beat_indices.append(i)

                    if len(beat_indices) <= 1:
                        shuffled_bars.append(bar)
                        continue

                    # split bar into beats and shuffle
                    beats = []
                    for i in range(len(beat_indices)):
                        start = beat_indices[i]
                        end = beat_indices[i+1] if i+1 < len(beat_indices) else len(bar)
                        beats.append(bar[start:end])

                    random.shuffle(beats)
                    shuffled_bar = [tok for beat in beats for tok in beat]
                    shuffled_bars.append(shuffled_bar)

                corrupted_tracks.append([tok for bar in shuffled_bars for tok in bar])
            return self._ensure_non_empty_tracks(corrupted_tracks, original_tracks)

        # note corruption - modify individual notes with bert-style masking
        elif mode == "note_corruption":
            corrupted_tracks = []
            corruption_type = random.choice(["pitch_shift", "velocity_random", "note_drop"])

            for track in tracks:
                # identify note positions
                note_positions = []
                for i, tok in enumerate(track):
                    if isinstance(tok, list) and tok[2] != self.ignore_idx:
                        note_positions.append(i)
                
                if corruption_type == "note_drop" and note_positions:
                    # apply bert-style masking to notes
                    corrupted = self._apply_bert_mask_ratio(note_positions, track, mask_prob=corrupt_prob)
                else:
                    # pitch shift or velocity randomization
                    corrupted = []
                    for tok in track:
                        if isinstance(tok, list):
                            if tok[2] != self.ignore_idx:  # it's a note
                                if corruption_type == "pitch_shift":
                                    # shift pitch by 1-3 semitones
                                    new_tok = list(tok)
                                    shift = random.randint(-3, 3)
                                    new_pitch = max(0, min(self.num_pitch_bins - 1, tok[2] + shift))
                                    new_tok[2] = new_pitch
                                    corrupted.append(new_tok)
                                elif corruption_type == "velocity_random":
                                    # randomize velocity if present
                                    new_tok = list(tok)
                                    if tok[3] != self.ignore_idx:  # has velocity
                                        new_tok[3] = random.choice(self.velocity_bins)
                                    corrupted.append(new_tok)
                            else:
                                corrupted.append(tok)
                        else:
                            corrupted.append(tok)

                if not corrupted and track:
                    corrupted = [track[0]]
                corrupted_tracks.append(corrupted)
            
            return self._ensure_non_empty_tracks(corrupted_tracks, original_tracks)

        # bart span - mask spans respecting pooling mode
        elif mode == "bart_span":
            mask_token_id = self.mask_token_id
            corrupted_tracks = []

            for track in tracks:
                # prepare units and parameters based on pooling mode
                if self.pooling_mode == "measure":
                    units = self._split_into_bars(track)
                    span_prob = corrupt_prob
                    is_bar_based = True
                elif self.pooling_mode == "beat":
                    # TODO: implement beat-level masking if needed
                    corrupted_tracks.append(track)
                    continue
                else:  # no pooling - token-level masking
                    units = track
                    span_prob = corrupt_prob
                    is_bar_based = False

                if len(units) <= 1:
                    corrupted_tracks.append(units[0] if is_bar_based else units)
                    continue

                # collect spans to mask
                masked_spans = []
                i = 0
                while i < len(units):
                    if random.random() < span_prob:
                        span_len = min(np.random.poisson(3), len(units) - i)
                        if span_len > 0:
                            masked_spans.append((i, i + span_len))
                            i += span_len
                            continue
                    i += 1

                # apply BART-style masking to the selected spans
                result_units = []

                for idx, unit in enumerate(units):
                    # check if this unit is part of a masked span
                    in_span = False
                    span_start = False

                    for start, end in masked_spans:
                        if start <= idx < end:
                            in_span = True
                            span_start = (idx == start)
                            break

                    if in_span:
                        # only add mask token for the first unit in the span
                        if span_start:
                            if self.encoder_tokenizer.is_multi_voc:
                                mask_unit = self.mask_token_ids  # already a list!
                            else:
                                mask_unit = mask_token_id
                            result_units.append([mask_unit] if is_bar_based else mask_unit)
                        # skip other units in the span
                    else:
                        # not part of any masked span, keep as is
                        result_units.append(unit)

                # reconstruct track
                if is_bar_based:
                    corrupted_track = [tok for bar in result_units for tok in bar]
                else:
                    corrupted_track = result_units

                if not corrupted_track and track:
                    corrupted_track = [track[0]]
                corrupted_tracks.append(corrupted_track)

            return self._ensure_non_empty_tracks(corrupted_tracks, original_tracks)

        elif mode == "element_corruption":
            # element-level masking for cpword tokens (PianoBart key innovation)
            corrupted_tracks = []
            corruption_type = random.choice(["single_mask", "bar_mask", "n-bar_mask"])

            for track in tracks:
                if corruption_type == "single_mask":
                    # mask random individual elements within tokens with bert ratios
                    corrupted_track = []
                    for token in track:
                        if isinstance(token, list) and len(token) == self.n_encoder_vocabs:
                            new_token = token.copy()
                            # mask 1-3 random elements per token with 15% probability
                            if random.random() < 0.15:
                                # find non-ignore elements to ensure meaningful masking
                                non_ignore_indices = [i for i in range(self.n_encoder_vocabs) 
                                                    if token[i] != self.ignore_idx]

                                if non_ignore_indices:
                                    # mask at least one non-ignore element
                                    num_to_mask = random.randint(1, min(3, len(non_ignore_indices)))
                                    mask_indices = random.sample(non_ignore_indices, num_to_mask)

                                    # apply bert-style masking to selected indices
                                    for idx in mask_indices:
                                        rand = random.random()
                                        if rand < 0.8:  # 80% mask
                                            new_token[idx] = self.mask_token_ids[idx]
                                        elif rand < 0.9:  # 10% random
                                            vocab = self.encoder_tokenizer.vocab[idx]
                                            regular_tokens = [tid for t, tid in vocab.items() 
                                                            if not any(s in t for s in ['PAD', 'BOS', 'EOS', 'MASK', 'Ignore'])]
                                            if regular_tokens:
                                                new_token[idx] = random.choice(regular_tokens)
                                        # else: 10% unchanged

                            corrupted_track.append(new_token)
                        else:
                            corrupted_track.append(token)

                elif corruption_type == "bar_mask":
                    # mask aligned elements across a bar with bert ratios
                    bars = self._split_into_bars(track)
                    corrupted_bars = []

                    for bar in bars:
                        # choose 1-2 vocab indices to mask across the entire bar
                        if random.random() < corrupt_prob and len(bar) > 0:
                            # find vocabularies that have meaningful content in this bar
                            vocab_has_content = [False] * self.n_encoder_vocabs
                            for token in bar:
                                if isinstance(token, list):
                                    for v_idx in range(self.n_encoder_vocabs):
                                        if token[v_idx] != self.ignore_idx:
                                            vocab_has_content[v_idx] = True

                            meaningful_vocabs = [i for i, has_content in enumerate(vocab_has_content) if has_content]
                            if meaningful_vocabs:
                                vocab_indices_to_mask = random.sample(
                                    meaningful_vocabs, 
                                    random.randint(1, min(2, len(meaningful_vocabs)))
                                )

                                corrupted_bar = []
                                for token in bar:
                                    if isinstance(token, list) and len(token) == self.n_encoder_vocabs:
                                        new_token = token.copy()
                                        for idx in vocab_indices_to_mask:
                                            rand = random.random()
                                            if rand < 0.8:  # 80% mask
                                                new_token[idx] = self.mask_token_ids[idx]
                                            elif rand < 0.9:  # 10% random
                                                vocab = self.encoder_tokenizer.vocab[idx]
                                                regular_tokens = [tid for t, tid in vocab.items() 
                                                                if not any(s in t for s in ['PAD', 'BOS', 'EOS', 'MASK', 'Ignore'])]
                                                if regular_tokens:
                                                    new_token[idx] = random.choice(regular_tokens)
                                            # else: 10% unchanged
                                        corrupted_bar.append(new_token)
                                    else:
                                        corrupted_bar.append(token)
                                corrupted_bars.append(corrupted_bar)
                            else:
                                corrupted_bars.append(bar)
                        else:
                            corrupted_bars.append(bar)

                    corrupted_track = [token for bar in corrupted_bars for token in bar]

                elif corruption_type == "n-bar_mask":
                    # mask aligned elements across multiple bars (up to 2)
                    bars = self._split_into_bars(track)
                    corrupted_bars = []

                    i = 0
                    while i < len(bars):
                        if random.random() < corrupt_prob and i + 1 < len(bars):
                            # mask 1-2 consecutive bars
                            n_bars = min(2, len(bars) - i)
                            vocab_indices_to_mask = random.sample(
                                range(self.n_encoder_vocabs), 
                                random.randint(1, min(2, self.n_encoder_vocabs))
                            )

                            for j in range(n_bars):
                                corrupted_bar = []
                                for token in bars[i + j]:
                                    if isinstance(token, list) and len(token) == self.n_encoder_vocabs:
                                        new_token = token.copy()
                                        for idx in vocab_indices_to_mask:
                                            new_token[idx] = self.mask_token_ids[idx]
                                        corrupted_bar.append(new_token)
                                    else:
                                        corrupted_bar.append(token)
                                corrupted_bars.append(corrupted_bar)
                            i += n_bars
                        else:
                            corrupted_bars.append(bars[i])
                            i += 1

                    corrupted_track = [token for bar in corrupted_bars for token in bar]

                # ensure we have at least one token
                if not corrupted_track and track:
                    corrupted_track = [track[0]]

                corrupted_tracks.append(corrupted_track)

            return self._ensure_non_empty_tracks(corrupted_tracks, original_tracks)

        # token deletion - delete random tokens
        elif mode == "token_deletion":
            corrupted_tracks = []

            for track in tracks:
                if not track:
                    corrupted_tracks.append(track)
                    continue

                corrupted_track = []
                for token in track:
                    if random.random() > corrupt_prob:
                        corrupted_track.append(token)

                # ensure at least some tokens remain
                if not corrupted_track and track:
                    # keep at least 20% of original tokens
                    num_keep = max(1, int(len(track) * 0.2))
                    keep_indices = sorted(random.sample(range(len(track)), num_keep))
                    corrupted_track = [track[i] for i in keep_indices]

                corrupted_tracks.append(corrupted_track)

            return self._ensure_non_empty_tracks(corrupted_tracks, original_tracks)


    def _pad_encoder_tokens(self, tokens_list, target_len):
        """Pad CP-Word tokens ([seq, n_vocab]).  
        If a flat 1-D list slips through, expand it to shape [seq, n_vocab]."""
        padded = []
        for tokens in tokens_list:
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(1).repeat(1, self.n_encoder_vocabs)
            if tokens.shape[0] < target_len:
                pad_rows = target_len - tokens.shape[0]
                padding = torch.tensor(
                    [self.encoder_pad_id] * pad_rows, dtype=torch.long)
                tokens = torch.cat([tokens, padding], dim=0)
            else:
                tokens = tokens[:target_len]
            assert tokens.dim() == 2 and tokens.shape[1] == self.n_encoder_vocabs, \
                   f"encoder token shape after pad: {tokens.shape}"
            padded.append(tokens)

        return torch.stack(padded)

    def _pad_decoder_tokens(self, tokens_list, target_len, pad_value):
        padded = []
        for tokens in tokens_list:
            if tokens.shape[0] < target_len:
                padding = torch.full((target_len - tokens.shape[0],), pad_value, dtype=torch.long)
                tokens = torch.cat([tokens, padding], dim=0)
            elif tokens.shape[0] > target_len:
                tokens = tokens[:target_len]
            padded.append(tokens)
        return torch.stack(padded)

    @property
    def training(self):
        return getattr(self, '_training', True)

    def set_training(self, mode: bool):
        """Set training mode for input corruption"""
        self._training = mode

    def _split_into_bars(self, track: List[int], preserve_order: bool = False):
        """Split a `List[int]` MIDI track over measures. 
        
        If `preserve_order` is False, the track will be converted to a `symusic.Score`,
        a class that, when constructed, internally sorts all MIDI events in temporal order. 
        In the case of measure or beat permutation,  it is recommended that `preserve_order` 
        be set to True to retain the intended corruption."""
        if preserve_order:
            segments, current = [], []

            def is_bar(tok):
                if self.bar_vocab_idx is not None:
                    return isinstance(tok, (list, tuple)) and tok[self.bar_vocab_idx] in self.bar_ids
                return tok in self.bar_ids

            for tok in track:
                current.append(tok)
                if is_bar(tok):
                    seg = current.copy()
                    segments.append(seg)
                    current.clear()

            if current:  # trailing tokens without final bar
                segments.append(current.copy())

            return segments

        as_score = self.encoder_tokenizer.decode([track.copy()])
        bars_ticks = get_bars_ticks(as_score)
        if len(bars_ticks) == 1:
            split_score = [as_score]
            as_bars = [t.ids for ss in split_score for t in self.encoder_tokenizer(ss)]
            return as_bars
        split_score = split_score_per_ticks(as_score, bars_ticks)
        as_bars = [t.ids for ss in split_score for t in self.encoder_tokenizer(ss)]

        return as_bars

