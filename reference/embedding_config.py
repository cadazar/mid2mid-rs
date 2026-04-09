"""Variable embedding dimensions for Mid2Mid CPWord encoder.

Based on vocabulary analysis:
- Total vocabularies: 10
- Total tokens: 3556
- Highly imbalanced sizes (6 to 2310)
"""

# vocabulary order from analysis
VOCAB_TYPES = [
    "Family",     # 0: 6 tokens (Note vs Metric)
    "Position",   # 1: 2310 tokens (2304 positions!)
    "Pitch",      # 2: 156 tokens (89 pitch + 62 drums)
    "Velocity",   # 3: 37 tokens
    "Duration",   # 4: 389 tokens (rhythmic info)
    "Program",    # 5: 134 tokens (instruments)
    "Chord",      # 6: 19 tokens
    "Rest",       # 7: 389 tokens (silence durations)
    "Tempo",      # 8: 37 tokens
    "TimeSig",    # 9: 79 tokens
]

VOCAB_SIZES = [6, 2310, 156, 37, 389, 134, 19, 389, 37, 79]

# cpword original (d_model=512):
# [tempo:128, chord:256, barbeat:64, type:32, pitch:512, duration:128, velocity:128]
# total concatenated: 1248 dims → projected to 512
# proportions: [0.25, 0.5, 0.125, 0.0625, 1.0, 0.25, 0.25] relative to d_model

# our d_model = 384, following cpword's proportions:
CPWORD_PROPORTIONAL = [
    24,   # Family (type): 384 * 0.0625 = 24
    48,   # Position (barbeat): 384 * 0.125 = 48  
    384,  # Pitch: 384 * 1.0 = 384 (full d_model!)
    96,   # Velocity: 384 * 0.25 = 96
    96,   # Duration: 384 * 0.25 = 96
    64,   # Program: (estimated ~0.167)
    192,  # Chord: 384 * 0.5 = 192
    96,   # Rest: same as duration
    96,   # Tempo: 384 * 0.25 = 96
    48,   # TimeSig: same as position
]

# more reasonable (pitch doesn't need full d_model):
PROPOSED_DIMS = [
    16,   # Family: small but important
    64,   # Position: even with 2000+ tokens, positions are simple
    192,  # Pitch: most important, but 192 is plenty
    32,   # Velocity: not used in minimal version
    96,   # Duration: critical for rhythm
    64,   # Program: instrument identity
    96,   # Chord: harmony info (cpword gives this 0.5x!)
    64,   # Rest: simpler than duration
    48,   # Tempo: global info
    32,   # TimeSig: structural markers
]

# if we remove timesig (9 vocabs, 95 positions):
DIMS_NO_TIMESIG = [
    16,   # Family
    48,   # Position (only 95 tokens now!)
    192,  # Pitch
    32,   # Velocity
    96,   # Duration
    64,   # Program
    96,   # Chord
    64,   # Rest
    48,   # Tempo
]

def get_total_dims(dims):
    """Calculate total embedding dimensions."""
    return sum(dims)

def get_dim_ratios(dims):
    """Get percentage of total dims for each vocab."""
    total = sum(dims)
    return [d/total * 100 for d in dims]

# analysis
print("Embedding Dimension Analysis:")
print("="*60)

configs = [
    ("CPWord Proportional", CPWORD_PROPORTIONAL),
    ("Proposed (Reasonable)", PROPOSED_DIMS),
    ("No TimeSig (9 vocabs)", DIMS_NO_TIMESIG),
]

for name, dims in configs:
    total = get_total_dims(dims)
    print(f"\n{name}:")
    print(f"  Total concatenated dims: {total}")
    print(f"  vs d_model: {384}")
    print("  Distribution:")
    ratios = get_dim_ratios(dims)
    for i, (vtype, dim, ratio) in enumerate(zip(VOCAB_TYPES, dims, ratios)):
        tokens_per_dim = VOCAB_SIZES[i] / dim
        print(f"    {vtype:10} {dim:3}d ({ratio:4.1f}%) - {tokens_per_dim:.1f} tokens/dim")

# recommendation for fusion methods
print("\nFusion Method Recommendations:")
print("1. Weighted sum with learned importance (simple, effective)")
print("2. Gated fusion with highway connections (preserve important info)")
print("3. Cross-vocab attention (let vocabs interact)")
print("4. Hierarchical: Family first, then others conditional")