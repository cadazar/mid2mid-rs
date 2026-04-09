"""Configuration for Mid2Mid models."""

from dataclasses import dataclass
from typing import Optional, Literal, List


@dataclass
class Mid2MidConfig:
    """Configuration for MIDI-to-MIDI translation models."""
    
    # Model architecture
    d_model: int = 512
    encoder_nlayer: int = 6
    decoder_nlayer: int = 6
    d_ff: int = 2048
    
    # Tokenizer info
    encoder_vocab_count: int = 10  # CPWord has 10 vocabularies
    embedding_dims_per_vocab: Optional[List[int]] = None  # variable dims per vocabulary
    use_vocab_weighting: bool = False  # learned importance weights
    
    # Attention configuration
    attention_mechanism: Literal["mha", "craft", "gla"] = "craft"
    num_heads: Optional[int] = 8
    d_prime: int = 256  # for CRAFT/GLA
    attn_window: int = 32  # local window for CRAFT
    gated_cross_attention: bool = False
    
    # GLA-specific settings
    gla_mode: str = "chunk"  # chunk, fused_chunk, or fused_recurrent
    gla_expand_k: float = 0.5  # key dimension expansion ratio
    gla_expand_v: float = 1.0  # value dimension expansion ratio  
    gla_use_short_conv: bool = True  # helps with local patterns
    gla_conv_size: int = 4
    gla_use_output_gate: bool = True  # gating is key to GLA
    gla_gate_fn: str = "swish"
    gla_gate_logit_normalizer: int = 16
    gla_gate_low_rank_dim: int = 16
    
    # CRAFT activation functions
    aft_k_activation: str = 'exp'
    aft_q_activation: str = 'sigmoid'
    ffn_activation: str = "swiglu"
    
    # Pooling configuration
    pooling_mode: Literal["sequence", "measure", "beat", "none"] = "sequence"
    
    # Decoder tokenizer options
    decoder_use_programs: bool = False
    
    # Dropout rates
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    cross_attn_pdrop: float = 0.15  # for cross-attention dropout
    layer_pdrop: float = 0.1  # for layer dropout
    
    # Training configuration
    layer_norm_epsilon: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    
    # RoPE configuration
    rope_theta: float = 10000.0
    n_positions: int = 8192
    
    # FLA optimizations
    use_fla_fused_mlp: bool = False
    use_fla_fused_norm: bool = False
    use_fla_fused_rotary: bool = False
    use_fla_fused_crossent_loss: bool = False
    
    # Required for Han2HanBlockCollection
    flipped_cross_attention: bool = False
    
    # For compatibility
    model_type: str = "mid2mid"
    pad_token_id: int = 0
    
    # Gradient checkpointing
    gradient_checkpointing: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.attention_mechanism == "mha" and self.num_heads is None:
            raise ValueError("num_heads must be specified for MHA attention")
        
        if self.attention_mechanism == "mha" and self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads for MHA")

        if self.attention_mechanism == 'mha':
            self.d_prime = self.d_model