"""Model configuration and the four experimental arms.

The arms exist so that the intervention is the ONLY difference between runs.
They are derived from a single `arm` string rather than set independently, so a
config cannot silently describe a model that is two interventions at once.

  A1  vanilla                 -- no provenance anywhere
  A2  prefix tokens           -- provenance as literal text tokens, vanilla model
  A3  additive bias           -- row-centered KxK authority bias on attention logits
  A4  full                    -- A3 + role/authority embeddings + monotone time bias

A2 is the decisive baseline. If A2 ties A4, six vocabulary entries did the work
and there is no architectural contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ARMS = ("A1", "A2", "A3", "A4")


@dataclass
class EncoderConfig:
    # --- shape ---
    vocab_size: int = 32_000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    max_segments: int = 8
    dropout: float = 0.1

    # --- provenance taxonomy ---
    n_roles: int = 8            # K. role 0 is reserved for UNK/pad.
    n_auth_buckets: int = 4

    # --- arm ---
    arm: str = "A4"

    # --- derived; do not set by hand ---
    use_role_embed: bool = field(init=False)
    use_authority_bias: bool = field(init=False)
    use_time_bias: bool = field(init=False)
    use_prefix_tokens: bool = field(init=False)

    # --- invariants ---
    center_authority: bool = True   # A <- A - rowmean(A) every forward pass
    zero_init_authority: bool = True  # never initialize at a hypothesized value

    # --- control: replace the segment bias with a pure distance bias (ALiBi) ---
    # A6. Uses NO segment or role information at all, only |i - j| on global
    # positions. If this recovers the gain, the mechanism is locality, not
    # segmentation, and the finding reduces further to a 2021 result.
    alibi_bias: bool = False

    # T5-style relative position bias: learned per (head, log-bucketed i-j).
    # This is the device's own comparison class -- segments average ~44 tokens, so
    # a segment bias is a boundary-aligned coarse relative-position bias. If
    # T5-RPE ties it, the finding is "the baseline lacked RPE", i.e. Raffel 2020.
    rpe_bias: bool = False
    rpe_buckets: int = 32
    rpe_max_distance: int = 128

    # Hard block-diagonal mask: the limit case of the learned segment bias.
    # If a fixed boolean mask matches the learned one, "learned" is decoration.
    hard_block_mask: bool = False

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")
        self.use_prefix_tokens = self.arm == "A2"
        self.use_authority_bias = self.arm in ("A3", "A4")
        self.use_role_embed = self.arm == "A4"
        self.use_time_bias = self.arm == "A4"

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def summary(self) -> str:
        flags = [
            f"role_embed={self.use_role_embed}",
            f"auth_bias={self.use_authority_bias}",
            f"time_bias={self.use_time_bias}",
            f"prefix={self.use_prefix_tokens}",
        ]
        return f"arm={self.arm} d={self.d_model} L={self.n_layers} H={self.n_heads} " + " ".join(flags)


def preset(name: str, arm: str = "A4") -> EncoderConfig:
    """Compute-matched size points. Sizes are non-embedding parameter targets."""
    sizes = {
        "10M": dict(d_model=384, n_layers=6, n_heads=6, d_ff=1536),
        "25M": dict(d_model=512, n_layers=8, n_heads=8, d_ff=2048),
        "40M": dict(d_model=576, n_layers=10, n_heads=9, d_ff=2304),
        "tiny": dict(d_model=128, n_layers=2, n_heads=4, d_ff=512),  # tests only
    }
    if name not in sizes:
        raise ValueError(f"unknown preset {name!r}; have {sorted(sizes)}")
    return EncoderConfig(arm=arm, **sizes[name])
