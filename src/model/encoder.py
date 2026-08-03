"""Provenance-aware transformer encoder, written from scratch (no backbone).

Design invariants, each traceable to a specific prior failure:

1. The authority matrix A is ROW-CENTERED at every forward pass.
   Softmax is shift-invariant, so A and A + c*1^T are the same function. Without
   centering, (a) a same-provenance sequence makes A a no-op, and (b) the
   singular spectrum of A is gauge-dependent and therefore not interpretable.
   Centering makes the identifiable part explicit.

2. A is initialized at ZERO, never at a hypothesized value.
   ACCORD initialized its authority scalars at [0.5, 1.0] and reported
   convergence to (0.508, 1.004) as recovery of a clinical hierarchy. The
   parameters had moved 1.6% and 0.4% from a hardcoded constant.

3. The temporal bias is MONOTONE in |delta log t|.
   A low-dimensional RoPE slice is a sum of cosines: periodic, so it wraps and
   spuriously boosts particular age ratios. Staleness is monotone.

4. The bias is a full-rank additive KxK table, not a low-rank factorization
   carved out of head channels. Rank-4 U,V on an 8x8 table is the same 64
   parameters, strictly less expressive, and costs 12.5% of content bandwidth.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EncoderConfig

NEG_INF = -1e9


class ProvenanceBias(nn.Module):
    """Additive attention-logit bias from (role_i, role_j) and event times.

    Returns a (B, H, S, S) float tensor added to the pre-softmax logits.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        K, H = cfg.n_roles, cfg.n_heads

        # Authority table, one KxK per head. Zero init (invariant 2).
        self.authority = nn.Parameter(torch.zeros(H, K, K))

        if cfg.use_time_bias:
            # softplus(decay_raw) is the decay rate; -5 -> ~0.0067, i.e. the model
            # starts with essentially no temporal prior and must learn one.
            self.decay_raw = nn.Parameter(torch.full((H,), -5.0))
            self.direction = nn.Parameter(torch.zeros(H))

    def centered_authority(self) -> torch.Tensor:
        """A - rowmean(A). The identifiable part; the only thing that affects the function."""
        A = self.authority
        if self.cfg.center_authority:
            A = A - A.mean(dim=-1, keepdim=True)
        return A

    def forward(
        self,
        role_ids: torch.Tensor,          # (B, S) long
        event_times: torch.Tensor | None,  # (B, S) float, hours; may be None
    ) -> torch.Tensor:
        B, S = role_ids.shape
        H, K = self.cfg.n_heads, self.cfg.n_roles

        A = self.centered_authority()                        # (H, K, K)
        flat_idx = (role_ids.unsqueeze(-1) * K + role_ids.unsqueeze(-2))  # (B, S, S)
        bias = A.reshape(H, K * K)[:, flat_idx.reshape(-1)]   # (H, B*S*S)
        bias = bias.view(H, B, S, S).permute(1, 0, 2, 3)      # (B, H, S, S)

        if self.cfg.use_time_bias and event_times is not None:
            lt = torch.log1p(event_times.clamp(min=0.0))      # (B, S)
            d = lt.unsqueeze(-1) - lt.unsqueeze(-2)           # (B, S, S) signed
            decay = F.softplus(self.decay_raw).view(1, H, 1, 1)
            direction = self.direction.view(1, H, 1, 1)
            bias = bias - decay * d.abs().unsqueeze(1) + direction * d.unsqueeze(1)

        return bias


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout_p = cfg.dropout

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        B, S, _ = x.shape
        H, dh = self.cfg.n_heads, self.cfg.d_head
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, H, dh).transpose(1, 2)
        k = k.view(B, S, H, dh).transpose(1, 2)
        v = v.view(B, S, H, dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, S, self.cfg.d_model)
        return self.proj(out)


class Block(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), attn_bias))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class ProvenanceEncoder(nn.Module):
    """Encoder whose arm is fully determined by cfg.arm."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Positions are LOCAL WITHIN SEGMENT: segment order in a bundle is not
        # semantic, so a position index spanning the bundle would encode an
        # ordering that does not exist.
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        if cfg.use_role_embed:
            self.role_embed = nn.Embedding(cfg.n_roles, cfg.d_model)
            self.auth_embed = nn.Embedding(cfg.n_auth_buckets, cfg.d_model)
            nn.init.zeros_(self.role_embed.weight)
            nn.init.zeros_(self.auth_embed.weight)

        self.bias_module = ProvenanceBias(cfg) if cfg.use_authority_bias else None

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Tied MLM head.
        self.mlm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.mlm_head.weight = self.tok_embed.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,                 # (B, S)
        local_positions: torch.Tensor,           # (B, S)
        attention_mask: torch.Tensor,            # (B, S) 1 = real, 0 = pad
        role_ids: torch.Tensor | None = None,    # (B, S)
        auth_ids: torch.Tensor | None = None,    # (B, S)
        event_times: torch.Tensor | None = None,  # (B, S) hours
    ) -> torch.Tensor:
        B, S = input_ids.shape
        x = self.tok_embed(input_ids) + self.pos_embed(local_positions)
        if self.cfg.use_role_embed:
            if role_ids is None or auth_ids is None:
                raise ValueError("arm A4 requires role_ids and auth_ids")
            x = x + self.role_embed(role_ids) + self.auth_embed(auth_ids)
        x = self.drop(x)

        # Padding mask as an additive float mask, broadcast over heads.
        pad = (1.0 - attention_mask[:, None, None, :].to(x.dtype)) * NEG_INF

        bias = None
        if self.bias_module is not None:
            if role_ids is None:
                raise ValueError(f"arm {self.cfg.arm} requires role_ids")
            bias = self.bias_module(role_ids, event_times)
            bias = bias + pad
        else:
            bias = pad.expand(B, self.cfg.n_heads, S, S)

        for blk in self.blocks:
            x = blk(x, bias)
        return self.ln_f(x)

    def mlm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlm_head(hidden)

    # --- accounting -------------------------------------------------------
    def n_params(self, non_embedding: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.tok_embed.weight.numel()
            total -= self.pos_embed.weight.numel()
        return total

    def provenance_params(self) -> dict[str, torch.Tensor]:
        """Every parameter this project would ever call 'learned' and interpret.

        The training loop records init snapshots and gradient norms for exactly
        these, so no interpretability claim can be made about a parameter that
        never moved.
        """
        out: dict[str, torch.Tensor] = {}
        if self.bias_module is not None:
            out["authority"] = self.bias_module.authority
            if self.cfg.use_time_bias:
                out["decay_raw"] = self.bias_module.decay_raw
                out["direction"] = self.bias_module.direction
        if self.cfg.use_role_embed:
            out["role_embed"] = self.role_embed.weight
            out["auth_embed"] = self.auth_embed.weight
        return out
