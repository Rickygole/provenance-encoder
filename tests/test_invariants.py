"""Executable versions of the failure modes that killed the previous designs.

These are not unit tests in the usual sense. Each one encodes a specific claim
that was made, believed, and turned out to be false. If any of them fails, a
paper claim is wrong.

Run: python3 -m pytest tests/test_invariants.py -q     (or python3 tests/test_invariants.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.config import EncoderConfig, preset  # noqa: E402
from src.model.encoder import ProvenanceEncoder  # noqa: E402

torch.manual_seed(0)


def _batch(B=2, S=16, roles=None, times=None, vocab=256):
    ids = torch.randint(1, vocab, (B, S))
    pos = torch.arange(S).unsqueeze(0).expand(B, S).contiguous()
    mask = torch.ones(B, S)
    if roles is None:
        roles = torch.zeros(B, S, dtype=torch.long)
    auth = torch.zeros(B, S, dtype=torch.long)
    if times is None:
        times = torch.zeros(B, S)
    return ids, pos, mask, roles, auth, times


def _cfg(arm: str) -> EncoderConfig:
    c = preset("tiny", arm=arm)
    c.vocab_size = 256
    c.max_seq_len = 64
    c.dropout = 0.0
    return c


# ---------------------------------------------------------------------------
# 1. THE KILL SHOT, made executable.
# ---------------------------------------------------------------------------
def test_single_provenance_bias_is_inert():
    """On a same-provenance sequence the authority bias does NOTHING.

    softmax(s + c) = softmax(s). If every token shares one role, A[r_i,r_j] is a
    single constant added to every logit. This is why the case-law design was
    dead: every benchmark it named is single-document. The cure is a
    multi-provenance INPUT, not a cleverer parameterization -- row-centering
    does not save it, as this test shows by using a trained-looking A.
    """
    cfg = _cfg("A3")
    m = ProvenanceEncoder(cfg).eval()
    with torch.no_grad():  # pretend training moved A somewhere non-trivial
        m.bias_module.authority.normal_(0, 1.0)

    ids, pos, mask, roles, auth, times = _batch()
    roles_all_same = torch.full_like(roles, 3)

    with torch.no_grad():
        out_biased = m(ids, pos, mask, roles_all_same, auth, times)
        m.bias_module.authority.zero_()
        out_zero = m(ids, pos, mask, roles_all_same, auth, times)

    assert torch.allclose(out_biased, out_zero, atol=1e-5), (
        "single-provenance input should be unaffected by A; if this fails the "
        "shift-invariance argument is wrong"
    )
    print("  [1] single-provenance input: authority bias confirmed INERT")


def test_multi_provenance_bias_is_active():
    """With >=2 roles in one sequence the bias changes the function."""
    cfg = _cfg("A3")
    m = ProvenanceEncoder(cfg).eval()
    ids, pos, mask, roles, auth, times = _batch()
    S = roles.shape[1]
    mixed = torch.zeros_like(roles)
    mixed[:, S // 2:] = 3  # two distinct provenances in one sequence

    with torch.no_grad():
        out_zero = m(ids, pos, mask, mixed, auth, times)
        m.bias_module.authority.normal_(0, 1.0)
        out_biased = m(ids, pos, mask, mixed, auth, times)

    diff = (out_biased - out_zero).abs().max().item()
    assert diff > 1e-4, f"multi-provenance bias had no effect (max diff {diff:.2e})"
    print(f"  [2] multi-provenance input: bias ACTIVE (max delta {diff:.4f})")


# ---------------------------------------------------------------------------
# 2. Identifiability. This is what row-centering actually buys.
# ---------------------------------------------------------------------------
def test_row_shift_is_a_gauge_symmetry():
    """A and A + c*1^T are the same function, so an uncentered spectrum is meaningless.

    v1 proposed reading the singular spectrum of A as evidence about whether
    authority is ordinal. Two runs converging to the same function can report
    different spectra unless the centered form is used.
    """
    cfg = _cfg("A3")
    m = ProvenanceEncoder(cfg).eval()
    ids, pos, mask, roles, auth, times = _batch()
    S = roles.shape[1]
    mixed = torch.zeros_like(roles)
    mixed[:, S // 2:] = 2

    with torch.no_grad():
        m.bias_module.authority.normal_(0, 1.0)
        out_a = m(ids, pos, mask, mixed, auth, times)
        c = torch.randn(cfg.n_heads, cfg.n_roles, 1)
        m.bias_module.authority.add_(c)          # add a constant per row
        out_b = m(ids, pos, mask, mixed, auth, times)

    assert torch.allclose(out_a, out_b, atol=1e-5), "row shift changed the function"

    raw = m.bias_module.authority
    centered = m.bias_module.centered_authority()
    raw_rank = torch.linalg.matrix_rank(raw[0].double()).item()
    cen_rank = torch.linalg.matrix_rank(centered[0].double()).item()
    assert torch.allclose(centered.mean(dim=-1), torch.zeros(cfg.n_heads, cfg.n_roles), atol=1e-6)
    print(f"  [3] row shift is a gauge symmetry (raw rank {raw_rank}, centered rank {cen_rank})")
    print("      -> only the centered spectrum may be interpreted")


# ---------------------------------------------------------------------------
# 3. The ACCORD alpha artifact, made impossible to repeat.
# ---------------------------------------------------------------------------
def test_authority_is_zero_initialized():
    """ACCORD initialized authority at [0.5, 1.0] and reported 'converging' to
    (0.508, 1.004) as a discovery. Init must never encode the hypothesis."""
    for arm in ("A3", "A4"):
        m = ProvenanceEncoder(_cfg(arm))
        A = m.bias_module.authority
        assert torch.count_nonzero(A) == 0, f"{arm}: authority not zero-initialized"
    print("  [4] authority zero-initialized in A3 and A4")


def test_time_bias_starts_negligible_and_is_monotone():
    """Temporal prior must be learned, not assumed; and monotone in |dt|."""
    cfg = _cfg("A4")
    m = ProvenanceEncoder(cfg).eval()
    import torch.nn.functional as F

    decay0 = F.softplus(m.bias_module.decay_raw).max().item()
    assert decay0 < 0.02, f"time decay starts at {decay0}, not negligible"
    assert torch.count_nonzero(m.bias_module.direction) == 0

    with torch.no_grad():  # force a decay and check monotonicity in |dt|
        m.bias_module.decay_raw.fill_(1.0)
        m.bias_module.direction.zero_()
        roles = torch.zeros(1, 3, dtype=torch.long)
        times = torch.tensor([[0.0, 10.0, 1000.0]])
        b = m.bias_module(roles, times)[0, 0]
    near, far = b[0, 1].item(), b[0, 2].item()
    assert far < near, f"bias not decreasing with time distance ({far} !< {near})"
    print(f"  [5] time bias negligible at init, monotone in |dt| ({near:.3f} -> {far:.3f})")


def test_provenance_params_are_all_tracked():
    """Every parameter that could ever be interpreted must be enumerable, so the
    trainer can record its trajectory from init and its gradient norms."""
    m4 = ProvenanceEncoder(_cfg("A4"))
    tracked = m4.provenance_params()
    assert set(tracked) == {"authority", "decay_raw", "direction", "role_embed", "auth_embed"}
    m1 = ProvenanceEncoder(_cfg("A1"))
    assert m1.provenance_params() == {}, "vanilla arm should expose no provenance params"
    print(f"  [6] provenance params tracked: {sorted(tracked)}")


# ---------------------------------------------------------------------------
# 4. Arms differ only by the intervention.
# ---------------------------------------------------------------------------
def test_arms_are_well_formed():
    flags = {}
    for arm in ("A1", "A2", "A3", "A4"):
        c = _cfg(arm)
        m = ProvenanceEncoder(c)
        flags[arm] = (c.use_role_embed, c.use_authority_bias, c.use_time_bias, c.use_prefix_tokens)
        n = m.n_params(non_embedding=True)
        print(f"      {arm}: non-emb params {n:,}  {c.summary()}")
    assert flags["A1"] == (False, False, False, False)
    assert flags["A2"] == (False, False, False, True)
    assert flags["A3"] == (False, True, False, False)
    assert flags["A4"] == (True, True, True, False)

    # A1 and A2 are the same MODEL; A2's intervention lives entirely in the data.
    n1 = ProvenanceEncoder(_cfg("A1")).n_params()
    n2 = ProvenanceEncoder(_cfg("A2")).n_params()
    assert n1 == n2, "A2 must be architecturally identical to A1"
    print("  [7] arms well-formed; A2 architecturally identical to A1")


def test_forward_shapes():
    cfg = _cfg("A4")
    m = ProvenanceEncoder(cfg)
    ids, pos, mask, roles, auth, times = _batch(B=3, S=24)
    roles = torch.randint(0, cfg.n_roles, (3, 24))
    auth = torch.randint(0, cfg.n_auth_buckets, (3, 24))
    times = torch.rand(3, 24) * 100
    h = m(ids, pos, mask, roles, auth, times)
    assert h.shape == (3, 24, cfg.d_model)
    assert m.mlm_logits(h).shape == (3, 24, cfg.vocab_size)
    print("  [8] forward shapes OK")


def test_padding_is_masked():
    cfg = _cfg("A3")
    m = ProvenanceEncoder(cfg).eval()
    ids, pos, mask, roles, auth, times = _batch(B=1, S=12)
    roles[:, 6:] = 2
    mask[:, 8:] = 0
    with torch.no_grad():
        out_a = m(ids, pos, mask, roles, auth, times)
        ids2 = ids.clone()
        ids2[:, 8:] = torch.randint(1, 250, (1, 4))  # scramble padded region
        out_b = m(ids2, pos, mask, roles, auth, times)
    assert torch.allclose(out_a[:, :8], out_b[:, :8], atol=1e-5), "padding leaked into real tokens"
    print("  [9] padding masked correctly")


if __name__ == "__main__":
    print("\nInvariant tests (each encodes a real prior failure):\n")
    for fn in [
        test_single_provenance_bias_is_inert,
        test_multi_provenance_bias_is_active,
        test_row_shift_is_a_gauge_symmetry,
        test_authority_is_zero_initialized,
        test_time_bias_starts_negligible_and_is_monotone,
        test_provenance_params_are_all_tracked,
        test_arms_are_well_formed,
        test_forward_shapes,
        test_padding_is_masked,
    ]:
        fn()
    print("\nAll invariants hold.\n")
