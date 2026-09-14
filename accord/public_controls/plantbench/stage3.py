"""C102 Stage 3: detector-defect checks. See ../C102_PREREGISTRATION.md.

Two run modes:
- `smoke_test()` / `main(mode="smoke")`: 2 tiny replicates, CPU, tiny
  dimensions, untrained. Proves the code runs; not a verdict. This is the
  ONLY thing that ran before the coordinator released the GPU.
- `registered_run()` / `main(mode="registered")`: the full, pre-registered
  run -- real frozen-file dimensions, MPS, 20 replicates per arm per check,
  fixed thresholds unchanged from the smoke test. Run only once the
  coordinator has confirmed the GPU is free (checked immediately before this
  runs, not cached from an earlier check).

Uses the real, frozen `AuthorityTemporalConflictGraph` and
`GuidelineGroundedHarmEstimator` classes via
`accord/public_controls/frozen_ablations_loader.py` (ast-parsed, sha256-pinned;
`accord/detector/accord_phase3_v2_ablations.py` is never imported or executed
as a whole). No MIMIC data, no network, ever. All inputs are synthetic
(iid Gaussian / categorical draws) -- there is no BiomedBERT feature cache to
read, and none is read; using synthetic tensors of the frozen classes' own
real shapes is the public/synthetic-data-only substitute, exactly as C101
used synthetic label-mix/noise-mix constructions rather than clinical data.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binom

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accord.public_controls import frozen_ablations_loader as fl  # noqa: E402

OUT_SMOKE = ROOT / "results" / "accord" / "plantbench" / "stage3_smoke.json"
OUT_REGISTERED = ROOT / "results" / "accord" / "plantbench" / "stage3_registered.json"
OUT = OUT_SMOKE  # backward-compatible default for existing callers

# Fixed thresholds, pre-registered, never tuned on these or any pilot seeds.
S3A_DISPLACEMENT_THRESHOLD = 0.02
S3B_GRADIENT_RMS_RATIO_THRESHOLD = 0.05
S3C_MACRO_F1_DROP_THRESHOLD = 0.20
S3D_WEIGHT_CHANGE_RATIO_THRESHOLD = 1e-6

SMOKE_SEEDS = [1001, 1002]  # "2 tiny replicates"
TINY_DIMS = dict(node_dim=8, hidden_dim=8, num_heads=2)
TINY_GGHE_DIMS = dict(edge_dim=16, prior_dim=4, hidden_dim=8, num_labels=3, num_rules=4)

# The frozen file's OWN default constructor dimensions (accord_phase3_v2_
# ablations.py: AuthorityTemporalConflictGraph.__init__, GuidelineGrounded
# HarmEstimator.__init__), used unmodified for the registered run so it
# probes the real architecture, not a scaled-down stand-in.
REAL_DIMS = dict(node_dim=256, hidden_dim=256, num_heads=4)
REAL_GGHE_DIMS = dict(edge_dim=512, prior_dim=64, hidden_dim=256, num_labels=3, num_rules=4)

N_REPLICATES_REGISTERED = 20
REGISTERED_BATCH_N = 64  # candidate items per replicate; not specified in the
# pre-registration's text, disclosed here: large enough for a stable macro-F1
# over 4 rule classes (S3c), small enough to run fast per replicate.
REGISTERED_SEED_ROOT = 42


def _macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def s3a_authority_displacement(seed, dims=None, device="cpu", clean: bool = False):
    """Displacement in `conflict_score` between AuthorityTemporalConflictGraph's
    authority_matrix used as-is and a row-normalized copy substituted in,
    same weights, same input. Deterministic given the seed.

    clean=False (PLANT): authority_matrix is left at the frozen file's own
    hardcoded init, `[[1.0, 0.5], [1.5, 1.0]]` -- NOT row-stochastic (rows
    sum to 1.5 and 2.5), the actual as-shipped behaviour. clean=True: the
    same starting values are first row-normalized in place before the
    "as-is vs normalized-copy" comparison runs, so there is nothing left for
    a further row-normalization to change and displacement should be ~0.
    This clean arm is an addition made when Stage 3 moved from a CPU smoke
    test to the registered run, needed to give S3a a false-alarm arm at all
    (the smoke-test version only ever probed the plant configuration);
    disclosed here rather than folded in silently. The fixed 0.02 threshold
    is unchanged.
    """
    dims = dims or TINY_DIMS
    torch.manual_seed(seed)
    ns = fl.load(["AuthorityTemporalConflictGraph"])
    model = ns["AuthorityTemporalConflictGraph"](**dims).to(device).eval()
    if clean:
        with torch.no_grad():
            raw = model.authority_matrix.clone()
            model.authority_matrix.copy_(raw / raw.sum(dim=1, keepdim=True))
    n_node = torch.randn(16, dims["node_dim"], device=device)
    d_node = torch.randn(16, dims["node_dim"], device=device)
    with torch.no_grad():
        _, score_before, _ = model(n_node, d_node)
        raw = model.authority_matrix.clone()
        row_sums = raw.sum(dim=1, keepdim=True)
        model.authority_matrix.copy_(raw / row_sums)
        _, score_after, _ = model(n_node, d_node)
        model.authority_matrix.copy_(raw)  # restore, this instance is discarded anyway
    displacement = float((score_before - score_after).abs().mean())
    return {"displacement": displacement, "fires": displacement >= S3A_DISPLACEMENT_THRESHOLD,
            "threshold": S3A_DISPLACEMENT_THRESHOLD, "clean": clean}


def _build_gghe(seed, gghe_dims=None, device="cpu"):
    gghe_dims = gghe_dims or TINY_GGHE_DIMS
    torch.manual_seed(seed)
    ns = fl.load(["GuidelineGroundedHarmEstimator"])
    return ns["GuidelineGroundedHarmEstimator"](**gghe_dims, use_multitask=True).to(device)


def s3b_gradient_rms_ratio(seed, plant: bool, gghe_dims=None, device="cpu", n=6):
    """Ratio of gradient RMS the rule-id (auxiliary) loss contributes to the
    shared `input_projection` weights, over the gradient RMS the
    contradiction (main) loss contributes to the same weights.

    plant=True: rule id target is an exact function of prior_vector's own
    rule-derivable feature (a shortcut the head can solve without the shared
    trunk). plant=False (clean, standing in for "a text-carried template
    family"): rule id target is independent of every engineered feature.
    """
    gghe_dims = gghe_dims or TINY_GGHE_DIMS
    torch.manual_seed(seed)
    model = _build_gghe(seed, gghe_dims, device)
    edge_repr = torch.randn(n, gghe_dims["edge_dim"], device=device)
    prior = torch.randn(n, gghe_dims["prior_dim"], device=device)
    rule_feature = torch.randint(0, 4, (n,), device=device)
    prior[:, 1] = rule_feature.float() / 4.0  # matches ACCORDv2Full._prior's own encoding
    rule_target = rule_feature if plant else torch.randint(0, 4, (n,), device=device)
    contra_target = torch.randint(0, 3, (n,), device=device)

    c_logits, r_logits, harm, expl, attn = model(edge_repr, prior)
    shared_w = model.input_projection[0].weight

    model.zero_grad()
    contra_loss = F.cross_entropy(c_logits, contra_target)
    contra_loss.backward(retain_graph=True)
    grad_contra_rms = float(shared_w.grad.pow(2).mean().sqrt())

    model.zero_grad()
    rule_loss = F.cross_entropy(r_logits, rule_target)
    rule_loss.backward()
    grad_rule_rms = float(shared_w.grad.pow(2).mean().sqrt())

    ratio = grad_rule_rms / grad_contra_rms if grad_contra_rms > 0 else float("inf")
    return {"grad_rms_rule": grad_rule_rms, "grad_rms_contradiction": grad_contra_rms,
            "ratio": ratio, "fires": ratio < S3B_GRADIENT_RMS_RATIO_THRESHOLD,
            "threshold": S3B_GRADIENT_RMS_RATIO_THRESHOLD, "plant": plant}


def s3c_macro_f1_drop_on_zeroed_feature(seed, plant: bool, n_items=40, gghe_dims=None, device="cpu"):
    """Macro-F1 of rule_logits argmax vs. the rule-id target, feature present
    vs. zeroed, same (untrained, random-init) weights. plant=True: rule
    target is the exact function of the feature that gets zeroed (a real
    shortcut, so zeroing it should hurt); plant=False: rule target is
    independent of the feature (zeroing it should not matter)."""
    gghe_dims = gghe_dims or TINY_GGHE_DIMS
    torch.manual_seed(seed)
    model = _build_gghe(seed, gghe_dims, device).eval()
    edge_repr = torch.randn(n_items, gghe_dims["edge_dim"], device=device)
    prior = torch.randn(n_items, gghe_dims["prior_dim"], device=device)
    rule_feature = torch.randint(0, 4, (n_items,), device=device)
    prior[:, 1] = rule_feature.float() / 4.0
    rule_target = (rule_feature if plant else torch.randint(0, 4, (n_items,), device=device)).cpu().numpy()

    with torch.no_grad():
        c_logits, r_logits_present, harm, expl, attn = model(edge_repr, prior)
        pred_present = r_logits_present.argmax(dim=-1).cpu().numpy()
        prior_zeroed = prior.clone()
        prior_zeroed[:, 1] = 0.0
        c_logits2, r_logits_zeroed, *_ = model(edge_repr, prior_zeroed)
        pred_zeroed = r_logits_zeroed.argmax(dim=-1).cpu().numpy()

    f1_present = _macro_f1(rule_target, pred_present, gghe_dims["num_rules"])
    f1_zeroed = _macro_f1(rule_target, pred_zeroed, gghe_dims["num_rules"])
    drop = f1_present - f1_zeroed
    return {"macro_f1_present": f1_present, "macro_f1_zeroed": f1_zeroed, "drop": drop,
            "fires": drop >= S3C_MACRO_F1_DROP_THRESHOLD,
            "threshold": S3C_MACRO_F1_DROP_THRESHOLD, "plant": plant}


def s3d_harm_weight_change_ratio(seed, harm_in_loss: bool, gghe_dims=None, device="cpu", n=6):
    """L2 norm of the change in harm_head's weights after one optimizer step,
    over the L2 norm of its initial weights. harm_in_loss=False is the
    plant ("harm head outside the loss": excluded from backward, so the
    optimizer step should leave it exactly unchanged); harm_in_loss=True is
    clean (included, so it should move)."""
    gghe_dims = gghe_dims or TINY_GGHE_DIMS
    torch.manual_seed(seed)
    model = _build_gghe(seed, gghe_dims, device)
    edge_repr = torch.randn(n, gghe_dims["edge_dim"], device=device)
    prior = torch.randn(n, gghe_dims["prior_dim"], device=device)
    contra_target = torch.randint(0, 3, (n,), device=device)
    harm_target = torch.rand(n, 1, device=device)

    harm_weight_before = model.harm_head[0].weight.clone()
    init_norm = float(harm_weight_before.norm())

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt.zero_grad()
    c_logits, r_logits, harm, expl, attn = model(edge_repr, prior)
    loss = F.cross_entropy(c_logits, contra_target)
    if harm_in_loss:
        loss = loss + F.binary_cross_entropy(harm, harm_target)
    loss.backward()
    opt.step()

    weight_change = float((model.harm_head[0].weight - harm_weight_before).norm())
    ratio = weight_change / init_norm if init_norm > 0 else float("inf")
    return {"weight_change_norm": weight_change, "init_norm": init_norm, "ratio": ratio,
            "fires": ratio < S3D_WEIGHT_CHANGE_RATIO_THRESHOLD,
            "threshold": S3D_WEIGHT_CHANGE_RATIO_THRESHOLD, "harm_in_loss": harm_in_loss}


def smoke_test():
    """Runs every check on SMOKE_SEEDS (2 tiny replicates), both plant and
    clean arms where applicable. Proves the code executes and returns finite
    numbers; does NOT compute a validated/unvalidated verdict (that needs the
    real feature cache, real dimensions, and a trained checkpoint on MPS)."""
    results = {"s3a_plant": [], "s3a_clean": [], "s3b_plant": [], "s3b_clean": [],
              "s3c_plant": [], "s3c_clean": [],
              "s3d_clean_harm_in_loss": [], "s3d_plant_harm_out_of_loss": []}
    for seed in SMOKE_SEEDS:
        results["s3a_plant"].append(s3a_authority_displacement(seed, clean=False))
        results["s3a_clean"].append(s3a_authority_displacement(seed, clean=True))
        results["s3b_plant"].append(s3b_gradient_rms_ratio(seed, plant=True))
        results["s3b_clean"].append(s3b_gradient_rms_ratio(seed, plant=False))
        results["s3c_plant"].append(s3c_macro_f1_drop_on_zeroed_feature(seed, plant=True))
        results["s3c_clean"].append(s3c_macro_f1_drop_on_zeroed_feature(seed, plant=False))
        results["s3d_clean_harm_in_loss"].append(s3d_harm_weight_change_ratio(seed, harm_in_loss=True))
        results["s3d_plant_harm_out_of_loss"].append(s3d_harm_weight_change_ratio(seed, harm_in_loss=False))

    all_finite = all(
        np.isfinite(r[k]) for group in results.values() for r in group
        for k in r if isinstance(r.get(k), float)
    )
    return {
        "preregistration": "accord/public_controls/C102_PREREGISTRATION.md",
        "stage": "3", "status": "CPU_SMOKE_TEST_ONLY_NOT_A_VERDICT",
        "note": ("2 tiny replicates, untrained random-init weights, tiny "
                "dimensions. Full run (real dims, 20 replicates, MPS) is "
                "`registered_run()`, run only once the coordinator confirms "
                "the GPU is free; thresholds above are fixed and unchanged."),
        "smoke_seeds": SMOKE_SEEDS, "dims": {**TINY_DIMS, **TINY_GGHE_DIMS},
        "all_outputs_finite": bool(all_finite),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Registered run: real dimensions, MPS, 20 replicates per arm per check.
# ---------------------------------------------------------------------------

def _fire_table(fires):
    n = len(fires)
    k = int(sum(fires))
    return {
        "n": n, "fires": k, "rate": k / n,
        "binomial_probability_of_this_count_at_true_rate": {
            str(r): float(binom.pmf(k, n, r)) for r in (0.5, 0.7, 0.9, 0.95)
        },
    }


def _verdict(plant_table, clean_tables):
    """General C102 verdict rule: validated if plant fires >=18/20 AND every
    clean arm fires <=2/20; unvalidated_false_alarm if any clean arm >5/20;
    specificity_marginal if 3-5/20 on any clean arm; not_validated_sensitivity
    if plant fires <18/20."""
    max_clean = max((t["fires"] for t in clean_tables), default=0)
    if max_clean > 5:
        return "unvalidated_false_alarm"
    if plant_table["fires"] >= 18:
        return "validated" if max_clean <= 2 else "specificity_marginal"
    if 3 <= max_clean <= 5:
        return "specificity_marginal"
    return "not_validated_sensitivity"


def registered_run(device="mps"):
    """The full, pre-registered Stage 3 run: real frozen-file dimensions,
    `N_REPLICATES_REGISTERED` (20) replicates per arm per check, on `device`.
    Seeds: numpy.random.SeedSequence(REGISTERED_SEED_ROOT), spawned once per
    check so no seed is reused across checks; within a check, the 20
    replicate seeds are shared between its plant and clean arms (paired
    seeds, same random draw of everything except the plant/clean switch
    itself) so the comparison isolates the switch, not sampling noise.
    """
    root = np.random.SeedSequence(REGISTERED_SEED_ROOT)
    s3a_seeds, s3b_seeds, s3c_seeds, s3d_seeds = root.spawn(4)
    seeds = lambda ss: [int(c.generate_state(1)[0]) for c in ss.spawn(N_REPLICATES_REGISTERED)]

    s3a_plant = [s3a_authority_displacement(sd, dims=REAL_DIMS, device=device, clean=False)
                for sd in seeds(s3a_seeds)]
    s3a_clean = [s3a_authority_displacement(sd, dims=REAL_DIMS, device=device, clean=True)
                for sd in seeds(s3a_seeds)]

    s3b_plant = [s3b_gradient_rms_ratio(sd, plant=True, gghe_dims=REAL_GGHE_DIMS, device=device,
                                        n=REGISTERED_BATCH_N) for sd in seeds(s3b_seeds)]
    s3b_clean = [s3b_gradient_rms_ratio(sd, plant=False, gghe_dims=REAL_GGHE_DIMS, device=device,
                                        n=REGISTERED_BATCH_N) for sd in seeds(s3b_seeds)]

    s3c_plant = [s3c_macro_f1_drop_on_zeroed_feature(sd, plant=True, n_items=REGISTERED_BATCH_N,
                                                     gghe_dims=REAL_GGHE_DIMS, device=device)
                for sd in seeds(s3c_seeds)]
    s3c_clean = [s3c_macro_f1_drop_on_zeroed_feature(sd, plant=False, n_items=REGISTERED_BATCH_N,
                                                     gghe_dims=REAL_GGHE_DIMS, device=device)
                for sd in seeds(s3c_seeds)]

    s3d_plant = [s3d_harm_weight_change_ratio(sd, harm_in_loss=False, gghe_dims=REAL_GGHE_DIMS,
                                              device=device, n=REGISTERED_BATCH_N)
                for sd in seeds(s3d_seeds)]
    s3d_clean = [s3d_harm_weight_change_ratio(sd, harm_in_loss=True, gghe_dims=REAL_GGHE_DIMS,
                                              device=device, n=REGISTERED_BATCH_N)
                for sd in seeds(s3d_seeds)]

    checks = {}
    for name, plant, clean in (
        ("S3a_authority_normalisation_displacement", s3a_plant, s3a_clean),
        ("S3b_rule_id_shortcut_gradient_rms_ratio", s3b_plant, s3b_clean),
        ("S3c_rule_feature_ablation_macro_f1_drop", s3c_plant, s3c_clean),
        ("S3d_harm_head_weight_change_ratio", s3d_plant, s3d_clean),
    ):
        plant_fires = [r["fires"] for r in plant]
        clean_fires = [r["fires"] for r in clean]
        plant_table = _fire_table(plant_fires)
        clean_table = _fire_table(clean_fires)
        checks[name] = {
            "kind": "stochastic_over_20_replicates",
            "plant": plant_table, "clean": clean_table,
            "verdict": _verdict(plant_table, [clean_table]),
            "raw_plant": plant, "raw_clean": clean,
        }

    all_validated = all(c["verdict"] == "validated" for c in checks.values())
    out = {
        "preregistration": "accord/public_controls/C102_PREREGISTRATION.md",
        "stage": "3", "status": "REGISTERED_RUN",
        "device": device, "dims": REAL_DIMS, "gghe_dims": REAL_GGHE_DIMS,
        "n_replicates": N_REPLICATES_REGISTERED, "batch_n": REGISTERED_BATCH_N,
        "seed_root": REGISTERED_SEED_ROOT,
        "note": ("Real frozen-file dimensions, synthetic (iid random) inputs "
                "-- no BiomedBERT feature cache exists or is read, no MIMIC "
                "data. Untrained (random-init) weights throughout: no "
                "training was ever in this build's scope. S3a's clean arm "
                "(row-pre-normalized authority_matrix) was added when this "
                "run was built, disclosed in s3a_authority_displacement's "
                "docstring; thresholds are unchanged from the smoke test."),
        "checks": checks,
        "all_stochastic_checks_validated": bool(all_validated),
    }
    return out


def stage3_to_stage4_bridge(device="mps", n=253, n_pos=12, seed=REGISTERED_SEED_ROOT + 1):
    """Produces Stage 4's required input: a (y, s) pair at Stage 4's
    pre-registered 12/253 size. `s` is the real, frozen
    AuthorityTemporalConflictGraph + GuidelineGroundedHarmEstimator pipeline's
    raw contradiction score (log p(contradiction) - log(1 - p(contradiction)))
    on `n` synthetic candidate items (untrained, random-init weights -- this
    build never trains anything); `y` is an INDEPENDENT 12/253 synthetic gold
    assignment, unrelated to `s` by construction. This is the honest
    substitute for "the real, trained detector's scores against this
    synthetic gold": there is no trained detector in this build's scope, so
    the untrained architecture's raw output is used as-is, and the gold is
    not constructed to correlate with it. A null (or near-null) result in
    Stage 4 is therefore the expected, calibration-appropriate finding, not
    evidence the pipeline is broken.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    atcg_ns = fl.load(["AuthorityTemporalConflictGraph"])
    gghe_ns = fl.load(["GuidelineGroundedHarmEstimator"])
    atcg = atcg_ns["AuthorityTemporalConflictGraph"](**REAL_DIMS).to(device).eval()
    gghe = gghe_ns["GuidelineGroundedHarmEstimator"](**REAL_GGHE_DIMS, use_multitask=True).to(device).eval()

    n_node = torch.randn(n, REAL_DIMS["node_dim"], device=device)
    d_node = torch.randn(n, REAL_DIMS["node_dim"], device=device)
    prior = torch.randn(n, REAL_GGHE_DIMS["prior_dim"], device=device)
    with torch.no_grad():
        edge_repr, _, _ = atcg(n_node, d_node)
        c_logits, r_logits, harm, expl, attn = gghe(edge_repr, prior)
        probs = F.softmax(c_logits, dim=-1)
        # LABEL_MAP (allow-listed assign in the frozen file) puts
        # contradiction at index 0.
        p_contra = probs[:, 0].clamp(1e-6, 1 - 1e-6)
        s = (torch.log(p_contra) - torch.log(1 - p_contra)).cpu().numpy()

    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_pos, replace=False)] = 1
    return y.tolist(), s.tolist()


def main(mode="smoke", device="mps"):
    if mode == "smoke":
        out = smoke_test()
        OUT_SMOKE.parent.mkdir(parents=True, exist_ok=True)
        OUT_SMOKE.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
        print(json.dumps(out["results"], indent=2))
    elif mode == "registered":
        out = registered_run(device=device)
        OUT_REGISTERED.parent.mkdir(parents=True, exist_ok=True)
        printable = {k: v for k, v in out.items() if k != "checks"}
        printable["checks"] = {
            name: {kk: vv for kk, vv in c.items() if kk not in ("raw_plant", "raw_clean")}
            for name, c in out["checks"].items()
        }
        OUT_REGISTERED.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(printable, indent=2))
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    _mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    _device = sys.argv[2] if len(sys.argv) > 2 else "mps"
    main(mode=_mode, device=_device)
