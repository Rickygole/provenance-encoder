"""C102 Stage 3: detector-defect checks. See ../C102_PREREGISTRATION.md.

Needs a frozen BiomedBERT feature cache and MPS for the full, validated run.
THIS FILE ONLY BUILDS THE CHECK CODE AND RUNS A CPU SMOKE TEST ON 2 TINY
REPLICATES; `main()` runs the smoke test only and writes a JSON explicitly
labelled as a smoke test, not a verdict. The full run is the coordinator's,
once the GPU is free.

Uses the real, frozen `AuthorityTemporalConflictGraph` and
`GuidelineGroundedHarmEstimator` classes via
`accord/public_controls/frozen_ablations_loader.py` (ast-parsed, sha256-pinned;
`accord/detector/accord_phase3_v2_ablations.py` is never imported or executed
as a whole). No MIMIC data, no MPS, no network.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accord.public_controls import frozen_ablations_loader as fl  # noqa: E402

OUT = ROOT / "results" / "accord" / "plantbench" / "stage3_smoke.json"

# Fixed thresholds, pre-registered, never tuned on these or any pilot seeds.
S3A_DISPLACEMENT_THRESHOLD = 0.02
S3B_GRADIENT_RMS_RATIO_THRESHOLD = 0.05
S3C_MACRO_F1_DROP_THRESHOLD = 0.20
S3D_WEIGHT_CHANGE_RATIO_THRESHOLD = 1e-6

SMOKE_SEEDS = [1001, 1002]  # "2 tiny replicates"
TINY_DIMS = dict(node_dim=8, hidden_dim=8, num_heads=2)
TINY_GGHE_DIMS = dict(edge_dim=16, prior_dim=4, hidden_dim=8, num_labels=3, num_rules=4)


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


def s3a_authority_displacement(seed):
    """Displacement in `conflict_score` between AuthorityTemporalConflictGraph's
    authority_matrix used as-is (the frozen file's own behaviour: raw,
    unnormalized) vs a row-normalized copy substituted in, same weights,
    same input. Deterministic given the seed."""
    torch.manual_seed(seed)
    ns = fl.load(["AuthorityTemporalConflictGraph"])
    model = ns["AuthorityTemporalConflictGraph"](**TINY_DIMS).eval()
    n_node = torch.randn(4, TINY_DIMS["node_dim"])
    d_node = torch.randn(4, TINY_DIMS["node_dim"])
    with torch.no_grad():
        _, score_before, _ = model(n_node, d_node)
        raw = model.authority_matrix.clone()
        row_sums = raw.sum(dim=1, keepdim=True)
        model.authority_matrix.copy_(raw / row_sums)
        _, score_after, _ = model(n_node, d_node)
        model.authority_matrix.copy_(raw)  # restore, this instance is discarded anyway
    displacement = float((score_before - score_after).abs().mean())
    return {"displacement": displacement, "fires": displacement >= S3A_DISPLACEMENT_THRESHOLD,
            "threshold": S3A_DISPLACEMENT_THRESHOLD}


def _build_gghe(seed):
    torch.manual_seed(seed)
    ns = fl.load(["GuidelineGroundedHarmEstimator"])
    return ns["GuidelineGroundedHarmEstimator"](**TINY_GGHE_DIMS, use_multitask=True)


def s3b_gradient_rms_ratio(seed, plant: bool):
    """Ratio of gradient RMS the rule-id (auxiliary) loss contributes to the
    shared `input_projection` weights, over the gradient RMS the
    contradiction (main) loss contributes to the same weights.

    plant=True: rule id target is an exact function of prior_vector's own
    rule-derivable feature (a shortcut the head can solve without the shared
    trunk). plant=False (clean, standing in for "a text-carried template
    family"): rule id target is independent of every engineered feature.
    """
    torch.manual_seed(seed)
    n = 6
    model = _build_gghe(seed)
    edge_repr = torch.randn(n, TINY_GGHE_DIMS["edge_dim"])
    prior = torch.randn(n, TINY_GGHE_DIMS["prior_dim"])
    rule_feature = torch.randint(0, 4, (n,))
    prior[:, 1] = rule_feature.float() / 4.0  # matches ACCORDv2Full._prior's own encoding
    rule_target = rule_feature if plant else torch.randint(0, 4, (n,))
    contra_target = torch.randint(0, 3, (n,))

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


def s3c_macro_f1_drop_on_zeroed_feature(seed, plant: bool, n_items=40):
    """Macro-F1 of rule_logits argmax vs. the rule-id target, feature present
    vs. zeroed, same (untrained, random-init) weights. plant=True: rule
    target is the exact function of the feature that gets zeroed (a real
    shortcut, so zeroing it should hurt); plant=False: rule target is
    independent of the feature (zeroing it should not matter)."""
    torch.manual_seed(seed)
    model = _build_gghe(seed).eval()
    edge_repr = torch.randn(n_items, TINY_GGHE_DIMS["edge_dim"])
    prior = torch.randn(n_items, TINY_GGHE_DIMS["prior_dim"])
    rule_feature = torch.randint(0, 4, (n_items,))
    prior[:, 1] = rule_feature.float() / 4.0
    rule_target = (rule_feature if plant else torch.randint(0, 4, (n_items,))).numpy()

    with torch.no_grad():
        c_logits, r_logits_present, harm, expl, attn = model(edge_repr, prior)
        pred_present = r_logits_present.argmax(dim=-1).numpy()
        prior_zeroed = prior.clone()
        prior_zeroed[:, 1] = 0.0
        c_logits2, r_logits_zeroed, *_ = model(edge_repr, prior_zeroed)
        pred_zeroed = r_logits_zeroed.argmax(dim=-1).numpy()

    f1_present = _macro_f1(rule_target, pred_present, TINY_GGHE_DIMS["num_rules"])
    f1_zeroed = _macro_f1(rule_target, pred_zeroed, TINY_GGHE_DIMS["num_rules"])
    drop = f1_present - f1_zeroed
    return {"macro_f1_present": f1_present, "macro_f1_zeroed": f1_zeroed, "drop": drop,
            "fires": drop >= S3C_MACRO_F1_DROP_THRESHOLD,
            "threshold": S3C_MACRO_F1_DROP_THRESHOLD, "plant": plant}


def s3d_harm_weight_change_ratio(seed, harm_in_loss: bool):
    """L2 norm of the change in harm_head's weights after one optimizer step,
    over the L2 norm of its initial weights. harm_in_loss=False is the
    plant ("harm head outside the loss": excluded from backward, so the
    optimizer step should leave it exactly unchanged); harm_in_loss=True is
    clean (included, so it should move)."""
    torch.manual_seed(seed)
    n = 6
    model = _build_gghe(seed)
    edge_repr = torch.randn(n, TINY_GGHE_DIMS["edge_dim"])
    prior = torch.randn(n, TINY_GGHE_DIMS["prior_dim"])
    contra_target = torch.randint(0, 3, (n,))
    harm_target = torch.rand(n, 1)

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
    results = {"s3a": [], "s3b_plant": [], "s3b_clean": [], "s3c_plant": [], "s3c_clean": [],
              "s3d_clean_harm_in_loss": [], "s3d_plant_harm_out_of_loss": []}
    for seed in SMOKE_SEEDS:
        results["s3a"].append(s3a_authority_displacement(seed))
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
                "dimensions. Full run (real dims, trained checkpoint, "
                "BiomedBERT feature cache) is deferred to the coordinator's "
                "MPS run; thresholds above are fixed and will not change."),
        "smoke_seeds": SMOKE_SEEDS, "dims": {**TINY_DIMS, **TINY_GGHE_DIMS},
        "all_outputs_finite": bool(all_finite),
        "results": results,
    }


def main():
    out = smoke_test()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
    print(json.dumps(out["results"], indent=2))


if __name__ == "__main__":
    main()
