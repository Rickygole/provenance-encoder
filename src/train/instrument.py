"""Assertions that fail runs. Every one encodes a specific prior failure.

These raise rather than warn. A run that silently produces an uninterpretable
number is worse than a run that stops -- that is the whole lesson of ACCORD,
where a parameter frozen at its initialization was reported as a discovery.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


class InputVarianceError(RuntimeError):
    """A categorical input is constant, so its embedding can only emit a bias."""


class DeadParameterError(RuntimeError):
    """A parameter being interpreted never moved from its initialization."""


class InertMechanismError(RuntimeError):
    """The advertised mechanism has no effect on the attention distribution."""


class GaugeError(RuntimeError):
    """The bias table is not in the canonical (row-centered) gauge."""


class Instrumentor:
    def __init__(self, model, run_dir: str | Path, arm: str):
        self.model = model
        self.arm = arm
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.dir / "metrics.jsonl"
        self.init = {k: v.detach().clone() for k, v in model.provenance_params().items()}
        self.grad_sums = {k: 0.0 for k in self.init}
        self.grad_counts = {k: 0 for k in self.init}

    # --- I1 ---------------------------------------------------------------
    def check_inputs(self, batch: dict) -> dict:
        role = batch["role_ids"].clone()
        role[batch["attention_mask"] == 0] = 0
        per_seq = [len(set(r[r > 0].tolist())) for r in role]
        multi_frac = sum(1 for n in per_seq if n >= 2) / max(len(per_seq), 1)
        t = batch["event_times"][batch["attention_mask"] > 0]
        t_std = float(t.std()) if t.numel() > 1 else 0.0

        if max(per_seq, default=0) < 2 and self.arm in ("A3", "A4"):
            raise InputVarianceError(
                "no sequence in this batch has 2+ distinct roles; the authority "
                "bias is provably inert on this input (see test_invariants.py)"
            )
        if self.arm == "A4" and t_std == 0.0:
            raise InputVarianceError(
                "event_times has zero variance; the temporal bias can only emit a "
                "constant. ACCORD hardcoded these inputs for every example."
            )
        return {"multi_prov_frac": multi_frac, "mean_distinct_roles": sum(per_seq) / max(len(per_seq), 1),
                "event_time_std": t_std}

    # --- I3 ---------------------------------------------------------------
    @torch.no_grad()
    def check_bias_effective(self, batch: dict) -> dict:
        bm = getattr(self.model, "bias_module", None)
        if bm is None:
            return {}
        bias = bm(batch["role_ids"], batch["event_times"] if self.model.cfg.use_time_bias else None)
        valid = batch["attention_mask"].bool()
        spread = []
        for b in range(bias.shape[0]):
            v = valid[b]
            if v.sum() < 2:
                continue
            spread.append(bias[b][:, :, v][:, v, :].std().item())
        eff = sum(spread) / max(len(spread), 1)
        return {"eff_bias_std": eff}

    # --- I4 ---------------------------------------------------------------
    @torch.no_grad()
    def check_gauge(self) -> None:
        bm = getattr(self.model, "bias_module", None)
        if bm is None:
            return
        m = bm.centered_authority().mean(dim=-1).abs().max().item()
        if m > 1e-5:
            raise GaugeError(f"authority not row-centered (max |rowmean| = {m:.2e})")

    # --- I2 ---------------------------------------------------------------
    def accumulate_grads(self) -> None:
        for k, p in self.model.provenance_params().items():
            if p.grad is not None:
                self.grad_sums[k] += float(p.grad.norm())
                self.grad_counts[k] += 1

    @torch.no_grad()
    def movement(self) -> dict:
        out = {}
        for k, p in self.model.provenance_params().items():
            init = self.init[k]
            delta = float((p.detach() - init).norm())
            denom = float(init.norm())
            out[k] = {
                "abs_movement": delta,
                # Zero-init means relative movement is undefined; report absolute.
                "rel_movement": (delta / denom) if denom > 1e-12 else None,
                "final_norm": float(p.detach().norm()),
                "mean_grad_norm": self.grad_sums[k] / max(self.grad_counts[k], 1),
            }
        return out

    def assert_alive(self, frac_done: float) -> None:
        """After enough training, an interpreted parameter must have moved."""
        if frac_done < 0.10:
            return
        for k, m in self.movement().items():
            if m["abs_movement"] < 1e-4 and m["mean_grad_norm"] < 1e-8:
                raise DeadParameterError(
                    f"{k}: moved {m['abs_movement']:.2e} with mean grad norm "
                    f"{m['mean_grad_norm']:.2e}. It is frozen at initialization and "
                    f"may not be interpreted. This is the ACCORD alpha failure."
                )

    def log(self, step: int, **kw) -> None:
        rec = {"step": step, **kw}
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def final_report(self) -> dict:
        rep = {"arm": self.arm, "provenance_params": self.movement()}
        (self.dir / "provenance_report.json").write_text(json.dumps(rep, indent=2))
        return rep
