"""C133: interval tipping point. How many greedily chosen label changes does it
take before the UPPER END of the row-level percentile bootstrap interval, the
quantity the thesis's bounded null is defined by, reaches the registered power
floor 0.7466? C114 task B counted flips until the POINT ESTIMATE reached a
floor; this check measures fragility by the criterion the claim itself uses.
See C133_PREREGISTRATION.md, committed before this script's full mode was run.

No model is trained or scored. The per-row scores and gold labels are loaded
by C132's own loading code (c132_run.load_inputs, load_registered and
input_paths are imported, not re-implemented), so the score files, the labels,
the row orders and the join gate are identical to C132's. No identifier,
pair_id or per-row value is written, printed or logged anywhere, and a flip is
recorded by its step number only.

    python3 accord/c133_interval_tipping/c133_run.py --dry-run
    python3 accord/c133_interval_tipping/c133_run.py            # full run

Local only. No hosted API, no Ollama call. Everything that touches
~/mimic_local_only runs with stdout/stderr redirected to a file under
~/mimic_local_only/logs; that log is never read back. --dry-run checks that
every input exists, that the registered values are present and that C132's
join gate passes; it computes no AUROC, no flip and no interval, writes nothing
to results/, and prints counts and booleans only. The full run writes exactly
one aggregate JSON, results/accord/c133_interval_tipping/c133.json.
register_c133.py adds the c133.* register keys from that artifact.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import pathlib
import sys
import time
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accord" / "c132_cluster_bootstrap"))

from accord.cloud_paths import assert_local_only  # noqa: E402
# C132's loading code, imported rather than copied, so this run cannot drift
# onto a different score file, label column, row order or gate. Importing
# c132_run also puts accord/control, accord/stats and the C130 directory on
# sys.path, which its load_inputs() relies on. It opens no file at import.
import c132_run as c132  # noqa: E402
from c132_run import AbortC, N_G253, N_POS_G253, N_NEG_G253  # noqa: E402
from recompute_mde import hanley_mcneil_mde  # noqa: E402

MIMIC_LOCAL_ONLY_DIR = pathlib.Path(os.environ.get("MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
LOG_DIR = MIMIC_LOCAL_ONLY_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = assert_local_only(LOG_DIR / f"c133_run_{time.strftime('%Y%m%dT%H%M%S')}.log")
RESULTS_OUT = assert_local_only(ROOT / "results" / "accord" / "c133_interval_tipping" / "c133.json")

N_BOOT = 10_000
SEED = 42                 # the seed both registered row-level intervals were recorded with
MAX_ATTEMPT_FACTOR = 20   # as accord/control/report.py::bootstrap_auroc_ci and C132
MAX_FLIPS = 8
MIN_COUNT_FOR_OUTCOME_A = 3
POOL_CHUNK = 2_048        # attempts drawn per call; does not change the stream (see DrawPool)

ENSEMBLES = (("decontam_ensemble", "decontam"), ("c130_ensemble", "c130"))
TASKS = (("task_a", "pos_to_neg"), ("task_b", "neg_to_pos"))


# ---------------------------------------------------------------------------
# Exact AUROC from a pairwise kernel.
# ---------------------------------------------------------------------------

def pair_kernel(s) -> np.ndarray:
    """K2[i, j] = 2 if s[i] > s[j], 1 if s[i] == s[j], else 0. With row i read
    as a positive and row j as a negative, sum(K2) / (2 * n_pos * n_neg) is the
    AUROC with ties counted one half, the Mann-Whitney form roc_auc_score
    computes. Held as float64 so the products below go through BLAS; every
    value is a small integer, so the arithmetic is exact."""
    s = np.asarray(s, dtype=float)
    return 2.0 * (s[:, None] > s[None, :]) + 1.0 * (s[:, None] == s[None, :])


def u2_statistic(y, K2) -> float:
    """Twice the Mann-Whitney U of labelling y: an exact integer in float64."""
    yf = np.asarray(y, dtype=float)
    return float(yf @ K2 @ (1.0 - yf))


def auroc(y, K2) -> float:
    n_pos = int(np.sum(y))
    n_neg = int(len(y) - n_pos)
    return u2_statistic(y, K2) / (2.0 * n_pos * n_neg)


# ---------------------------------------------------------------------------
# The bootstrap, vectorised, draw for draw the registered row-level bootstrap.
# ---------------------------------------------------------------------------

class DrawPool:
    """The first M attempts of the registered row-level bootstrap's index
    stream, held as an M x n matrix of row multiplicities.

    recompute_decontam_aggregate.py::boot_default_rng and C132's row-level
    check call rng.integers(0, n, n) once per attempt on
    numpy.random.default_rng(seed). The index stream does not depend on the
    labels or the scores, so a fresh generator with the same seed gives every
    interval the same attempts; only which attempts are single-class, and so
    redrawn, depends on the labelling. rng.integers(0, n, size=(m, n)) fills
    row by row from the same stream as m successive size-n calls, so the pool
    is drawn in chunks once and shared by all intervals, which is what makes
    2 ensembles x 2 directions x 8 steps x 10,000 resamples run in minutes.
    The k = 0 validity check is the test that this is so on the real inputs:
    it must reproduce both registered intervals to four decimals."""

    def __init__(self, n: int, seed: int = SEED):
        self.n = int(n)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros((0, self.n), dtype=float)

    def first(self, m: int) -> np.ndarray:
        while len(self.counts) < m:
            idx = self.rng.integers(0, self.n, size=(POOL_CHUNK, self.n))
            flat = (idx + np.arange(POOL_CHUNK)[:, None] * self.n).ravel()
            chunk = np.bincount(flat, minlength=POOL_CHUNK * self.n).reshape(POOL_CHUNK, self.n)
            self.counts = np.vstack([self.counts, chunk.astype(float)])
        return self.counts[:m]


def percentile_interval(y, K2, pool: DrawPool, B: int = N_BOOT) -> dict:
    """Row-level percentile bootstrap 95 percent interval of the AUROC of
    labelling y: the first B attempts of the pool that hold both classes, in
    stream order. An attempt with no positive or no negative is redrawn and
    counted, exactly as the registered bootstrap skips it."""
    yf = np.asarray(y, dtype=float)
    n = len(yf)
    cap = B * MAX_ATTEMPT_FACTOR
    m = B + POOL_CHUNK
    while True:
        counts = pool.first(min(m, cap))
        n_pos_b = counts @ yf
        valid = np.flatnonzero((n_pos_b > 0) & (n_pos_b < n))
        if len(valid) >= B:
            break
        if m >= cap:
            raise AbortC("bootstrap attempt cap reached before B valid resamples")
        m += POOL_CHUNK
    valid = valid[:B]
    attempts = int(valid[-1]) + 1
    c = counts[valid]
    c_pos, c_neg = c * yf, c * (1.0 - yf)
    u2 = ((c_pos @ K2) * c_neg).sum(axis=1)
    aucs = u2 / (2.0 * c_pos.sum(axis=1) * c_neg.sum(axis=1))
    return collections.OrderedDict([
        ("seed", pool.seed), ("n_boot_valid", int(len(aucs))),
        ("n_redrawn_single_class", int(attempts - B)), ("redraw_rate", float((attempts - B) / attempts)),
        ("ci95_lo", float(np.percentile(aucs, 2.5))), ("ci95_hi", float(np.percentile(aucs, 97.5))),
    ])


# ---------------------------------------------------------------------------
# The greedy relabelling. Steps are numbered; no row identity leaves here.
# ---------------------------------------------------------------------------

def greedy_step(y, K2, direction: str):
    """Relabel the single row of the source class whose relabelling gives the
    highest AUROC of the resulting labelling. Every candidate leaves the same
    class counts, so the AUROCs are compared through their exact integer
    2U statistics; np.argmax returns the first maximum, and the candidates are
    in ascending row position, so ties go to the lowest row position in the
    ensemble's own row order (the rule C114's strict > comparison also gives).
    Returns (new labelling, number of candidates tied for the best)."""
    source = 1 if direction == "pos_to_neg" else 0
    candidates = np.flatnonzero(np.asarray(y) == source)
    if len(candidates) <= 1:
        raise AbortC("a relabelling step would leave a class empty")
    # 2U of every candidate labelling in closed form, exact integers in float64. Moving row r across
    # removes the pairs it formed on its old side and adds the pairs it forms on its new side; the
    # -1 is K2[r, r], the row's tie with itself, which is a pair in neither labelling.
    yf = np.asarray(y, dtype=float)
    row_over_negatives = K2 @ (1.0 - yf)   # [r] = sum over current negatives j of K2[r, j]
    positives_over_col = yf @ K2           # [r] = sum over current positives i of K2[i, r]
    sign = 1.0 if source == 1 else -1.0
    u2 = (u2_statistic(y, K2)
          + sign * (positives_over_col[candidates] - row_over_negatives[candidates]) - 1.0)
    best = int(np.argmax(u2))
    out = y.copy()
    out[candidates[best]] = 1 - source
    return out, int((u2 == u2[best]).sum())


def tipping_path(y0, K2, pool: DrawPool, direction: str, floor: float) -> dict:
    steps = collections.OrderedDict()
    y = np.asarray(y0).astype(int).copy()
    first_k = None
    for k in range(1, MAX_FLIPS + 1):
        y, n_tied = greedy_step(y, K2, direction)
        step = collections.OrderedDict([
            ("k", k), ("n_pos", int(y.sum())), ("n_neg", int(len(y) - y.sum())),
            ("auroc", float(auroc(y, K2))), ("n_candidates_tied_for_best", n_tied),
        ])
        step.update(percentile_interval(y, K2, pool))
        step["upper_end_at_or_above_floor"] = bool(step["ci95_hi"] >= floor)
        if first_k is None and step["upper_end_at_or_above_floor"]:
            first_k = k
        steps[f"k{k}"] = step
    return collections.OrderedDict([
        ("direction", direction),
        ("reached_within_max_flips", bool(first_k is not None)),
        ("k_first_upper_end_at_or_above_floor", first_k if first_k is not None else f"not reached within {MAX_FLIPS}"),
        ("steps", steps),
    ])


def floors_neg_to_pos() -> dict:
    """Information only: the analytic minimum detectable AUROC at the class
    counts each negative-to-positive flip leaves. Arithmetic on fixed counts,
    no data; the function is imported, not re-implemented, and rounded as
    recompute_mde.py and C132 task C round it."""
    out = collections.OrderedDict([("alpha", 0.05), ("power", 0.80), ("method", "accord/stats/recompute_mde.py::hanley_mcneil_mde")])
    for k in range(1, MAX_FLIPS + 1):
        n_pos, n_neg = N_POS_G253 + k, N_NEG_G253 - k
        mde = round(hanley_mcneil_mde(n_pos, n_neg), 4)
        out[f"k{k}"] = collections.OrderedDict([
            ("n_pos", n_pos), ("n_neg", n_neg), ("minimum_detectable_auc", mde), ("gap_above_chance", round(mde - 0.5, 4)),
        ])
    return out


# ---------------------------------------------------------------------------
# Modes. Everything below load_inputs() runs under the log redirect.
# ---------------------------------------------------------------------------

def gate_ok(gate: dict) -> bool:
    # C132's join gate: both score sources cover the same 253 unique pairs, labels agree with the
    # pharmacist label, none ambiguous, 12 positives and 241 negatives in each.
    return bool(gate["cluster_join_covers_all_253_pairs"])


def run_dry_run() -> dict:
    paths = c132.input_paths()
    exists = collections.OrderedDict((name, bool(p.is_file())) for name, p in paths.items())
    out = collections.OrderedDict([("inputs_exist", exists), ("all_inputs_exist", bool(all(exists.values())))])
    if not out["all_inputs_exist"]:
        out["overall"] = "BLOCKED_MISSING_INPUT"
        return out
    c132.load_registered()
    out["registered_values_present_and_round_to_preregistered"] = True
    _, _, _, gate = c132.load_inputs()
    out["gate"] = gate
    out["max_flips"] = MAX_FLIPS
    out["task_a_n_pos_left_after_max_flips"] = int(gate["decontam_n_pos"] - MAX_FLIPS)
    out["task_b_n_neg_left_after_max_flips"] = int(gate["decontam_n_neg"] - MAX_FLIPS)
    out["overall"] = "READY_TO_RUN" if gate_ok(gate) else "WOULD_ABORT_OUTCOME_C"
    return out


def run_full() -> dict:
    registered = c132.load_registered()
    missing = [name for name, p in c132.input_paths().items() if not p.is_file()]
    if missing:
        raise AbortC(f"input file(s) missing: {', '.join(missing)}")
    ens_i, ens_ii, _, gate = c132.load_inputs()
    if not gate_ok(gate):
        raise AbortC("the two score sources do not cover the same 253 non-ambiguous pairs with 12 positives and 241 negatives")
    floor = registered["power_floor"]
    ensembles = {"decontam_ensemble": ens_i, "c130_ensemble": ens_ii}

    pool = DrawPool(N_G253)
    kernels = {name: pair_kernel(ensembles[name]["s"]) for name, _ in ENSEMBLES}

    # Validity check first: k = 0, same function, same pool. Nothing is relabelled unless both reproduce.
    validity = collections.OrderedDict()
    for name, reg_prefix in ENSEMBLES:
        y, K2 = ensembles[name]["y"], kernels[name]
        k0 = percentile_interval(y, K2, pool)
        reg_lo, reg_hi = registered[f"{reg_prefix}_ci95_lo"], registered[f"{reg_prefix}_ci95_hi"]
        a = float(auroc(y, K2))
        validity[name] = collections.OrderedDict([
            ("k", 0), ("n", int(len(y))), ("n_pos", int(y.sum())), ("n_neg", int(len(y) - y.sum())),
            ("auroc", a),
            ("auroc_reproduces_registered_4dp", bool(round(a, 4) == round(registered[f"{reg_prefix}_auroc"], 4))),
        ])
        validity[name].update(k0)
        validity[name]["registered_ci95_lo"], validity[name]["registered_ci95_hi"] = reg_lo, reg_hi
        validity[name]["reproduces_registered_4dp"] = bool(
            round(k0["ci95_lo"], 4) == round(reg_lo, 4) and round(k0["ci95_hi"], 4) == round(reg_hi, 4)
        )
        validity[name]["upper_end_at_or_above_floor"] = bool(k0["ci95_hi"] >= floor)
    if not all(v["reproduces_registered_4dp"] for v in validity.values()):
        raise AbortC("k = 0 check did not reproduce the registered intervals to four decimals")

    out = collections.OrderedDict()
    out["preregistration"] = "accord/c133_interval_tipping/C133_PREREGISTRATION.md"
    out["n_boot"] = N_BOOT
    out["seed"] = SEED
    out["max_flips"] = MAX_FLIPS
    out["power_floor"] = floor
    out["validity_check"] = validity

    for task, direction in TASKS:
        out[task] = collections.OrderedDict(
            (name, tipping_path(ensembles[name]["y"], kernels[name], pool, direction, floor)) for name, _ in ENSEMBLES
        )

    counts = collections.OrderedDict(
        (f"k_{task[-1]}_{name}", out[task][name]["k_first_upper_end_at_or_above_floor"])
        for task, _ in TASKS for name, _ in ENSEMBLES
    )
    below = [key for key, v in counts.items() if isinstance(v, int) and v < MIN_COUNT_FOR_OUTCOME_A]
    reached = [v for v in counts.values() if isinstance(v, int)]
    out["decision"] = collections.OrderedDict([
        ("outcome", "B" if below else "A"),
        ("power_floor", floor),
        ("min_count_for_outcome_a", MIN_COUNT_FOR_OUTCOME_A),
        ("counts", counts),
        ("n_counts_below_min", len(below)),
        ("counts_below_min", below),
        ("smallest_count", int(min(reached)) if reached else f"not reached within {MAX_FLIPS}"),
    ])
    out["floors_neg_to_pos"] = floors_neg_to_pos()
    out["gate"] = gate
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    failure = None
    with LOG_PATH.open("w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        print(f"c133_run starting {time.strftime('%Y-%m-%d %H:%M:%S')} dry_run={args.dry_run}")
        try:
            result = run_dry_run() if args.dry_run else run_full()
        except AbortC as e:
            print(f"ABORT (Outcome C): {e}")
            failure = ("C", str(e))
        except Exception as e:  # traceback goes to the local log only; real stderr gets the type name
            traceback.print_exc()
            failure = ("ERROR", type(e).__name__)
        print("c133_run finished")

    if failure and failure[0] == "ERROR":
        raise SystemExit(f"c133_run failed with {failure[1]}; traceback is in the local log (not read back): {LOG_PATH}")
    if failure:
        if not args.dry_run:
            RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
            RESULTS_OUT.write_text(json.dumps(collections.OrderedDict([
                ("outcome", "C"), ("abort_reason", failure[1]), ("floors_neg_to_pos", floors_neg_to_pos())]), indent=2) + "\n")
        raise SystemExit(f"ABORT (Outcome C): {failure[1]}")

    if args.dry_run:
        print(json.dumps(result, indent=2))
        print(f"log: {LOG_PATH}")
        return

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {RESULTS_OUT}")
    print(f"log: {LOG_PATH}")
    print(json.dumps({"decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
