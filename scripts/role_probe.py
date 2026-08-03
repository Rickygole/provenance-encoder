"""Premise check: is note provenance recoverable from the text itself?

If a bag-of-words classifier recovers note type from role-marker-stripped text at
high accuracy, then a provenance-conditioned architecture is supplying a shortcut
the model would have extracted anyway. Any measured gain would then be a
convergence-speed effect that vanishes at convergence -- fatal in a
compute-matched regime. This is the cheapest experiment that can kill the thesis.

Runs entirely locally on MIMIC-III. Prints aggregate metrics only; no note text
is emitted, logged, or written outside local_only/.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit

NOTEEVENTS = os.environ.get("MIMIC_NOTEEVENTS_PATH", "data/NOTEEVENTS.csv.gz")
OUT_DIR = Path(__file__).resolve().parent.parent / "local_only"

# Categories that form multi-provenance bundles. Collapsed to the role taxonomy
# the encoder will use.
ROLE_MAP = {
    "Nursing": "nursing",
    "Nursing/other": "nursing",
    "Physician ": "physician",
    "Physician": "physician",
    "Discharge summary": "discharge",
}

# --- Level 1: explicit self-naming -----------------------------------------
# Strings by which a note announces its own type.
EXPLICIT = [
    r"\bnursing\b", r"\bnurse\b", r"\brn\b", r"\bnpn\b",
    r"\bphysician\b", r"\bresident\b", r"\battending\b", r"\bintensivist\b",
    r"\bmd\b", r"\bm\.d\.", r"\bdo\b", r"\bfellow\b",
    r"\bdischarge\s+summary\b", r"\bdischarge\b",
    r"\bprogress\s+note\b", r"\bgeneric\s+note\b", r"\bconsult\b",
    r"\bdictated\s+by\b", r"\bsigned\s+by\b", r"\bcompleted\s+by\b",
    r"\bjob#\b", r"\bcc\s*by\b",
]

# --- Level 2: structural headers characteristic of one role ----------------
# Discharge summaries carry admin scaffolding no progress note has; nursing
# notes use the Assessment/Action/Response shift format.
STRUCTURAL = [
    r"admission\s+date\s*:", r"discharge\s+date\s*:", r"date\s+of\s+birth\s*:",
    r"service\s*:", r"allergies\s*:", r"chief\s+complaint\s*:",
    r"major\s+surgical\s+or\s+invasive\s+procedure\s*:",
    r"history\s+of\s+present\s+illness\s*:", r"past\s+medical\s+history\s*:",
    r"medications\s+on\s+admission\s*:", r"facility\s*:",
    r"followup\s+instructions\s*:", r"follow-?up\s+instructions\s*:",
    r"^\s*action\s*:", r"^\s*response\s*:", r"^\s*assessment\s*:", r"^\s*plan\s*:",
    r"demographics", r"\bshift\b", r"\bicu\b",
]

DEID = re.compile(r"\[\*\*.*?\*\*\]")
WS = re.compile(r"\s+")
EXPLICIT_RE = re.compile("|".join(EXPLICIT), re.IGNORECASE)
STRUCTURAL_RE = re.compile("|".join(STRUCTURAL), re.IGNORECASE | re.MULTILINE)


def scrub(text: str, level: int) -> str:
    """Remove role-revealing surface markers. level 0=raw, 1=explicit, 2=+structural."""
    t = DEID.sub(" ", text)
    if level >= 1:
        t = EXPLICIT_RE.sub(" ", t)
    if level >= 2:
        t = STRUCTURAL_RE.sub(" ", t)
    return WS.sub(" ", t).strip()


def sample_notes(path: str, per_role: int, chunk: int, seed: int):
    """Reservoir-free balanced sample: take the first `per_role` notes per role,
    capped per subject so one patient cannot dominate a class."""
    rng = np.random.default_rng(seed)
    buckets: dict[str, list] = defaultdict(list)
    per_subject: dict[tuple[str, int], int] = defaultdict(int)
    MAX_PER_SUBJECT = 3

    reader = pd.read_csv(
        path,
        chunksize=chunk,
        usecols=["SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT", "ISERROR"],
        compression="gzip",
        low_memory=False,
    )
    for i, ch in enumerate(reader):
        ch = ch[ch["ISERROR"].isna()]
        ch = ch[ch["CATEGORY"].isin(ROLE_MAP)]
        if ch.empty:
            continue
        # shuffle within chunk so we don't take a time-ordered prefix
        ch = ch.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        for sid, cat, txt in zip(ch["SUBJECT_ID"], ch["CATEGORY"], ch["TEXT"]):
            role = ROLE_MAP[cat]
            if len(buckets[role]) >= per_role:
                continue
            if not isinstance(txt, str) or len(txt) < 200:
                continue
            key = (role, int(sid))
            if per_subject[key] >= MAX_PER_SUBJECT:
                continue
            per_subject[key] += 1
            buckets[role].append((int(sid), txt))
        if all(len(v) >= per_role for v in buckets.values()) and len(buckets) == len(set(ROLE_MAP.values())):
            break
        if i % 20 == 0:
            got = {k: len(v) for k, v in buckets.items()}
            print(f"  ...chunk {i} collected {got}", flush=True)
    return buckets


def length_matched(buckets, seed: int, n_bins: int = 10):
    """Resample so every role has the same character-length distribution.

    Discharge summaries average 9,619 chars against 799 for Nursing/other -- a
    12x gap. Any classifier can exploit that without learning anything about
    authority. This restricts to the overlapping length range and equalizes
    counts per decile, so length carries no signal by construction.
    """
    rng = np.random.default_rng(seed)
    lens = {r: np.array([len(t) for _, t in items]) for r, items in buckets.items()}
    lo = max(float(v.min()) for v in lens.values())
    hi = min(float(v.max()) for v in lens.values())
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, n_bins + 1)

    binned: dict[str, list[list[int]]] = {}
    for role, items in buckets.items():
        per_bin: list[list[int]] = [[] for _ in range(n_bins)]
        for idx, (_, txt) in enumerate(items):
            L = len(txt)
            if L < lo or L > hi:
                continue
            b = min(int(np.searchsorted(edges, L, side="right")) - 1, n_bins - 1)
            per_bin[max(b, 0)].append(idx)
        binned[role] = per_bin

    out: dict[str, list] = {r: [] for r in buckets}
    for b in range(n_bins):
        k = min(len(binned[r][b]) for r in buckets)
        if k == 0:
            continue
        for role in buckets:
            pick = rng.choice(binned[role][b], size=k, replace=False)
            out[role].extend(buckets[role][i] for i in pick)
    return out if all(len(v) > 50 for v in out.values()) else None


def length_only_baseline(buckets, seed: int) -> dict:
    """Predict role from document length alone. If this is high, the corpus has a
    length confound strong enough to explain any 'authority' result."""
    X, y, g = [], [], []
    for role, items in buckets.items():
        for sid, txt in items:
            L = len(txt)
            X.append([np.log1p(L), L / 1000.0])
            y.append(role)
            g.append(sid)
    X, y, g = np.array(X), np.array(y), np.array(g)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(gss.split(X, y, g))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    return {
        "accuracy": float(accuracy_score(y[te], pred)),
        "macro_f1": float(f1_score(y[te], pred, average="macro")),
        "n_test": int(len(te)),
    }


def run_level(buckets, level: int, seed: int) -> dict:
    texts, labels, groups = [], [], []
    for role, items in buckets.items():
        for sid, txt in items:
            texts.append(scrub(txt, level))
            labels.append(role)
            groups.append(sid)
    texts = np.array(texts, dtype=object)
    labels = np.array(labels)
    groups = np.array(groups)

    # Group-disjoint split by SUBJECT_ID. ACCORD used a row-level random split;
    # that is the contamination bug the reviewers caught.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(gss.split(texts, labels, groups))
    assert not (set(groups[tr]) & set(groups[te])), "subject leaked across split"

    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xtr = vec.fit_transform(texts[tr])
    Xte = vec.transform(texts[te])

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr, labels[tr])
    pred = clf.predict(Xte)

    acc = accuracy_score(labels[te], pred)
    macro_f1 = f1_score(labels[te], pred, average="macro")
    report = classification_report(labels[te], pred, output_dict=True, zero_division=0)
    return {
        "level": level,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": {k: float(v["f1-score"]) for k, v in report.items() if isinstance(v, dict)},
        "vocab": int(len(vec.vocabulary_)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=NOTEEVENTS)
    ap.add_argument("--per-role", type=int, default=6000)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    print("Sampling notes (local only, no text emitted)...", flush=True)
    buckets = sample_notes(args.path, args.per_role, args.chunk, args.seed)
    print("\nSampled:", {k: len(v) for k, v in buckets.items()}, flush=True)
    if len(buckets) < 2:
        sys.exit("not enough roles sampled")

    lb = length_only_baseline(buckets, args.seed)
    print(f"\nLENGTH-ONLY baseline: acc={lb['accuracy']:.4f} macro_f1={lb['macro_f1']:.4f}"
          f"  (document length is the ONLY feature)", flush=True)

    results = []
    for level in (0, 1, 2):
        r = run_level(buckets, level, args.seed)
        results.append(r)
        print(
            f"\nlevel {level}  acc={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}  "
            f"vocab={r['vocab']:,}  (train {r['n_train']:,} / test {r['n_test']:,})",
            flush=True,
        )
        for k, v in sorted(r["per_class_f1"].items()):
            if k in ("accuracy",):
                continue
            print(f"    {k:<16} f1={v:.4f}")

    # Scrubbed AND length-matched: the hardest condition, and the one that
    # actually tests whether provenance carries signal beyond surface artifacts.
    lm_result = None
    lm = length_matched(buckets, args.seed)
    if lm is not None:
        print(f"\nlength-matched subsample: {[(k, len(v)) for k, v in lm.items()]}", flush=True)
        lm_lb = length_only_baseline(lm, args.seed)
        print(f"  length-only on matched set: acc={lm_lb['accuracy']:.4f} (should be ~chance)")
        lm_result = run_level(lm, 2, args.seed)
        lm_result["length_only_after_matching"] = lm_lb["accuracy"]
        print(f"  scrubbed + length-matched : acc={lm_result['accuracy']:.4f} "
              f"macro_f1={lm_result['macro_f1']:.4f}")
    else:
        print("\nlength matching failed (insufficient overlap)", flush=True)

    payload = {"length_only": lb, "levels": results, "length_matched_level2": lm_result}
    out = OUT_DIR / "role_probe_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")

    a0, a2 = results[0]["accuracy"], results[2]["accuracy"]
    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    print(f"  length only         : {lb['accuracy']:.1%}")
    print(f"  raw text            : {a0:.1%}")
    print(f"  fully scrubbed      : {a2:.1%}")
    print(f"  drop from scrubbing : {a0 - a2:+.1%}")
    if lm_result:
        print(f"  scrubbed + len-matched: {lm_result['accuracy']:.1%}")
    if lb["accuracy"] > 0.80:
        print(f"\n  Document length ALONE reaches {lb['accuracy']:.1%}. The corpus has a")
        print("  length confound strong enough to explain an 'authority' result on its")
        print("  own. Length-matched evaluation is mandatory, not optional.")
    if a2 > 0.95:
        print("\n  >95% after scrubbing. Provenance is textually redundant.")
        print("  A provenance-conditioned architecture supplies a shortcut, not new")
        print("  information. Expect convergence-speed gains only. PREMISE FAILS.")
    elif a2 > 0.80:
        print("\n  80-95%. Mostly redundant. Any architectural gain will be small and")
        print("  must be demonstrated at convergence, not at fixed step count.")
    else:
        print("\n  <80% after scrubbing. Provenance carries non-redundant signal.")
        print("  The premise survives; proceed to the A1-A4 arms.")


if __name__ == "__main__":
    main()
