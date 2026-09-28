"""Register the four unregistered C134 reanalysis artifacts under c134b.*, c134c.*,
c134d.* and c134f.*

c134a_bootstrap_coverage.json was registered by
accord/stats/register_bootstrap_coverage.py; its four siblings were not, so none
of their numbers could be printed without failing pass 2 of verify_thesis.py.
The four are:

  c134b  per-fold AUROC for the gold-only nested cross-validation whose pooled
         out-of-fold AUROC Chapter 4 reports as 0.3434
  c134c  every three-seed standard deviation the thesis reports, recomputed under
         both the population and the sample convention
  c134d  the threshold sweep on the decontaminated ensemble, the one the chapter
         still records as not run
  c134f  the decomposition of the pooled 0.3434 into the pooling and the estimator

Each prefix is deleted wholesale and rewritten from its own committed artifact,
the convention every generator in this directory follows. Only numeric and
boolean leaves are registered; string leaves are prose describing the design and
stay in the artifact. Lists of numbers are registered element by element and
lists of objects are descended into by index, the shape
testfamrecount.entries.14.p already uses.

Run:
    python3 accord/stats/register_c134_reanalysis.py
"""

import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
C134 = ROOT / "results" / "accord" / "c134_reanalysis"

# leaf names that are prose, a path, or a timing, in any of the four artifacts
SKIP_LEAF_NAMES = {
    "what", "closes", "targets_claim", "addresses", "verdict", "statement",
    "description", "why", "caveat", "answer", "granularity_per_fold", "note",
    "notes", "source", "runtime_seconds", "source_artifact", "reading",
    "direction", "sentence_the_thesis_should_carry", "stratum", "quantity",
    "printed_in_thesis", "printed_where", "sd_ddof0_rounded", "sd_ddof1_rounded",
    "error", "conventions_found", "convention_the_printed_value_used",
    "column_alignment",
}

TARGETS = [
    ("c134b", C134 / "c134b_perfold_auroc.json"),
    ("c134c", C134 / "c134c_seed_sd_ddof1.json"),
    ("c134d", C134 / "c134d_decontam_threshold_sweep.json"),
    ("c134f", C134 / "c134f_pooling_vs_estimator.json"),
]


def sanitize(key):
    return (str(key).replace("::", "_").replace(":", "").replace(" ", "_")
            .replace("(", "").replace(")", "").replace(",", ""))


def flatten(prefix, node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in SKIP_LEAF_NAMES:
                continue
            sub = sanitize(k)
            flatten(f"{prefix}.{sub}" if prefix else sub, v, out)
    elif isinstance(node, list):
        for i, x in enumerate(node):
            flatten(f"{prefix}.{i}", x, out)
    elif isinstance(node, bool) or isinstance(node, (int, float)):
        out[prefix] = node


def check(prefix, artifact):
    """Assertions that fail loudly if an artifact is not the one this expects."""
    if prefix == "c134b":
        rep = artifact["reproduction_of_the_published_pooled_figures"]
        assert rep["matched_to_10_decimals"] is True, rep
        assert abs(rep["pooled_out_of_fold_auroc_recomputed"] - 0.3433609958506224) < 1e-12
        assert artifact["per_fold_summary"]["n_folds"] == 12
    elif prefix == "c134c":
        assert artifact["n_quantities"] == 44, artifact["n_quantities"]
        assert artifact["n_printed_values_that_change_under_ddof1"] == 11, artifact
        for r in artifact["rows"]:
            if "sd_ddof0" in r and r["sd_ddof0"] > 0:
                assert abs(r["ddof1_over_ddof0_ratio"] - artifact["sqrt_3_over_2"]) < 1e-9, r
    elif prefix == "c134d":
        chk = artifact["comparability_check_against_released_run"]
        assert chk["p_reproduces"] is True and chk["null_median_reproduces"] is True, chk
        g = artifact["geometry"]
        assert g["n"] == 253 and g["n_positives"] == 12, g
    elif prefix == "c134f":
        h = artifact["how_much_is_the_pooling"]
        assert h["pooled_minus_mean_per_fold"] > 0 and \
            h["pooled_minus_rank_normalised_pooled"] > 0, h


def main():
    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    total = 0
    for prefix, path in TARGETS:
        artifact = json.loads(path.read_text())
        check(prefix, artifact)
        leaves = {}
        flatten("", artifact, leaves)

        for k in [k for k in facts if k.split(".")[0] == prefix]:
            del facts[k]

        rel = path.relative_to(ROOT)
        for dotted, value in sorted(leaves.items()):
            facts[f"{prefix}.{dotted}"] = collections.OrderedDict(
                [("artifact", f"{rel}::{dotted}"), ("value", value)])
        print(f"registered {len(leaves)} keys under {prefix}.*")
        total += len(leaves)

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"{total} keys written, register now holds {len(facts)}")


if __name__ == "__main__":
    main()
