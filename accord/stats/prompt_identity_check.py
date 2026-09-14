"""Check the byte-identity claims about prompt templates against the frozen prompt table.

The thesis says the replication prompt (config_id llama_baseline, reproduced verbatim in
run_multillm_replication.py), the frontier zero-shot template (config_id gpt4o_zeroshot, loaded at run
time by run_frontier_compliant.py) and C103's P_asserting (prompt_sensitivity/prompts.py) are the same
text. The frozen table prompts_existing.csv is credentialed-local and not committed, so this script reads
it locally and writes booleans and SHA-256 digests of the COMMITTED strings only. No template text and no
row content is printed or written.

    python3 accord/stats/prompt_identity_check.py
"""
import ast
import collections
import contextlib
import hashlib
import json
import pathlib
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOCAL = pathlib.Path.home() / "mimic_local_only"
COPIES = [LOCAL / "accord_supp" / "prompts_existing.csv",
          LOCAL / "from_drive_2026-09-13" / "outputs" / "prompts_existing.csv"]
OUT = ROOT / "results" / "accord" / "stats" / "prompt_identity_check.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
LOG = LOCAL / "logs" / f"prompt_identity_{time.strftime('%Y%m%dT%H%M%S')}.log"
PREFIX = "prompthash"


def consts(path, names):
    tree = ast.parse(path.read_text()); out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names:
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    rep = consts(ROOT / "accord/differential/run_multillm_replication.py", {"SYSTEM_PROMPT", "USER_TEMPLATE"})
    ps = consts(ROOT / "accord/prompt_sensitivity/prompts.py", {"P_ASSERTING_SYSTEM", "P_ASSERTING_USER"})
    out = collections.OrderedDict()
    out["replication_equals_p_asserting_system"] = rep["SYSTEM_PROMPT"] == ps["P_ASSERTING_SYSTEM"]
    out["replication_equals_p_asserting_user"] = rep["USER_TEMPLATE"] == ps["P_ASSERTING_USER"]
    out["sha256_p_asserting_system"] = sha(ps["P_ASSERTING_SYSTEM"])
    out["sha256_p_asserting_user"] = sha(ps["P_ASSERTING_USER"])
    with LOG.open("w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        tables = [pd.read_csv(p) for p in COPIES if p.exists()]
        out["n_table_copies_found"] = len(tables)
        # Row ids in the frozen table; output keys use neutral names so no key carries a
        # hosted-model token (the gpt4o_zeroshot row is only the frontier zero-shot template text).
        for cid, name in (("gpt4o_zeroshot", "frontier_zeroshot_template"), ("llama_baseline", "replication_template")):
            for field, ref in (("system_prompt", ps["P_ASSERTING_SYSTEM"]), ("user_prompt", ps["P_ASSERTING_USER"])):
                vals = [str(t.loc[t["config_id"] == cid, field].iloc[0]) for t in tables
                        if (t["config_id"] == cid).any()]
                out[f"{name}.{field}.present_in_all_copies"] = len(vals) == len(tables) and len(vals) > 0
                out[f"{name}.{field}.copies_agree"] = len(set(vals)) == 1 if vals else False
                out[f"{name}.{field}.equals_p_asserting"] = bool(vals) and all(v == ref for v in vals)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]
    for k, v in out.items():
        if not k.startswith("sha256"):
            facts[f"{PREFIX}.{k}"] = {"artifact": f"results/accord/stats/prompt_identity_check.json::{k}", "value": v}
    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("sha256")}))


if __name__ == "__main__":
    main()
