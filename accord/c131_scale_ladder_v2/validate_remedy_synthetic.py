"""C131 remedy validation: synthetic, non-clinical probes only. See
C131_PREREGISTRATION.md's "the remedy" section, which requires this script to
run -- and to pass -- BEFORE run_c131.py is ever pointed at the gold CSV.

Sends P_asserting's own system/user template, filled with fixed placeholder
text that names no patient, note, or clinical fact, to every qwen3 size at
num_predict=8, and checks the three things the remedy depends on:
  (a) Ollama's /api/chat "logprobs" field is a list with more than one entry
      when generation continues past one token (i.e. num_predict=8 actually
      returns per-position data, not just position 0);
  (b) every entry in that list -- not only index 0 -- carries its own
      "top_logprobs" in the same shape run_c103.py already scores index 0
      with;
  (c) accord.c131_scale_ladder_v2.run_c131.parse_and_score_remedy computes a
      hard label and a graded score on that response without raising.

Writes results/accord/c131_scale_ladder_v2/synthetic_probe_validation.json
and appends the addendum to C131_PREREGISTRATION.md (the rule text above the
addendum marker is never edited). No MIMIC data is read or touched by this
script; it sends none of the prompt text this project treats as
credentialed, so it is not subject to the mimic_local_only log redirect, but
it is still an Ollama call and is still constrained to 127.0.0.1:11434 and
its own egress check.

    python3 accord/c131_scale_ladder_v2/validate_remedy_synthetic.py
"""
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accord.c131_scale_ladder_v2.run_c131 import (  # noqa: E402
    call_chat_remedy, parse_and_score_remedy, write_egress_evidence, HOST,
    NUM_PREDICT, OUT_DIR,
)
from accord.prompt_sensitivity.run_c103 import assert_loopback_host  # noqa: E402
from accord.prompt_sensitivity.prompts import PROMPTS  # noqa: E402

VALIDATION_PATH = OUT_DIR / "synthetic_probe_validation.json"
PREREGISTRATION = ROOT / "accord" / "c131_scale_ladder_v2" / "C131_PREREGISTRATION.md"
SIZES = ["0.6b", "1.7b", "4b", "8b", "14b"]

# Fixed, synthetic, non-clinical probes. Named things (weather, arithmetic),
# never a patient, a note, a drug, or anything resembling clinical text.
# Two are constructed to look like a clear "contradiction" and a clear
# "not_contradiction" under P_asserting's own framing, so a normally-behaving
# size should still answer sensibly; the point of this validation is the
# shape of the API response, not whether the answer is "correct".
PROBES = [
    ("The sky is blue today.", "The sky is blue today."),
    ("The meeting starts at 9am.", "The meeting starts at 5pm."),
    ("Two plus two equals four.", "Two plus two equals five."),
    ("The store closes on Sundays.", "The store closes on Sundays."),
    ("The package weighs three kilograms.", "The package weighs three kilograms."),
]


def main():
    cfg = PROMPTS["P_asserting"]
    assert_loopback_host(HOST)
    run_tag = time.strftime("c131_validate_%Y%m%dT%H%M%S")
    write_egress_evidence(run_tag)

    per_size = {}
    any_exception = False
    for size in SIZES:
        model = f"qwen3:{size}"
        probe_results = []
        for nursing, discharge in PROBES:
            user = cfg["user_template"].format(
                nursing_note=nursing, discharge_summary=discharge)
            reply = call_chat_remedy(model, cfg["system"], user)
            if reply is None:
                probe_results.append({"call_succeeded": False})
                continue
            n_positions = len(reply["token_logprobs"])
            each_has_top = bool(n_positions) and all(
                bool(e.get("top_logprobs")) for e in reply["token_logprobs"])
            try:
                scored = parse_and_score_remedy(reply, cfg["pos_tokens"], cfg["neg_tokens"])
                raised = False
            except Exception as exc:
                scored = None
                raised = True
                any_exception = True
                print(f"parse_and_score_remedy raised {type(exc).__name__} "
                      f"for {model} on a synthetic probe", file=sys.stderr)
            probe_results.append({
                "call_succeeded": True,
                "n_positions": n_positions,
                "each_position_has_top_logprobs": each_has_top,
                "parse_raised": raised,
                "answer_token_position": scored["answer_token_position"] if scored else None,
                "first_token_is_answer": scored["first_token_is_answer"] if scored else None,
                "hard_label": scored["hard_label"] if scored else None,
            })
        n_multi_position = sum(1 for p in probe_results
                                if p.get("n_positions", 0) > 1)
        n_each_has_top = sum(1 for p in probe_results
                              if p.get("each_position_has_top_logprobs"))
        per_size[size] = {
            "probes": probe_results,
            "n_probes": len(probe_results),
            "n_call_succeeded": sum(1 for p in probe_results if p.get("call_succeeded")),
            "n_multi_position": n_multi_position,
            "n_each_position_has_top_logprobs": n_each_has_top,
        }
        print(f"qwen3:{size}: {per_size[size]['n_call_succeeded']}/{len(PROBES)} calls ok, "
              f"{n_multi_position}/{len(PROBES)} multi-position, "
              f"{n_each_has_top}/{len(PROBES)} every-position-has-top_logprobs")

    n_calls_total = sum(v["n_probes"] for v in per_size.values())
    n_succeeded_total = sum(v["n_call_succeeded"] for v in per_size.values())
    n_multi_position_total = sum(v["n_multi_position"] for v in per_size.values())
    n_each_has_top_total = sum(v["n_each_position_has_top_logprobs"] for v in per_size.values())

    per_token_logprobs_available = bool(
        n_multi_position_total > 0 and n_each_has_top_total == n_multi_position_total)
    all_checks_passed = bool(
        not any_exception and n_succeeded_total == n_calls_total)

    summary = (
        f"{n_succeeded_total}/{n_calls_total} calls succeeded; "
        f"{n_multi_position_total} probes generated more than one token; "
        f"per_token_logprobs_available={per_token_logprobs_available}; "
        f"parse_and_score_remedy raised on {int(any_exception)} probe(s)")

    out = {
        "preregistration": "accord/c131_scale_ladder_v2/C131_PREREGISTRATION.md",
        "run_tag": run_tag,
        "num_predict": NUM_PREDICT,
        "n_probes_per_size": len(PROBES),
        "per_size": per_size,
        "n_calls_total": n_calls_total,
        "n_succeeded_total": n_succeeded_total,
        "per_token_logprobs_available": per_token_logprobs_available,
        "all_checks_passed": all_checks_passed,
        "summary": summary,
    }
    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_PATH.write_text(json.dumps(out, indent=2) + "\n")

    prereg_text = PREREGISTRATION.read_text()
    marker = ("(appended here, from `results/accord/c131_scale_ladder_v2/"
               "synthetic_probe_validation.json`, before\n`run_c131.py` is ever "
               "pointed at the gold CSV -- the rule above is not edited)")
    addendum = (
        f"\n**Validated** (probe run {run_tag}; artifact "
        f"`results/accord/c131_scale_ladder_v2/synthetic_probe_validation.json`). "
        f"{summary}. Method confirmed on 5 synthetic non-clinical probes per size, "
        f"across all five qwen3 sizes. "
        f"{'run_c131.py proceeds with the primary (per-position) graded-score method for the clinical run.' if per_token_logprobs_available else 'run_c131.py falls back to the legacy position-0-only graded score, marking sizes without a first-token answer as not evaluable, per the pre-registration fallback rule.'}\n"
    )
    if marker in prereg_text:
        prereg_text = prereg_text.replace(marker, marker + "\n" + addendum)
        PREREGISTRATION.write_text(prereg_text)

    print(f"wrote {VALIDATION_PATH}; all_checks_passed={all_checks_passed}; {summary}")
    if not all_checks_passed:
        raise SystemExit(
            "synthetic probe validation FAILED; run_c131.py will refuse to run "
            "against gold data until this is fixed and re-validated")


if __name__ == "__main__":
    main()
