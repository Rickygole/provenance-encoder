"""Re-run the frontier-model comparison through a compliant channel.

WHY THIS SCRIPT EXISTS
----------------------
The original comparison was run against OpenAI's public API. PhysioNet's
guidance of 18 April 2023 states that the Credentialed Data Use Agreement
"explicitly prohibits sharing access to the data with third parties, including
sending it through APIs provided by companies like OpenAI, or using it in
online platforms like ChatGPT". The original run therefore breached Clause 3
of the agreement signed on 25 February 2026.

That guidance also names the routes that ARE acceptable. This script uses two
of them:

  anthropic  (default)  Named as not training on prompts and not performing
                        routine human review by default. Needs no opt-out
                        form, no cloud account, no approval queue.
  azure                 Same GPT-4o family, but ONLY after Microsoft approves
                        Modified Abuse Monitoring for the resource. Without
                        that, prompts may be retained and human-reviewed,
                        which is the disclosure Clause 3 forbids.

It also closes a Clause 9 gap: the original run's code was never in this
repository, which made those two tables the only results in the thesis with no
committed code path.

WHAT LEAVES THIS MACHINE
------------------------
Exactly what the original run sent, bound from the prompts recorded verbatim
in prompts_existing.csv: the nursing and discharge sentences from the
annotation sheet: one mining-extracted sentence per note side, not
whole notes. No admission identifier, no
subject identifier, no free note body. The few-shot exemplars are synthetic.

Note the template's placeholders are named nursing_note and discharge_summary
while the content is a single sentence. That mismatch caused two separate
reviews of this project to conclude whole notes had been transmitted. The
binding is reproduced exactly as the original had it, and recorded here so it
is not misread a third time.

WHAT IS WRITTEN BACK
--------------------
Aggregates and per-pair labels only, keyed by the project-internal pair id.
No note text is ever written into this repository or printed to the console,
including on error, and no header carrying a key is ever logged.

CONFIGURATION, via environment. This script never reads a key from a file and
never logs one:
    ANTHROPIC_API_KEY         for --provider anthropic
    ANTHROPIC_MODEL           optional, defaults below
    AZURE_OPENAI_ENDPOINT     for --provider azure
    AZURE_OPENAI_KEY
    AZURE_OPENAI_DEPLOYMENT
    AZURE_OPENAI_API_VERSION  optional, defaults below
    ACCORD_SHEET              path to annotation_sheet_for_pharmacist_clean.csv
    ACCORD_PROMPTS            path to prompts_existing.csv

Run:
    python3 accord/differential/run_frontier_compliant.py --strategy zero_shot
    python3 accord/differential/run_frontier_compliant.py --strategy few_shot
"""

import argparse
import json
import os
import pathlib
import sys
import time

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "accord" / "differential"
DEFAULT_API_VERSION = "2024-02-01"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

# The parser from the original run: case-insensitive match on the first word.
# Anything else is a parse failure and is counted, never defaulted to a class.
COMPLIANCE_NOTE = {
    "anthropic": ("Run through the Anthropic API, which PhysioNet's guidance "
                  "of 18 April 2023 names as a compliant route on the grounds "
                  "that prompts are not used for training and routine human "
                  "review is not performed. Supersedes the original run "
                  "against OpenAI's public API, which breached Clause 3."),
    "azure": ("Run through Azure OpenAI with human review opted out, the route "
              "named in PhysioNet's guidance of 18 April 2023. Supersedes the "
              "original public-API run, which breached Clause 3."),
}

POSITIVE = "contradiction"
NEGATIVE = "not_contradiction"


def need(var):
    v = os.environ.get(var)
    if not v:
        sys.exit(f"{var} is not set. See this file's docstring; the key is read "
                 f"from the environment and must never be committed.")
    return v


def load_prompts(path, config_id):
    df = pd.read_csv(path)
    row = df.loc[df["config_id"] == config_id]
    if row.empty:
        sys.exit(f"config_id {config_id!r} not found in {path}")
    r = row.iloc[0]
    return str(r["system_prompt"]), str(r["user_prompt"]), r


def parse(reply):
    """First word, case-insensitive. Returns 1, 0, or None for unparseable."""
    if not reply:
        return None
    first = reply.strip().split()[0].strip(".,:;").lower() if reply.strip() else ""
    if first.startswith(NEGATIVE[:7]):        # not_con...
        return 0
    if first.startswith(POSITIVE[:7]):        # contrad...
        return 1
    return None


def call(cfg, system, user, temperature, max_tokens, retries=4):
    """One request, to whichever compliant provider is configured.

    Both providers named here appear in PhysioNet's guidance of 18 April 2023
    as acceptable routes for credentialed data. Azure requires human review to
    be opted out first; Anthropic is listed as not training on prompts and not
    performing routine human review by default, so it needs no such form.
    """
    if cfg["provider"] == "azure":
        url = (f"{cfg['endpoint'].rstrip('/')}/openai/deployments/"
               f"{cfg['deployment']}/chat/completions"
               f"?api-version={cfg['api_version']}")
        headers = {"api-key": cfg["key"], "Content-Type": "application/json"}
        # temperature_supported is an OUTPUT field and was pasted into this
        # request body by mistake. Azure rejects unrecognised properties, and
        # the 400 fallback below is anthropic-only, so leaving it here would
        # have nulled out every pair on --provider azure.
        payload = {"messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "temperature": (None if cfg.get("no_temperature") else temperature),
                   "max_tokens": max_tokens}
        pick = lambda j: j["choices"][0]["message"]["content"]
    else:                                              # anthropic
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": cfg["key"],
                   "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        payload = {"model": cfg["model"], "system": system,
                   "messages": [{"role": "user", "content": user}],
                   "max_tokens": max_tokens}
        # Newer Anthropic models reject `temperature` outright. The original
        # run used temperature 0 for determinism, so it is sent when the model
        # accepts it and dropped when it does not. Which happened is recorded
        # in the output, because it is a real difference from the original run.
        if not cfg.get("no_temperature"):
            payload["temperature"] = temperature
        # Do NOT assume content[0] is the answer. A response may lead with a
        # thinking block, which carries no "text" key at all. Scan for the
        # first text block; if there is none, return None so the caller counts
        # it as unparseable rather than crashing or inventing a label.
        def pick(j):
            if os.environ.get("ACCORD_DIAGNOSE"):
                print(f"      stop_reason={j.get('stop_reason')!r} "
                      f"blocks={[b.get('type') for b in j.get('content', [])]} "
                      f"usage={j.get('usage', {}).get('output_tokens')}",
                      file=sys.stderr)
            for block in j.get("content", []):
                if block.get("type") == "text" and "text" in block:
                    return block["text"]
            return None

    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            if r.status_code in (429, 529):
                print(f"    rate limited ({r.status_code}), retry "
                      f"{attempt + 1}/{retries}", file=sys.stderr)
                time.sleep(2 ** attempt)
                continue
            if (r.status_code == 400 and cfg["provider"] == "anthropic"
                    and not cfg.get("no_temperature")):
                try:
                    msg = r.json().get("error", {}).get("message", "")
                except Exception:
                    msg = ""
                if "temperature" in msg:
                    cfg["no_temperature"] = True
                    print("    model rejects temperature; resending without "
                          "it, and recording that in the output",
                          file=sys.stderr)
                    payload.pop("temperature", None)
                    continue
            if r.status_code >= 400:
                # Status and the provider's error TYPE only. The body is not
                # echoed: a 400 can quote the offending request back at you,
                # and that request holds note text.
                try:
                    etype = r.json().get("error", {}).get("type", "unknown")
                except Exception:
                    etype = "unparseable"
                print(f"    HTTP {r.status_code} ({etype})", file=sys.stderr)
            r.raise_for_status()
            return pick(r.json())
        except requests.RequestException as exc:
            # Deliberately does not echo the request body: it holds note text.
            # Nor the headers: they hold the key.
            print(f"    attempt {attempt + 1}/{retries} failed: "
                  f"{type(exc).__name__}", file=sys.stderr)
            if attempt == retries - 1:
                print(f"    request failed after {retries} attempts: "
                      f"{type(exc).__name__}", file=sys.stderr)
                return None
            time.sleep(2 ** attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["zero_shot", "few_shot"],
                    required=True)
    ap.add_argument("--provider", choices=["anthropic", "azure"],
                    default="anthropic",
                    help="which compliant route to use. anthropic needs no "
                         "opt-out form; azure requires Modified Abuse "
                         "Monitoring to be approved first.")
    ap.add_argument("--replicates", type=int, default=1,
                    help="calls per pair. The default of 1 reproduces the "
                         "single-draw run. Above 1 the per-pair label becomes "
                         "the majority over replicates and the disagreement "
                         "rate is reported, which is the only honest way to "
                         "quote a figure from a model whose decoding cannot be "
                         "pinned to a temperature.")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of pairs, for a dry run")
    ap.add_argument("--diagnose", metavar="PAIR_IDS",
                    help="comma-separated pair ids to re-run, printing the raw "
                         "reply for each so unparseable answers can be seen. "
                         "Prints to the console only and writes nothing.")
    args = ap.parse_args()
    if args.replicates < 1:
        # Zero or negative still takes one draw per pair, but labels the output
        # "majority of 0 draws" and, having no _rep suffix, writes it over the
        # committed single-draw artifact that VERIFIED_FACTS.json cites.
        ap.error("--replicates must be at least 1")

    config_id = "gpt4o_zeroshot" if args.strategy == "zero_shot" else "gpt4o_fewshot"

    if args.provider == "azure":
        cfg = {"provider": "azure",
               "endpoint": need("AZURE_OPENAI_ENDPOINT"),
               "key": need("AZURE_OPENAI_KEY"),
               "deployment": need("AZURE_OPENAI_DEPLOYMENT"),
               "api_version": os.environ.get("AZURE_OPENAI_API_VERSION",
                                             DEFAULT_API_VERSION)}
        model_label = cfg["deployment"]
    else:
        cfg = {"provider": "anthropic",
               "key": need("ANTHROPIC_API_KEY"),
               "model": os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)}
        model_label = cfg["model"]
    sheet = need("ACCORD_SHEET")
    prompts = need("ACCORD_PROMPTS")

    # prow is the recorded prompt row. It must NOT be called cfg:
    # that name already holds the provider config built above, and
    # shadowing it made call() receive a pandas Series.
    system, user_tmpl, prow = load_prompts(prompts, config_id)
    temperature = float(prow["temperature"])
    max_tokens = int(prow["max_tokens"])
    # The recorded config used 50, which was ample for GPT-4o's one-word reply.
    # Reasoning models spend that budget on a thinking block and return no text
    # at all: stop_reason='max_tokens', blocks=['thinking'], usage=50. Six of
    # 300 few-shot pairs failed exactly that way. Raising the ceiling is a
    # deviation from the original run and is recorded in the output as one.
    max_tokens_override = os.environ.get("ACCORD_MAX_TOKENS")
    if max_tokens_override:
        max_tokens = int(max_tokens_override)

    df = pd.read_csv(sheet)
    if args.diagnose:
        want = [s.strip() for s in args.diagnose.split(",")]
        df = df[df["pair_id"].isin(want)]
        os.environ["ACCORD_DIAGNOSE"] = "1"
        print(f"diagnosing {len(df)} pair(s); replies print here and are NOT "
              f"written to the repository\n")
        for _, row in df.iterrows():
            prompt = user_tmpl.format(
                nursing_note=row.get("nursing_sentence", ""),
                discharge_summary=row.get("discharge_sentence", ""))
            reply = call(cfg, system, prompt, temperature, max_tokens)
            print(f"  {row['pair_id']}: parsed={parse(reply)!r}  "
                  f"reply={reply!r}")
        return
    if args.limit:
        df = df.head(args.limit)

    print(f"strategy={args.strategy}  provider={args.provider}  "
          f"model={model_label}  pairs={len(df)}  temperature={temperature}")

    preds, failed, failed_pairs = [], 0, 0
    for i, row in df.iterrows():
        # The template's placeholders are named nursing_note and
        # discharge_summary, but the original run filled them from the
        # annotation sheet's nursing_sentence and discharge_sentence columns.
        # Those hold one sentence per side, segmented in phase2_medication.py
        # and selected in phase3_mining.py (lines 211 to 212). The names say
        # "note"; the content is a sentence. That mismatch is what made two
        # separate reviews of this project conclude whole notes were sent.
        # There is no character cap anywhere in the pipeline; an earlier
        # version of this comment asserted a 200-character one and was wrong. The
        # binding is reproduced here exactly as the original run had it, so
        # this re-run is comparable, and recorded so nobody misreads it again.
        prompt = user_tmpl.format(
            nursing_note=row.get("nursing_sentence", ""),
            discharge_summary=row.get("discharge_sentence", ""))
        draws = []
        for _ in range(max(1, args.replicates)):
            draws.append(parse(call(cfg, system, prompt, temperature,
                                    max_tokens)))
        usable = [d for d in draws if d is not None]
        failed += len(draws) - len(usable)

        if not usable:
            p, resolved = None, False
            failed_pairs += 1
        elif usable.count(1) * 2 == len(usable):
            # An even split. Breaking it arbitrarily would invent a label and
            # conceal exactly the instability this mode exists to measure, so
            # the pair is recorded unresolved and counted as such.
            p, resolved = None, False
        else:
            p = 1 if usable.count(1) * 2 > len(usable) else 0
            resolved = True

        rec = {"pair_id": row["pair_id"], "pred": p}
        if args.replicates > 1:
            rec["n_draws"] = len(draws)
            rec["n_usable"] = len(usable)
            rec["n_positive"] = usable.count(1)
            rec["unanimous"] = bool(usable) and len(set(usable)) == 1
            rec["resolved"] = resolved
        preds.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(df)}")

    flagged = sum(1 for p in preds if p["pred"] == 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # A replicated run must never overwrite the single-draw artifact. Both are
    # reported: the single draw is what the first pass measured, the replicated
    # run is what the thesis should quote, and the difference between them is
    # itself the finding.
    suffix = f"_rep{args.replicates}" if args.replicates > 1 else ""
    # Tag the filename with the model whenever it is not the default. Without
    # this, running a second model writes over the first: a Haiku run would
    # silently destroy the committed Sonnet artifacts that Chapter 4 depends
    # on, and the loss would not be visible until the metrics were recomputed.
    model_tag = ""
    if args.provider == "anthropic" and model_label != DEFAULT_ANTHROPIC_MODEL:
        model_tag = "_" + model_label.replace("-", "").replace(".", "")
    out = OUT_DIR / f"frontier_{args.provider}{model_tag}_{args.strategy}{suffix}.json"

    stability = None
    if args.replicates > 1:
        n = len(preds)
        unstable = [p for p in preds if p["n_usable"] > 1
                    and not p["unanimous"]]
        unresolved = [p for p in preds if not p["resolved"]]
        # How far would the reported numbers have moved? Compare the majority
        # label against the single draw already committed, pair by pair.
        # Must carry the same model tag as `out`. Without it this compared a
        # replicated Haiku run against the SONNET single-draw artifact and
        # reported six label changes that were a difference between two models,
        # not between two runs of one. The same tag-position mistake as in
        # frontier_metrics.py, made twice.
        prior = OUT_DIR / f"frontier_{args.provider}{model_tag}_{args.strategy}.json"
        changed = None
        if prior.exists():
            was = {r["pair_id"]: r["pred"]
                   for r in json.load(open(prior))["predictions"]}
            # Resolved pairs only. A pair that lost its label to an even split
            # has not flipped, and counting it here would report a label change
            # that never happened; it is already in n_pairs_unresolved.
            changed = sum(1 for p in preds
                          if p["resolved"] and p["pair_id"] in was
                          and was[p["pair_id"]] != p["pred"])
        stability = {
            "replicates_per_pair": args.replicates,
            "n_pairs": n,
            "n_pairs_not_unanimous": len(unstable),
            "pair_disagreement_rate": len(unstable) / n if n else None,
            "n_pairs_unresolved": len(unresolved),
            "n_labels_changed_vs_single_draw": changed,
            "tie_rule": ("an even split among usable draws is recorded as "
                         "unresolved rather than broken toward either class"),
        }

    json.dump({
        "strategy": args.strategy,
        "channel": args.provider,
        "model": model_label,
        "compliance_note": COMPLIANCE_NOTE[args.provider],

        "temperature": (None if cfg.get("no_temperature") else temperature),
        "temperature_supported": not cfg.get("no_temperature"),
        "max_tokens": max_tokens,
        "max_tokens_original_config": int(prow["max_tokens"]),
        "max_tokens_overridden": bool(max_tokens_override),
        "n_shots": int(prow["n_shots"]) if pd.notna(prow["n_shots"]) else 0,
        "demo_source": (None if pd.isna(prow["demo_source"])
                        else str(prow["demo_source"])),
        "n_pairs": len(df),
        "n_flagged": flagged,
        "flag_rate": flagged / len(df) if len(df) else None,
        # failed_parses stays a PAIR count. frontier_metrics.py copies it
        # straight into its own output and VERIFIED_FACTS.json quotes it beside
        # n_pairs_scored, so 7 bad draws out of 1500 would be read as 7 pairs
        # out of 300. The draw total is a separate field, not overloaded here.
        "failed_parses": failed_pairs,
        "failed_parse_draws": failed,
        "replicates_per_pair": args.replicates,
        "label_rule": ("single draw" if args.replicates == 1
                       else f"majority of {args.replicates} draws"),
        "stability": stability,
        "predictions": preds,
    }, open(out, "w"), indent=2)
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"  flagged {flagged}/{len(df)}, {failed} unparseable draw(s), "
          f"{failed_pairs} pair(s) with none usable")
    if stability:
        s = stability
        print(f"  {s['n_pairs_not_unanimous']}/{s['n_pairs']} pairs not "
              f"unanimous across {args.replicates} draws "
              f"(disagreement rate {s['pair_disagreement_rate']:.4f})")
        print(f"  {s['n_pairs_unresolved']} unresolved (even split or no "
              f"usable draw)")
        if s["n_labels_changed_vs_single_draw"] is not None:
            print(f"  {s['n_labels_changed_vs_single_draw']} label(s) differ "
                  f"from the committed single-draw run")
    print("  no note text was written to this repository or to the console.")


if __name__ == "__main__":
    main()
