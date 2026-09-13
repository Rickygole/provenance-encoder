# C100 pre-registration: an off-the-shelf NLI baseline

Committed before the first datum. The outcome is appended below after the run; the
rule above the outcome line is never edited.

## Why

Every labeller measured so far in this thesis is either a purpose-built detector or
a general chat model prompted for a one-word label. Neither is an off-the-shelf
natural-language-inference model, the class of system the medication-contradiction
task most directly resembles on paper. This run measures one, under the same
gold-standard and decontamination discipline as everything else in this thesis.

## What runs

- Model: `FacebookAI/roberta-large-mnli`, pinned at commit SHA
  `2a8f12d27941090092df78e4ba6f0928eb5eac98`, fetched by `accord/differential/fetch_nli_model.py`
  and recorded in `results/accord/differential/nli_model_manifest.json` with a
  SHA-256 per file.
- Served offline, CPU only, float32. `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`,
  `HF_DATASETS_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY` and `DO_NOT_TRACK` are set as the
  first lines of `accord/differential/nli_baseline.py`, before any other import; the
  model loads with `local_files_only=True` from the local directory only; a socket
  guard is installed after the model loads and before the gold CSV opens; the whole
  run executes under `sandbox-exec -p '(version 1)(allow default)(deny network*)'`.
- Score per pair: mean contradiction probability over both directions (nursing
  sentence as premise / discharge sentence as hypothesis, and the reverse), softmax
  over the model's three logits. The contradiction index is read from
  `model.config.id2label` and never hard-coded.
- No truncation: every encoding is built with `truncation=False`, and every encoding
  is asserted to fit 512 tokens before it reaches the model.
- Pairs: all 299 gold-labelled pairs from `pharmacist_labeled_clean.csv`.

## Decision rule

**Primary, G_253 (the 253 non-ambiguous pairs):** AUROC on `contra_prob_mean` against
`label_clean`, and a one-sided Mann-Whitney p. The threshold is read from
`mde_recomputed.gold_ambiguous_excluded.minimum_detectable_auc` (0.7355), never
retyped.

- **Outcome A, discriminates at a resolvable size:** AUROC >= 0.7355 AND one-sided
  Mann-Whitney p < 0.05. Reported as: an off-the-shelf NLI model separates the
  confirmed cases at a size this evaluation can resolve, at a size no trained system
  measured in this thesis does.
- **Outcome B, not resolved:** anything else. Reported as: the off-the-shelf NLI
  baseline does not reach the effect size this evaluation can resolve; nothing is
  concluded about a smaller effect.
- **Outcome C, invalid:** any encoding over 512 tokens, any NaN score, n != 253 (for
  the primary basis) or n != 299 (for the secondary basis), the `id2label`
  assertion fails to find a contradiction class, a SHA-256 mismatch against the
  fetch manifest, the network-egress sandbox guard trips, or the detector's
  recomputed AUROC fails to match the registered
  `signal.contra_prob_vs_pharm_label__ambiguous_excluded_n253.auroc` /
  `__ambiguous_as_negative_n299.auroc` to four decimal places.

**Secondary, G_299 (ambiguous pairs counted as negative), reported regardless of the
primary outcome:**

- AUROC, with a stratified percentile bootstrap CI (B=10000, seed 42).
- Average precision.
- True positives within the top 15 pairs by score.
- A paired DeLong test against the detector's per-pair score
  (`~/mimic_local_only/derived/gold_standard_accord_results.csv`, joined on
  `pair_id`), on both G_253 and G_299, via `accord/control/delong.py`, run only after
  the detector's own AUROC is recomputed from that file and matches the two
  registered values above to four decimal places (the script aborts otherwise rather
  than testing against a stale score).
- Each direction's own AUROC (nursing-premise, discharge-premise), reported as
  descriptive only; neither is a pre-registered test.

## Kill condition

If interrupted, restarted from the first pair, never spliced.

## Outcome

Recorded after the run; the rule above is unchanged.

PLACEHOLDER_OUTCOME_FILLED_AFTER_RUN
