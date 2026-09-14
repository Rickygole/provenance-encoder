> **Superseded.** This README describes an earlier conference supplement and is kept for the record only.
> Several files it lists, including results from hosted model services, are withdrawn and are not part of
> the thesis; see `results/accord/withdrawn_hosted/README.md` and `accord/CORRECTIONS.md`.

# Supplemental Material

# Rate Calibration Is Not Safety: The Necessary Conditions
# for Clinical LLM Supervision under Extreme Prevalence Imbalance
# NeurIPS 2026 Submission (Anonymous)

This folder contains everything a reviewer needs to verify the numbers reported in the paper. The folder is organized into a flat top level for the core artifacts that reviewers will look at first, and a results subfolder for the raw per-configuration JSON outputs.

## What is, and is not, in this package

This package distributes only pair-level predictions, prompts, parsing code, and aggregate
result files. It does not, and will not, include row-level files that join a MIMIC-III
identifier (such as `HADM_ID`) to verbatim note text. Two such files existed during earlier
preparation of this package (a weak-supervision label pool and a decontamination audit trail,
both keyed on `HADM_ID` with verbatim nursing-note and discharge-summary sentences attached);
they have been removed from distribution and are not present anywhere in this repository. They
remain, credentialed-access-only, outside version control, alongside the MIMIC-III source data
they were derived from. See "Data Access" below for what that means for reproducing the
decontaminated retrain from scratch.

## Repository Structure

supplemental/

  README.md                                       this file
  predictions_253_core.csv                        pair-level predictions for the 12 positive pairs (corrected v2)
  prompts_existing.csv                            verbatim prompts and model version strings
  parser.py                                       exact parser used to convert raw model outputs into binary labels
  error_inventory_12.csv                          all 12 true contradictions with caught/missed status (corrected v2)

  results/

    prompt_intervention_results.json              Llama 3 8B results across all 5 prompt strategies
    mistral_prompt_intervention_results.json      Mistral 7B v0.3 results across all 5 prompt strategies
    gpt4o_clinical_miscalibration.json            GPT-4o zero-shot raw output (original N=299 run)
    gpt4o_fewshot_pairlevel.json                  GPT-4o 6-shot raw output (N=253 run)
    full_metric_suite_results.json                ACCORD on the gold standard (full metric suite)
    ablation_v2_results.json                      ACCORD v2 ablation results (VA, VB, VC, VD)
    supervised_baseline_results.json              PubMedBERT supervised baseline (5-fold CV, 0/12 contradictions caught)
    kappa_bootstrap_cis.json                      stratified bootstrap kappa CIs for archived configurations
    stratified_bootstrap_results.json             full stratified bootstrap output (B=10,000)
    accord_v2_decontam_gold_aggregate.json        per-seed and aggregate metrics for the decontaminated ACCORD retrain
    accord_v2_decontam_results.json               raw per-seed outputs for the decontaminated ACCORD retrain
    accord_decontamination_paper_table.csv        contaminated vs decontaminated comparison table with deltas

## Decontaminated ACCORD: how to verify the main paper numbers

The primary ACCORD result in Table 1 of the paper is the decontaminated retrain. Three random seeds were trained from scratch on a weak-supervision pool with all 299 pharmacist-annotated pairs removed. The training pool itself and the audit trail of removed pairs are row-level, `HADM_ID`-joined artifacts and are not distributed in this package (see "What is, and is not, in this package" above). The relevant distributable artifacts are the aggregate outputs built from that training run:

  results/accord_v2_decontam_gold_aggregate.json: per-seed confusion matrices, recall, MCC, kappa, F-beta, PR-AUC, ECE, and authority scales for all three decontaminated seeds, plus aggregate means and bootstrap CIs
  results/accord_v2_decontam_results.json: the raw per-seed result records that the aggregate file is built from
  results/accord_decontamination_paper_table.csv: the side-by-side comparison of the contaminated single-run and the decontaminated three-seed mean

Per-seed confusion matrices in the aggregate JSON:

  Seed 0: TP=9, FP=216, FN=3, TN=25 (sum 253)
  Seed 1: TP=10, FP=210, FN=2, TN=31 (sum 253)
  Seed 2: TP=9, FP=207, FN=3, TN=34 (sum 253)
  Mean:   TP=9.33, FP=211.00, FN=2.67, TN=30.00

These matrices recover the aggregate numbers reported in the main text and in Appendix B Table 7: recall = 77.8%, MCC = -0.063, PABAK = -0.689, balanced accuracy = 0.451, αN = 0.508, αD = 1.004.

## Correction log for the corrected v2 prediction files

The gpt4o_zeroshot column in predictions_253_core.csv (for the 12 positive pairs) was corrected against gpt4o_clinical_miscalibration.json (original N=299 run). Six values were wrong in the prior version of this CSV:

  PAIR_0067: 1 to 0 (original returned not_contradiction)
  PAIR_0218: 1 to 0 (original returned a failed parse, treated as missed)
  PAIR_0094: 0 to 1 (original returned contradiction, caught)
  PAIR_0131: 0 to 1 (original returned contradiction, caught)
  PAIR_0190: 0 to 1 (original returned contradiction, caught)
  PAIR_0065: 0 to 1 (original returned contradiction, caught)

The corrected GPT-4o zero-shot caught set is PAIR_0010, PAIR_0065, PAIR_0094, PAIR_0131, PAIR_0190, PAIR_0268 = 6 pairs = 50.0% recall. This matches the paper's reported TP=6, FN=6, MCC=+0.109.

The gpt4o_fewshot column was corrected against gpt4o_fewshot_pairlevel.json (N=253 run). Eight values were wrong in the prior version:

  PAIR_0067: 1 to 0 (missed)
  PAIR_0131: 0 to 1 (caught)
  PAIR_0010: 1 to 0 (missed)
  PAIR_0134: 1 to 0 (missed)
  PAIR_0190: 0 to 1 (caught)
  PAIR_0247: 0 to 1 (caught)
  PAIR_0003: 0 to 1 (caught)
  PAIR_0065: 0 to 1 (caught)

The corrected GPT-4o few-shot caught set is PAIR_0003, PAIR_0065, PAIR_0094, PAIR_0131, PAIR_0190, PAIR_0218, PAIR_0247 = 7 pairs = 58.3% recall. This matches the paper's reported 58.3% recall.

The gpt4o_zeroshot_result column in error_inventory_12.csv was corrected to match the same source of truth:

  PAIR_0067: caught to missed
  PAIR_0218: caught to missed
  PAIR_0094: missed to caught
  PAIR_0131: missed to caught
  PAIR_0190: missed to caught
  PAIR_0065: missed to caught

All Llama and ACCORD values in both files were correct in the original version and are unchanged.

## Data Access

The MIMIC-III dataset requires credentialed access via PhysioNet. Registration is at https://physionet.org/content/mimiciii/1.4/

The annotation file (annotation_pharmacist_complete.xlsx) contains verbatim de-identified MIMIC-III note sentences, not derived features, and is therefore restricted data under the data use agreement rather than a derived artifact. It is not redistributed here in any form. The pair IDs (PAIR_0001 through PAIR_0299) correspond to the annotated subset described in Section 3 of the paper, but they are an internal identifier local to this project's own pipeline, not a MIMIC-III admission or subject identifier, and no crosswalk from a pair ID back to the source note text or to `HADM_ID`/`SUBJECT_ID` is distributed or retained outside the credentialed environment. A researcher with PhysioNet access cannot use the pair IDs alone to reconstruct the note pair text; doing so would require re-running this project's own note-pairing pipeline against the credentialed MIMIC-III release.

Files containing verbatim note text or MIMIC-III admission/subject identifiers are not redistributable under the PhysioNet data use agreement and are not included in this package, in any form, at any level.

## predictions_253_core.csv

253 rows expected. One row per binary pair in the evaluation set, but only the 12 positive pairs are released at the individual row level. Aggregate confusion matrix entries for the 241 negative pairs are provided as comments at the bottom of the file. The 241 negative pairs are not redistributable under the MIMIC-III data use agreement.

Columns:

  pair_id           internal identifier (PAIR_XXXX)
  gold_label        1 = contradiction, 0 = not_contradiction
  danger_rating     pharmacist danger rating 1 to 5 (positives only, NA for negatives)
  ambiguous_flag    1 = excluded from primary analysis, 0 = included
  llama_baseline    Llama 3 8B baseline prediction (0 or 1)
  gpt4o_zeroshot    GPT-4o zero-shot prediction (0 or 1)
  gpt4o_fewshot     GPT-4o 6-shot prediction (0 or 1)
  accord_score      ACCORD v2 contradiction probability (float)
  accord_pred       ACCORD v2 binary prediction at the 0.5 decision threshold (0 or 1)

Configurations marked NA in the paper (predictions not archived): llama_cot, llama_baserate, llama_expert, llama_calibration, mistral_baseline, mistral_cot, mistral_baserate, mistral_expert, mistral_calibration.

## prompts_existing.csv

One row per archived configuration.

Columns:

  config_id       configuration identifier
  model_id        exact model version string
  temperature     decoding temperature (0 for all)
  max_tokens      maximum output tokens
  system_prompt   verbatim system prompt
  user_prompt     verbatim user prompt template ({nursing_note} and {discharge_summary} are placeholders)
  n_shots         number of demonstrations (0 for zero-shot)
  demo_source     source of demonstrations (synthetic or NA)

## parser.py

The exact function used to convert raw model output strings into binary contradiction labels. Case-insensitive string match on the first word. A re-run of chain-of-thought with verbatim prompts produced a 20% parse failure rate, which is documented in Section 9 of the paper.

## results/accord_v2_decontam_gold_aggregate.json

The primary verification artifact for the decontaminated ACCORD result reported in Table 1, Figure 1, Section 5.3, and Appendix B Table 7 of the paper. Contains:

  configuration: dataset dimensions, seed list, decision threshold, bootstrap B
  aggregate: mean, std, min, max across seeds for recall, precision, F1, F2, F4, PR-AUC, kappa, ECE, αN, αD
  per_seed: for each seed (0, 1, 2): the confusion matrix, all metrics, Clopper-Pearson and stratified-bootstrap CIs (B=10,000), the per-seed authority weights αN and αD, and recall-at-budget at every 5% review-budget level from 5% to 100%

## results/accord_v2_decontam_results.json

The raw per-seed result records that the aggregate file is built from. Sanity-check artifact: a reviewer can recompute the aggregate file from this one.

## results/accord_decontamination_paper_table.csv

The side-by-side comparison of the contaminated single-run ACCORD and the decontaminated three-seed mean. One row per metric (recall, precision, F1, F2, F4, PR-AUC, kappa, ECE, αN, αD) with the contaminated value, the decontaminated mean and std, the absolute delta, and a one-line interpretation. This is the table that drove the v6 paper's decontamination integration.
