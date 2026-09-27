# Must-not-publish audit of the reproduction closure

Scope: the 168 files in `manifest_include.txt`. Every text file was scanned for
clinical note text, `HADM_ID` / `SUBJECT_ID` / `ICUSTAY_ID` values, hosted-vendor and
product names, API keys, AWS account numbers, absolute machine paths, institutional
email addresses, and reads from a cloud-synced directory. The 25 figure PDFs and
`main_msu.pdf` were scanned as rendered text with `pdftotext`.

Ranked by severity. Severity is about what the file would disclose if published,
not about how hard it is to fix.

---

## HIGH

### H1. `results/accord/VERIFIED_FACTS.json` carries the whole withdrawn hosted record, including the vendor and product inventory

Required (it is the central register; three of the six commands read it). It contains
**1,089 `withdrawn.*` keys, 823 of them `withdrawn.hosted.*`**, with their numeric
values intact. 567 register key names and 696 register artifact strings name a hosted
vendor or product. Examples of the key names, verbatim:

    withdrawn.hosted.capability.ladder.claude_haiku_45.ratio
    withdrawn.hosted.capability.ladder.claude_sonnet_5.ratio
    withdrawn.hosted.capability.ladder.metallama370binstructv10.ratio
    withdrawn.hosted.ceiling.haiku_fs.ceiling_mcc
    withdrawn.hosted.capability.llama70b_few_shot_ratio

The register also still cites, by path, the six hosted artifacts that were removed
from disk, one of which is `results/accord/differential/transmission_ledger.json`.
So publishing the register publishes the fact that a transmission ledger existed,
the filenames of the withdrawn hosted metric files, the vendor and product list, and
the hosted numbers themselves.

This is the same class of content as `results/accord/withdrawn_hosted/**`, which the
instruction says must never reach a public repository. Excluding that directory while
publishing the register does not achieve the stated goal.

**Can it be published after a change: yes.** Stripping every `withdrawn.*` key was
tested. With 1,089 keys removed (6,524 down to 5,435), `verify_thesis.py` still
reports all gating passes clear, `trace_claims.py` still resolves the same 1,337
occurrences to the same 660 register entries, and no live prose number becomes
untraceable. The only consequences are that Appendix B goes from 82 groups / 6,524
values to 81 / 5,435, and that the counts the thesis prints for the register change.
Those counts are already stale (see M3), so this is one regeneration, not two.
**Decision needed from the author**, because the withdrawal record is also part of
the compliance story and deleting it from the published register is a choice about
what the public artifact says happened.

### H2. `accord/hosted_withdrawal.py` is a vendor-name inventory by construction, and it is load-bearing

Required: `verify_thesis.py` imports it, so it is in the closure with no way around
it. Its `HOSTED_ARTIFACT_TOKENS` tuple names `frontier_anthropic` ("Claude Sonnet 5
and Claude Haiku 4.5 metrics" in the adjacent comment), `frontier_bedrock` ("Llama 3
70B and Llama 3 8B metrics"), `frontier_label_noise_claudehaiku`,
`transmission_ledger`, `::anthropic`, `::bedrock`, `gpt4o`, `sonnet`, `haiku`,
`claude`, `llama70b`, `metallama`. The module docstring names PhysioNet's guidance
and states that "the GPT-4o rows were described in the results chapter as withdrawn
while all 25 of their keys sat live in the register".

**Plainly: publishing the reproduction set does publish that vendor list, and there
is no version of the closure that avoids it while keeping the gate honest.** The
classifier cannot enforce a withdrawal against vendors it is not allowed to name.

**But the premise the question rests on is already false.** The thesis itself names
them, in `accord/thesis/msu_ch4_results.tex`, which is bound into `main_msu.pdf`:

    line 538-539: "to hosted API services, Claude Sonnet 5 and Claude Haiku 4.5
    through the Anthropic API and Llama 3 70B and Llama 3 8B through AWS Bedrock;
    an earlier baseline sent pairs to GPT-4o through OpenAI's API."
    line 649-650: "every one was run through the Anthropic API or AWS Bedrock"

`main_msu.tex` even carries "Anthropic" in its `\hyphenation` list. So the code does
not contradict the thesis. If the intention is to name no vendor, **the thesis text
is what has to change first**, and that is a disclosure decision, not a packaging
one: the PhysioNet withdrawal passage is arguably not honest without naming which
services received the data. If the thesis keeps the passage, `hosted_withdrawal.py`
discloses nothing new and can be published as is.

---

## MEDIUM

### M1. `results/accord/c135_dissociation_cost/c135_bias_curve.json` leaks an absolute path with the account name

Required (`verify_thesis.py` reads it). It contains:

    "parquet": "/Users/rickygole/.cache/huggingface/hub/datasets--stanfordnlp--snli/
     snapshots/cdb5c3d5eed6ead6e5a341c8e56e669bb666725b/plain_text/train-00000-of-00001.parquet"

This is the only absolute machine path anywhere in the closure. The data is public
SNLI, so nothing restricted is disclosed; what leaks is the macOS account name and
the local cache layout. **Publishable after a one-field change.** Note that
`accord/stats/make_appendices.py` already implements exactly this redaction
(`normalize_artifact_path`) for register artifact strings, but it is applied to the
appendix, not to the artifact file itself, so this value would reach a reader who
opens the JSON.

**RESOLVED.** The field now reads
`[CACHE]/hub/datasets--stanfordnlp--snli/snapshots/cdb5c3d5eed6ead6e5a341c8e56e669bb666725b/plain_text/train-00000-of-00001.parquet`,
prefix rewritten and the snapshot digest kept. `accord/c135_dissociation_cost/c135_run.py`
records that symbolic form directly, so a rerun cannot reintroduce the expansion; the
kill condition that tests whether the parquet really sits under the local cache still
tests the real path. The claim above that this was the only absolute machine path in the
closure was true of the closure and not of the tree:
`results/accord/c134_reanalysis/c134e_ambiguous46_score_audit.json` held 46 more, and
`accord/audit_c137/*.py` hard-coded the repository root in ten files. Both are fixed on
the same rule (see finding M1a).

### M1a. Absolute paths outside the closure: the ambiguous-46 search record and the C137 audit scripts

Found while closing M1, by sweeping `results/accord/` and `accord/` rather than the
closure alone.

`results/accord/c134_reanalysis/c134e_ambiguous46_score_audit.json` held 46 absolute
paths. That file is the record of an exhaustive search for a per-pair score file that
does not exist, so the paths are its evidence and none was dropped: only the two
prefixes were rewritten, the credentialed directory to `[LOCAL]` and this repository to
a repo-relative path, which is the convention `make_appendices.py` already uses. Every
directory name and filename below the prefix survives, so the record still names exactly
where it looked; what stops being published is the account name, which was never part of
the evidence. Keeping one file exempt from the rule because it holds many paths is the
"gate that cannot fail" shape in another form, so it was not made exempt. The generator
applies the same rewrite once over its assembled output, immediately before writing, so a
rerun reproduces the committed form byte for byte; it is not applied to the dicts the
search carries internally, because one candidate's path is read back and opened.

`accord/audit_c137/*.py`, ten tracked files, opened with
`ROOT = pathlib.Path("/Users/rickygole/provenance-encoder")`. That is a reproducibility
defect as well as a name leak: none of the ten runs in a clone. All ten now derive the
root as `pathlib.Path(__file__).resolve().parents[2]`, the form every other script in
the tree uses, and they run from any working directory. Separately, `a10_coverage.py`
still raises `ValueError: cannot convert float NaN to integer` on a register value that
is NaN; that is a pre-existing defect in the script, unrelated to the path change, and
is left open.

### M2. Four required artifacts describe where MIMIC data physically lives, including a Google Drive mount

`results/accord/artifact_recovery.json` and `results/accord/reconciliation_c28.json`
(both read by `verify_thesis.py`) record the locations searched during artifact
recovery: "Google Drive thesis/outputs/, results_2026-08/, analysis_2026-08/,
code_2026-08/", "all of ~/mimic_local_only including runs/, accord*/ and
from_drive_2026-08-08/", "~/Downloads", "the quarantine".
`reconciliation_c28.json` adds that "run_experiments.py hard-codes the Google Drive
copy of annotation_sample_300.csv".

No restricted content, no identifiers. What it discloses is that credentialed MIMIC
derivatives were at one point on a cloud-synced Google Drive, which is a data-security
posture a PhysioNet reviewer could read as a DUA clause 4 issue. **Publishable, but
the author should decide deliberately**, because this is the kind of sentence that
invites a question. `accord/CORRECTIONS.md` (excluded from publication) states the
same fact more directly and withdraws an earlier claim that no MIMIC data was
cloud-synced.

**Separately: no file in the closure READS from a cloud-synced directory at runtime.**
`accord/cloud_paths.py` is the module that resolves Google Drive paths and it is NOT
in the closure; the eight scripts that import it are all outside the closure. Every
path opened by the six commands is repo-relative.

### M3. The repository does not currently pass its own gates, and the minimal set inherits that exactly

Not a disclosure issue, but it would be visible to anyone who clones the published
repository and runs the commands. Two failures, both pre-existing in the full
repository and reproduced identically in the minimal tree:

- `verify_thesis_deep.py` **FAILS on pass 11**: the appendix prints 6,336 total /
  5,247 citable / 659 distinct / 199 unique / 460 ambiguous; the tools compute
  6,524 / 5,435 / 660 / 198 / 462. Running `accord/stats/sync_appendix_counts.py`
  and rebuilding is the documented fix.
- `trace_claims.py` exits 1 with **3 untraced numbers**, all of them a decimal
  fragment split across a LaTeX interval (`38.8`, `44.1` in ch1 and `38.8` in ch4).

Both should be closed before publication, or the README must say the gate is
expected to fail and why.

### M4. `accord/thesis/main_msu.toc` is in the closure but `.gitignore` excludes it, so a fresh clone runs pass 9 blind

167 of the 168 files in the include set are tracked by git. The one exception is
`accord/thesis/main_msu.toc`, which is matched by the LaTeX-byproduct rules in
`.gitignore`. `verify_thesis_deep.py` pass 9 reads it to check that every TOC entry
carries the `Chapter N:` prefix FORMAT.md requires, and that check is guarded by
`if toc.exists()`. So in any clone of the repository as it stands today, running
`verify_thesis_deep.py` before `build.sh` reports "pass: 0 template violations"
having never looked at a table of contents. The comment above that code says the
pass "reported zero template violations for months" for exactly this reason, and
the fix that was applied made the gate read the TOC without making the TOC present.

Same shape for `main_msu.pdf`, which IS tracked, so the page-count and
internal-filename checks do have something to read.

**Two ways to close it.** Either commit `main_msu.toc` alongside `main_msu.pdf`
(the manifest assumes this: it is in the include set), or make the published README
state that `bash build.sh` must run before `verify_thesis_deep.py`. Committing it is
the safer of the two, because the gate cannot be made to fail by forgetting an order.

### M5. `fig_batch3.py` builds two required figures and no gate runs it, so its data input is invisible to the closure

`verify_thesis_deep.py` pass 10 rebuilds figures by running the five scripts named in
its own `FIG_SCRIPTS` list. `accord/thesis/figures/fig_batch3.py` is not one of them.
It builds `fig-scale-ladder.pdf` and `fig-answer-polarity.pdf`, and
`check_figure_invariants.py` FAILS with "pending inclusion but its PDF is absent"
if either PDF is missing, so both are load-bearing. Pass 10 reports
"7 of 7 bound figures rebuild to the committed PDF" and is telling the truth: the two
figures are built but not yet bound into a chapter, so they fall outside the set it
checks, and nothing at all checks that they still rebuild.

This is CLAUDE.md's own "gate that cannot fail" pattern, one entry later in the list.

The practical consequence for publication is concrete. `fig_batch3.py` reads
`results/accord/framing_mechanism/answer_polarity.json`. Because no gate runs
`fig_batch3.py`, no gate opens that artifact, so the automated closure did not
contain it, and a published repository built from the automated closure would have
shipped a generator that raises `FileNotFoundError` on its first line of work. It was
found by running the generator directly and is now in the include set.

**Fix: add `"fig_batch3.py"` to `FIG_SCRIPTS` in `accord/stats/verify_thesis_deep.py`.**
`accord/publication_manifest/build_manifest.py` already runs it for this reason, so
the manifest is correct either way, but the gate is not.

---

## LOW / CLEARED

### L1. No clinical note text anywhere in the closure. Cleared.
Every JSON in the include set was scanned for long strings carrying clinical
markers (dose units, route abbreviations, drug names). The only two hits are
self-attestations that say the opposite: `artifact_recovery.json` ("This file
contains no clinical note text and no patient or admission identifiers of any
kind") and `clinical_core/displayed_exemplar_lower_bound.json` ("No clinical note
text and no patient or admission identifier appears in this artifact; asserted in
code before write"). `pdftotext` over `main_msu.pdf` and all 25 figure PDFs found
no note text either.

### L2. No `HADM_ID` / `SUBJECT_ID` / `ICUSTAY_ID` VALUES. Cleared.
Six files in the closure contain those tokens; in every case it is a **column name
in a methodological sentence**, never a value:
`c132.json` and `c133.json` (`n_gold_rows_missing_subject_id: 0`),
`exemplar/step2plus_results.json` ("the pair_id<->HADM_ID join needed for the real
test does not exist"), `reconciliation_c28.json` (a join-key description),
`step0_1_join_and_calibration.json` ("audit.csv keys on HADM_ID+drug"), and
`main_msu.pdf` line 1343 ("SUBJECT_ID, asserted in code rather than assumed").
No integer identifier appears. Publishable as is.

### L3. No API keys, no AWS account numbers. Cleared.
Scanned for `sk-`, `AKIA`, bearer tokens, `api_key`, `aws_secret`, and both account
numbers named in `CLAUDE.md` (`495599736555`, `162774483375`). Zero hits in the
closure.

### L4. No institutional or personal email address. Cleared.
Zero email addresses in any text file in the closure.
`make_appendices.py` documents that the Google Drive mount path *would* carry the
operator's institutional email and redacts it; no such path survives in the closure.

### L5. `accord/dictionaries.py` is medication vocabulary, not MIMIC content. Cleared.
Brand-to-generic mappings (`lasix`/`furosemide`, `coumadin`/`warfarin`) and negation
and action regexes. It is described as reused verbatim from the reference notebook
but the content is public drug nomenclature. Publishable.

### L6. `accord/scaling/C44_convergence_check.ipynb` has zero stored outputs. Cleared.
8 cells, 0 with outputs. The one cell comment says "No Google Drive mount. Public
AG News data only -- no MIMIC, no credentialed data in this notebook".

### L7. `results/accord/multillm/per_model_results.json` is local-only. Cleared.
Models are `llama3:8b`, `granite3.3:8b`, `qwen3:8b`, `qwen3:14b` through local Ollama.
Prompts use `{nursing_note}` / `{discharge_summary}` placeholders. No hosted output,
no note text.

### L8. `CLAUDE.md` is referenced from inside five closure files. Cosmetic.
`verify_thesis.py`, `verify_thesis_deep.py`, `build.sh`, `measure_ink.py` and
`trace_claims.py` cite `CLAUDE.md` in comments as the source of a rule. The published
repo will therefore point at a file it does not contain. Harmless, but a reader will
notice. Either rename the referenced document to something that is published, or
leave it.

---

## Publishability verdict

**The minimal set cannot be published exactly as it stands.** Three changes are
mandatory and two are decisions:

MANDATORY
1. Redact the absolute path in `results/accord/c135_dissociation_cost/c135_bias_curve.json`
   (M1). One field. Nothing else in the closure has to change to make it safe.
2. Add `LICENSE`, `NOTICE`, `README.md` and `requirements.txt`; they are not in the
   reproduction closure but a public repository needs them, and the README has to
   state the run order and the two known gate failures.
3. Close or document the two pre-existing gate failures (M3), so a cloner's first run
   is not a red gate they cannot interpret.
4. Add `fig_batch3.py` to `FIG_SCRIPTS` in `verify_thesis_deep.py` (M5), so the two
   newest figures are covered by the gate that is supposed to cover figures.

DECISIONS FOR THE AUTHOR
5. Whether the published `results/accord/VERIFIED_FACTS.json` keeps its 823
   `withdrawn.hosted.*` keys (H1). Stripping them was tested and costs nothing but a
   regenerated Appendix B.
6. Whether the thesis keeps naming Claude Sonnet 5, Claude Haiku 4.5, the Anthropic
   API, AWS Bedrock, Llama 3 70B / 8B and GPT-4o in Chapter 4 (H2). That decision, not
   the packaging, is what determines whether `accord/hosted_withdrawal.py` discloses
   anything.

With change 1 applied, nothing in the include set discloses restricted data,
credentials, or patient information.

## Bottom line

The closure contains **no clinical text, no patient identifiers, no credentials and
no cloud-synced read**. Two things must be decided before it goes public: what the
published register says about the withdrawn hosted runs (H1), and whether the thesis
keeps naming the hosted vendors (H2, which decides whether `hosted_withdrawal.py` is
a disclosure at all). One file needs a one-line redaction (M1). One posture statement
deserves a deliberate decision (M2). Two pre-existing gate failures should be closed
so a cloner does not see a red gate on first run (M3).
