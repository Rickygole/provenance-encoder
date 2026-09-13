# provenance-encoder — working agreement

## The pipeline. This is the base. Every substantive change goes through it.

No edit to the thesis, the figures, the analysis code or the register is "done"
until it has been round-tripped through the loop below. Do not skip a stage
because a change looks small; the defects this project has actually shipped were
all small-looking. State which stage you are in when you report.

```
  1 DISCOVER   Agents in parallel. Find issues. Do not fix anything yet.
  2 FIX        I fix. Reviewers are read-only by design.
  3 RECHECK    Ten verification passes + figure gates + both PDFs build.
  4 COMMITTEE  Adversarial agents acting as a brutal thesis committee.
  5 GATE       Committee passes?  no -> back to 1 with their findings
                                  yes -> 6
  6 WRITE      Prose and figures to top-journal standard.
  7 DELIVER    Rebuild, refresh both PDFs on the Desktop, commit, push.
```

**The loop is not optional and not a formality.** It runs until the committee
stage returns no finding that would change a claim, a number, or a figure. Record
each round: what was found, what was fixed, what the committee said next.

### Stage 1, DISCOVER
Launch in ONE message so they run concurrently. Read every report yourself and
relay the substance; their reports are not shown to the user.
- `brutal-reviewer` — the harsh senior-reviewer pass
- `results-auditor` — re-derive every number from artifacts, check table/prose agreement
- `gavin-redteam` — the one thing that kills it
- `ron-lawyer` — MIMIC/PhysioNet, the data use agreement, disclosure (any round touching data, ethics, or release)
- `camera-ready-checker` — template and format compliance against FORMAT.md

### Stage 3, RECHECK. All of these, every round:
```
python3 accord/stats/verify_thesis.py          # 5 gating passes
python3 accord/stats/verify_thesis_deep.py     # 10 passes total
cd accord/thesis/figures && python3 tools/check_figure_invariants.py
cd accord/thesis/figures && python3 tools/measure_ink.py
python3 accord/stats/make_appendices.py && python3 accord/stats/trace_claims.py
cd accord/thesis && bash build.sh              # builds BOTH documents
```

### Stage 4, COMMITTEE. Brutal, not collegial.
- `paper-reviewer` — would this pass at a top venue
- `idea-stress-tester` — Reviewer 2 on the contribution and framing
- `monica-skeptic` — is the thesis defending the right claim at all
A committee finding is not closed by arguing with it. It is closed by a change,
or by a recorded reason in CORRECTIONS.md for why no change was needed.

### Stage 6, figures to journal standard
Every figure argues one claim, names it in the headline, and is readable in
greyscale at one column. `check_figure_invariants.py` and `measure_ink.py` are
floors, not goals; a figure that passes both and still says nothing is a failure.
No figure is "random": if it does not carry a claim the text depends on, delete it.

## Hard constraints. These override everything, including the pipeline.

- **MIMIC-III is PhysioNet-credentialed.** Local only. Aggregates only. Never
  print or write clinical note text. Never write HADM_ID or SUBJECT_ID outside
  `local_only/` or `mimic_local_only/`. Never echo an API exception body: it can
  quote the request back, and the request carries note sentences.
- **Never accept an API key pasted in chat.** Direct the user to revoke it.
- **Never use the `attacker-user` AWS account** (`495599736555`). Patient data goes
  only through accounts the user owns; the Bedrock account is `162774483375`.
- **refs.bib stays at 96 entries.** Only references the user supplied or approved. Never add one unasked.
  Verified by the user on 2026-09-13: Tarone (1990) against PubMed 2364136; Liu et al. (2019) RoBERTa
  against Semantic Scholar corpus 198953378; Saerens, Latinne and Decaestecker (2002) against MIT Press,
  doi 10.1162/089976602753284446.
- **No en dashes or em dashes anywhere.** Pass 1 gates this.
- **Nothing in prose that is not backed by an artifact.** Pass 2 gates the digits;
  the gate is a digit match, not a provenance proof, so it is a floor not a proof.
- **Primary sources only.** No Wikipedia.
- **Pre-register before running.** Commit the decision rule with a kill condition
  before the first datum. Record the outcome in the same file afterwards, and never
  edit the rule. Convention set by C44.
- **Every number is generated, never typed.** If a number reaches prose, a committed
  script put it in the register first.
- **Withdrawn and superseded values live under `withdrawn.*`** in the register.
  Every generator deletes its own prefix wholesale and will eat anything else
  parked there.

## Things that have bitten this project. Check them.

- A gate that cannot fail. `build.sh` counted TODOs in `ch*.tex`; the files are
  `msu_ch*.tex`, so it reported zero by scanning nothing.
- A second document going stale. The supplement was unbuilt for weeks and was a
  broken partial, because the harness reads `main_msu.pdf` only.
- A register key maintained by hand. All 96 `labelnoise.*` keys survived the
  correction that withdrew the script behind them.
- A model tag in the wrong filename position, twice, which scored one model's
  predictions against another's artifact.
- A count in prose that the chapter outgrew. Fourteen tests became sixteen became
  twenty, and the stated figure was always the one under which the headline cleared.

## Build

TeX packages live in the user tree, no root needed:
`tlmgr --usermode install apacite tocloft newtx chngcntr needspace placeins xurl`
`git` needs `export DEVELOPER_DIR=/Library/Developer/CommandLineTools`.
