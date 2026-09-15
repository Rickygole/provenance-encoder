# Source summaries

Aggregate count files written by the original pipeline runs (cohort construction, medication extraction and
candidate mining, and the detector's rule-type head result). They hold scalar counts only: no note text, no
identifiers. The cohort and mining scripts that produce the first files are in `accord/` (for example
`accord/phase1_cohort.py`, `accord/phase3_mining.py`); `accord_v2_results.json` was exported from a training
notebook that is not in this repository.
