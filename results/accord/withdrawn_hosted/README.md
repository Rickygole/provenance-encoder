# Withdrawn: results obtained through hosted LLM services

Every artifact in this directory was produced by sending sentence pairs extracted
from MIMIC-III notes to a third party's hosted model service. PhysioNet's guidance
of 24 September 2025, *Use of MIMIC Data with Large Language Models and Online
Services*, states that the Credentialed Health Data Use Agreement prohibits
sharing access to the data with third parties "including sending it through APIs
or using it on online platforms", and that "MIMIC data must not be stored or
retained by third-party LLM services".

No number reported in the thesis derives from anything here. The numeric
provenance gate in `accord/stats/verify_thesis.py` enforces that by holding every
value read from these files under the `withdrawn.hosted.` namespace of the
register, which is excluded from the set a reported figure may resolve against.
The classification is in `accord/hosted_withdrawal.py` and the relocation in
`accord/stats/withdraw_hosted.py`.

These files are retained rather than deleted so that the correction record can say
what was withdrawn and what each withdrawal cost. They are excluded from the
deposit because `scripts/export_results.py` builds it from an allow-list of the
artifacts the paper's numbers come from, and these files are not on it.

Whether the transmissions were consistent with the agreement is a determination
for PhysioNet and for the institution, not one made here.

## Where the files are

The per-pair JSON outputs were removed from the tracked tree on 2026-09-14 and are held in
credentialed local storage only. Earlier commits of this repository still contain them; they carry
model predictions keyed by sequential pair index and no note text or identifiers.
