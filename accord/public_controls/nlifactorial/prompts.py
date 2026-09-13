"""C104 frozen prompts: negative hardness x prompt x model scale factorial.

Committed once, before any datum, per C104_PREREGISTRATION.md. Never edited
after the first run.

P0 is a plain neutral contradiction question. P1 mirrors the STRUCTURE of the
clinical baseline prompt (names a conflict type / rule id in the system
message and again in the question, before asking) with no clinical content
at all -- a generic "RULE" label, not a medication or clinical claim. P2 is
P0 plus a stated base rate, with no abstention option offered.

All three share one answer vocabulary (Contradiction / Not), the same one
graded_score_probe.py and accord/prompt_sensitivity/prompts.py's P_asserting
use, so a single parser and a single log-odds convention serve all of C98,
C103 and C104.
"""

POS_TOKENS = {"contr", "Contr", "con", "Con", "CONTR", "Contradiction"}
NEG_TOKENS = {"not", "Not", "NOT", "N", "No"}

P0_SYSTEM = "Answer with one word: Contradiction or Not."
P0_USER = (
    "Sentence 1: {premise}\n"
    "Sentence 2: {hypothesis}\n"
    "Do these two sentences contradict each other?"
)

P1_SYSTEM = (
    "You are checking sentence pairs for a RULE-7 LOGICAL CONTRADICTION: a "
    "case where the second sentence directly negates, reverses, or is "
    "incompatible with a claim made by the first sentence. "
    "Answer with one word: Contradiction or Not."
)
P1_USER = (
    "Sentence 1: {premise}\n"
    "Sentence 2: {hypothesis}\n"
    "Does sentence 2 contradict sentence 1 under RULE-7?"
)

P2_SYSTEM = (
    "About 5 in 100 sentence pairs you will see are contradictions. "
    "Answer with one word: Contradiction or Not."
)
P2_USER = P0_USER

PROMPTS = {
    "P0": {"system": P0_SYSTEM, "user_template": P0_USER},
    "P1": {"system": P1_SYSTEM, "user_template": P1_USER},
    "P2": {"system": P2_SYSTEM, "user_template": P2_USER},
}

PROMPT_NAMES = ["P0", "P1", "P2"]
