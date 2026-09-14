"""C109 frozen prompts: three NEW paraphrases of the reversed question,
none used in C103, plus the ORIGINAL P_reversed re-run as a determinism
check. See C109_PREREGISTRATION.md.

Committed once, before any datum. Never edited after the first run; a
change here is a new pre-registration, not a fix ("no further paraphrases
added after results" -- C109_PREREGISTRATION.md's kill condition).

All four keep the "reversed" convention P_reversed established in
accord/prompt_sensitivity/prompts.py: the question is phrased so that a
"No" answer is the flag (the two notes do NOT agree / could NOT both be
followed / are NOT compatible), and the recorded flag_logodds is
log p(No) - log p(Yes) (higher = more conflict), matching P_asserting and
P_neutral's orientation. This file only defines new wording; the sign flip
itself lives in one place, run_c109.py, exactly as C103 keeps it in
run_c103.py.

P_REVERSED_RERUN is not retyped here. It is the SAME dict object C103 froze
in prompts.py (PROMPTS["P_reversed"]), imported rather than copied, so the
determinism check compares C109's re-run against a byte-identical prompt,
not a hand-retyped approximation of one.
"""
from accord.prompt_sensitivity.prompts import PROMPTS as C103_PROMPTS
from accord.prompt_sensitivity.prompts import YESNO_NEG, YESNO_POS

P_REV_AGREE_SYSTEM = "Answer with one word: Yes or No."
P_REV_AGREE_USER = (
    "Nursing note: {nursing_note}\n"
    "Discharge summary: {discharge_summary}\n"
    "Do these two notes agree on the medication instruction? "
    "Answer Yes or No."
)

P_REV_FOLLOW_SYSTEM = "Answer with one word: Yes or No."
P_REV_FOLLOW_USER = (
    "Nursing note: {nursing_note}\n"
    "Discharge summary: {discharge_summary}\n"
    "Could a clinician follow both of these medication instructions at "
    "the same time? Answer Yes or No."
)

P_REV_COMPATIBLE_SYSTEM = "Answer with one word: Yes or No."
P_REV_COMPATIBLE_USER = (
    "Nursing note: {nursing_note}\n"
    "Discharge summary: {discharge_summary}\n"
    "Is the discharge summary compatible with the nursing note about the "
    "medication instruction? Answer Yes or No."
)

PROMPTS_C109 = {
    "P_rev_agree": {
        "system": P_REV_AGREE_SYSTEM,
        "user_template": P_REV_AGREE_USER,
        "pos_tokens": YESNO_POS,
        "neg_tokens": YESNO_NEG,
        # "reversed": a "No" (they do NOT agree) is the flag; see
        # run_c109.py for where the sign/label flip is applied.
        "reversed": True,
    },
    "P_rev_follow": {
        "system": P_REV_FOLLOW_SYSTEM,
        "user_template": P_REV_FOLLOW_USER,
        "pos_tokens": YESNO_POS,
        "neg_tokens": YESNO_NEG,
        "reversed": True,
    },
    "P_rev_compatible": {
        "system": P_REV_COMPATIBLE_SYSTEM,
        "user_template": P_REV_COMPATIBLE_USER,
        "pos_tokens": YESNO_POS,
        "neg_tokens": YESNO_NEG,
        "reversed": True,
    },
    # Determinism check: the exact C103 P_reversed prompt, re-run under
    # C109's harness. Imported, not retyped.
    "P_reversed_rerun": C103_PROMPTS["P_reversed"],
}

# New-paraphrase names only (excludes the determinism-check rerun), for the
# primary decision rule, which is stated over the three new paraphrases.
NEW_PARAPHRASE_NAMES = ["P_rev_agree", "P_rev_follow", "P_rev_compatible"]
ALL_PROMPT_NAMES = NEW_PARAPHRASE_NAMES + ["P_reversed_rerun"]
