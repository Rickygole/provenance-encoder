"""C103 frozen prompts for item 6 (prompt sensitivity on G_299 / G_253).

Committed once, before any datum, per C103_PREREGISTRATION.md. Never edited
after the first run; a change here is a new pre-registration, not a fix.

P_ASSERTING is copied VERBATIM (system_prompt, user_prompt) from the
config_id "gpt4o_zeroshot" row of the existing project's frozen prompt
table, prompts_existing.csv (the same row run_frontier_compliant.py loads
via load_prompts(path, "gpt4o_zeroshot")). It is the recovered clinical
baseline prompt: it names a conflict type / rule before asking whether the
two sentences conflict. It is reproduced here character-for-character, not
retyped, by a script that read the source row and wrote this file without
ever printing the row to a console or a log (see
accord/prompt_sensitivity/extract_p_asserting.py for that step, run once
against the local mimic_local_only prompt table -- a template file with
{nursing_note}/{discharge_summary} placeholders and no patient content).

P_NEUTRAL and P_REVERSED are authored fresh for this pre-registration. They
keep the same placeholder scheme ({nursing_note}, {discharge_summary}) so
all three prompts can share one calling convention, but neither names a
conflict type or a rule id.

All three are sent over /api/chat as a system message plus a user message,
exactly as graded_score_probe.py does; the gold columns nursing_sentence and
discharge_sentence are substituted into {nursing_note} and
{discharge_summary} respectively at call time.
"""


P_ASSERTING_SYSTEM = 'You are a clinical pharmacist reviewing medication reconciliation between nursing progress notes and hospital discharge summaries. Your task is to identify dangerous medication contradictions. Respond with exactly one word: contradiction or not_contradiction.'

P_ASSERTING_USER = 'Nursing Progress Note:\n{nursing_note}\n\nDischarge Summary:\n{discharge_summary}\n\nIs there a dangerous medication contradiction between these two notes?'

P_NEUTRAL_SYSTEM = (
    "Answer with one word: Yes or No."
)

P_NEUTRAL_USER = (
    "Nursing note: {nursing_note}\n"
    "Discharge summary: {discharge_summary}\n"
    "Do these two sentences give conflicting medication instructions? "
    "Answer Yes or No."
)

P_REVERSED_SYSTEM = (
    "Answer with one word: Yes or No."
)

P_REVERSED_USER = (
    "Nursing note: {nursing_note}\n"
    "Discharge summary: {discharge_summary}\n"
    "Are these two sentences consistent with each other about medication "
    "instructions? Answer Yes or No."
)

# Answer vocabularies. Each entry maps the first token (case-insensitive,
# stripped of punctuation) to a flag: 1 means "flagged as a conflict",
# 0 means "not flagged". P_REVERSED's vocabulary is inverted at the
# point of use (a "no" -- not consistent -- is the flag), never by
# rewriting these sets, so the mapping used for scoring is visible in one
# place: prompt_sensitivity/run_c103.py.
ASSERTING_POS = {"contr", "Contr", "con", "Con", "CONTR", "Contradiction"}
ASSERTING_NEG = {"not", "Not", "NOT", "N", "No"}

YESNO_POS = {"yes", "Yes", "YES", "y", "Y"}
YESNO_NEG = {"no", "No", "NO", "n", "N"}

PROMPTS = {
    "P_asserting": {
        "system": P_ASSERTING_SYSTEM,
        "user_template": P_ASSERTING_USER,
        "pos_tokens": ASSERTING_POS,
        "neg_tokens": ASSERTING_NEG,
        "reversed": False,
    },
    "P_neutral": {
        "system": P_NEUTRAL_SYSTEM,
        "user_template": P_NEUTRAL_USER,
        "pos_tokens": YESNO_POS,
        "neg_tokens": YESNO_NEG,
        "reversed": False,
    },
    "P_reversed": {
        "system": P_REVERSED_SYSTEM,
        "user_template": P_REVERSED_USER,
        "pos_tokens": YESNO_POS,
        "neg_tokens": YESNO_NEG,
        # "reversed" means the raw parsed label and log-odds sign are
        # flipped before comparison with the other two prompts: a "no"
        # (not consistent) becomes the positive/flag class, and
        # log-odds = log p(No) - log p(Yes) so that higher still means
        # "more conflict flagged", matching P_asserting and P_neutral.
        "reversed": True,
    },
}
