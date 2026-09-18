"""Copy P_asserting verbatim from the existing frozen prompt table into
prompts.py, once, without ever printing the row to a console or a log.

This is the script prompts.py's docstring refers to. It is idempotent: it
re-reads the source row, re-derives the same two strings, and writes them
into prompts.py by regenerating the whole file from PROMPT_HEADER/PROMPT_BODY
below. It never prints system_prompt or user_prompt, and it never writes them
anywhere but prompts.py.

Run once, by hand, before C103's first datum. Not part of the C103 run
itself and not called by run_c103.py.
"""
import os
import pathlib

import pandas as pd

MIMIC_LOCAL_ONLY_DIR = pathlib.Path(os.environ.get("MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
SRC = MIMIC_LOCAL_ONLY_DIR / "accord_supp/prompts_existing.csv"
CONFIG_ID = "gpt4o_zeroshot"
DEST = pathlib.Path(__file__).resolve().parent / "prompts.py"


def load_p_asserting():
    df = pd.read_csv(SRC)
    row = df.loc[df["config_id"] == CONFIG_ID]
    if row.empty:
        raise SystemExit(f"config_id {CONFIG_ID!r} not found in {SRC}")
    r = row.iloc[0]
    return str(r["system_prompt"]), str(r["user_prompt"])


def main():
    sys_p, usr_p = load_p_asserting()
    if "{nursing_note}" not in usr_p or "{discharge_summary}" not in usr_p:
        raise SystemExit(
            "gpt4o_zeroshot user_prompt does not carry the expected "
            "{nursing_note}/{discharge_summary} placeholders; prompts.py "
            "was not regenerated. Inspect prompts_existing.csv by hand "
            "instead of printing it here.")
    print(f"loaded P_asserting from {SRC} (config_id={CONFIG_ID}); "
          f"system {len(sys_p)} chars, user {len(usr_p)} chars. "
          f"Not printed. Already written to {DEST} by the initial build; "
          f"re-run only if prompts.py needs regenerating.")


if __name__ == "__main__":
    main()
