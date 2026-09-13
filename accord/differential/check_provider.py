"""Connectivity and credential check for the compliant re-run.

Sends ONE request built from a synthetic sentence pair that is invented here
in this file. No MIMIC data, no annotation sheet, no note text of any kind is
read or transmitted. Run this before the real script so a credential, model
name or network problem is diagnosed without touching credentialed data.

    export ANTHROPIC_API_KEY='...'
    python3 accord/differential/check_provider.py
"""

import json
import os
import sys
import time

import requests

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

# Invented for this check. Not from MIMIC, not from any patient record.
SYSTEM = "Answer with exactly one word: contradiction or not_contradiction."
USER = ("Nursing note: Furosemide held today, patient hypovolemic.\n"
        "Discharge summary: Continue furosemide 40 mg daily.\n"
        "Is there a dangerous medication contradiction between these two notes?")


def main():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ANTHROPIC_API_KEY is not set in this shell.")
    # The key prefix was printed here. Even seven characters reduces entropy in
    # any captured terminal log, and it bought nothing a boolean does not.
    print('    credential: present' if key else '    credential: MISSING')
    print(f"model      : {MODEL}")
    print("payload    : synthetic sentence pair, no MIMIC data\n")

    t0 = time.time()
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": MODEL, "system": SYSTEM, "max_tokens": 50,
                  
                  "messages": [{"role": "user", "content": USER}]},
            timeout=30)
    except requests.RequestException as exc:
        sys.exit(f"network error after {time.time() - t0:.1f}s: "
                 f"{type(exc).__name__}: {exc}")

    print(f"HTTP {r.status_code} in {time.time() - t0:.1f}s")
    if r.status_code != 200:
        # Safe to show in full: the payload is synthetic.
        print(json.dumps(r.json(), indent=2)[:900])
        sys.exit(1)

    j = r.json()
    print("reply      :", repr(j["content"][0]["text"].strip()))
    print("usage      :", j.get("usage"))
    print("\nProvider reachable and the credential works. "
          "Now run run_frontier_compliant.py.")


if __name__ == "__main__":
    main()
