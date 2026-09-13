"""Relocate every register value obtained through a hosted service, and gate on it.

The classification lives in accord/hosted_withdrawal.py; this script applies it.
Two modes:

  python3 accord/stats/withdraw_hosted.py            relocate, print a report
  python3 accord/stats/withdraw_hosted.py --check    exit nonzero if any live

The check mode is what makes the withdrawal a control rather than a claim. Every
generator that produces a hosted row rewrites its own namespace from scratch on
each run, so a relocation done once would silently come undone the next time the
register was rebuilt. Running the check at the end of the build catches that.

Relocation is idempotent and lossless: a key moves to
withdrawn.hosted.<original key> with its artifact and value intact, so the
correction record can quote what was withdrawn and Appendix A can say what each
withdrawal cost. Nothing is deleted here.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accord.hosted_withdrawal import is_hosted, withdrawn_name  # noqa: E402

REGISTER = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"


def main(check_only=False):
    facts = json.loads(REGISTER.read_text(),
                       object_pairs_hook=__import__("collections").OrderedDict)
    hosted = [k for k in list(facts) if is_hosted(k, facts[k])]

    if check_only:
        if hosted:
            print(f"FAIL: {len(hosted)} hosted-channel values are live in the "
                  f"register. Run withdraw_hosted.py.")
            for k in hosted[:20]:
                print(f"  {k}")
            if len(hosted) > 20:
                print(f"  ... and {len(hosted) - 20} more")
            return 1
        n_w = sum(1 for k in facts if k.startswith("withdrawn.hosted."))
        print(f"PASS: no hosted-channel value is live. "
              f"{n_w} are held under withdrawn.hosted for the record.")
        return 0

    if not hosted:
        print("Nothing to relocate; every hosted value is already withdrawn.")
        return 0

    out = __import__("collections").OrderedDict()
    moved = 0
    for k, v in facts.items():
        if is_hosted(k, v):
            out[withdrawn_name(k)] = v
            moved += 1
        else:
            out[k] = v
    REGISTER.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Relocated {moved} values to withdrawn.hosted.*  "
          f"({len(out) - moved} remain live of {len(out)})")
    ns = {}
    for k in hosted:
        ns[k.split(".")[0]] = ns.get(k.split(".")[0], 0) + 1
    for k, n in sorted(ns.items(), key=lambda kv: -kv[1]):
        print(f"  {k:46s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(check_only="--check" in sys.argv))
