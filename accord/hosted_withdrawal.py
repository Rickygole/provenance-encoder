"""Which register keys came through a hosted service, and are therefore withdrawn.

WHY THIS EXISTS
---------------
PhysioNet's guidance "Use of MIMIC Data with Large Language Models and Online
Services", published 24 September 2025, prohibits sharing credentialed data
"including sending it through APIs or using it on online platforms" and states
that MIMIC data "must not be stored or retained by third-party LLM services".
It replaced an April 2023 post that had named four suggested services, and it
withdrew that list on the stated ground that PhysioNet "cannot verify the data
practices of external services". Every run this thesis made through a hosted
service therefore has no safe harbour, and every result downstream of one is
withdrawn rather than qualified.

The withdrawal is enforced here rather than asserted in prose. This project has
already recorded what happens when a withdrawal lives only in a sentence: the
GPT-4o rows were described in the results chapter as withdrawn while all 25 of
their keys sat live in the register, so the provenance gate would have resolved
a withdrawn digit anywhere in the thesis without complaint (C89).

A key is hosted if the artifact it was read from names a hosted channel. The
test is on the artifact string, not the key name, because key names are chosen
by whoever registered them and the artifact path is written by the generator.
Two tokens are deliberately not pattern matched: "llama3" and "llama38b" name
the LOCAL Ollama labeller, while "metallama38binstructv1" names the Bedrock one,
and a substring rule over model families would have collapsed them. That is the
same class of defect as the model tag that once landed in the wrong filename
position and scored one model's predictions against another's artifact.

Nothing here deletes a value. Withdrawn keys are relocated under
WITHDRAWN_NS so the correction record can still quote what was withdrawn, and
the verification gate excludes that namespace from the citable set.
"""

import functools
import json
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Substrings that identify a hosted channel inside a registered artifact string.
# Checked against the whole artifact field, which carries both the file path and
# the node path inside it.
HOSTED_ARTIFACT_TOKENS = (
    "frontier_anthropic",          # Claude Sonnet 5 and Claude Haiku 4.5 metrics
    "frontier_bedrock",            # Llama 3 70B and Llama 3 8B metrics
    "frontier_label_noise_claudehaiku",
    "transmission_ledger",         # the ledger of the transmissions themselves
    # node paths inside the derived analyses, which key off the run label
    "::anthropic",
    "::bedrock",
    "anthropic_few_shot",
    "anthropic_zero_shot",
    "anthropic_claudehaiku",
    "bedrock_metallama",
    "sonnet_zero_shot_minus_detector",
    "sonnet_few_shot_minus_detector",
    "haiku_zero_shot_minus_detector",
    "haiku_few_shot_minus_detector",
    "llama70b_few_shot_minus_detector",
    "llama70b_zero_shot_minus_detector",
    "llama8b_few_shot_minus_detector",
    "llama8b_zero_shot_minus_detector",
    "gpt4o",
    # Model names appearing in a key that was registered from a shared artifact
    # whose path names no channel. The five stats.*_mcc_ci_at_corrected_alpha.*
    # keys are exactly this case: the file is a general interval computation, and
    # only the key says whose interval it is. "llama70b" and "metallama" are
    # hosted-only slugs; the local Ollama 8B is reached through LOCAL_KEY_TOKENS
    # below, which is tested first.
    "sonnet",
    "haiku",
    "claude",
    "llama70b",
    "metallama",
    # Quantities that only ever existed as quotients of hosted rows. The rebuilt
    # capability ladder is local-only and reuses the same artifact path, so the
    # path cannot identify them; these names can, and no local analysis produces
    # them. Without these, a regenerated hosted quotient would pass as local.
    "_70b",
    "channel_effect_at_8b",
    "across_family_step_c67",
)

# Local tokens win over hosted ones. capability_ladder.json holds both the
# hosted rungs and the two locally served ones, so the file alone cannot decide
# a key; the model slug in the key name does.
LOCAL_KEY_TOKENS = (
    "granite33",
    "granite338b",
    ".llama38b.",
    "ollama",
)

# Keys that name a stratum rather than a labeller. These are properties of the
# gold standard, identical across every arm, and the defect-free denominators
# every surviving number is read against. They are registered from a hosted
# artifact only because the hosted arm happened to be the first node in the
# file; withdrawing them would delete 108 and 6 from the thesis and break every
# reading of the defect-free stratum. They are re-sourced, not withdrawn.
STRATUM_KEYS = ("defectfree.n", "defectfree.n_positives")

WITHDRAWN_NS = "withdrawn.hosted"

# Relocation is idempotent: a key already under the withdrawn namespace is not
# withdrawn again. quoted_historical.* is deliberately NOT exempt. That prefix
# exists so prose can cite a withdrawn figure as history while the gate still
# treats it as live, and a hosted value reached through it would be a hosted
# digit printed in the thesis by another route. There is no reading of the
# guidance under which the number becomes usable because the sentence around it
# calls it old.
ALREADY_WITHDRAWN_PREFIXES = ("withdrawn.",)


def _provenance_inputs(path):
    """Every results artifact this artifact references, anywhere in its body.

    The first version read only a provenance block's input fields, and C96 showed
    why a narrow reading fails: a derived file can name its hosted source in any
    field. The capability ladder names its rows under ladder[].source, not under
    provenance. So every string in the artifact that is a path to a results JSON is
    followed. A path that does not exist on disk is still classified by its name,
    which is how a withdrawn source that has been moved away is still caught.
    """
    try:
        d = json.loads((_ROOT / path).read_text())
    except Exception:
        return ()
    found = []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and o.startswith("results/") and ".json" in o:
            found.append(o.split("::")[0].strip())

    walk(d)
    # The register is not an input artifact. It holds every value, including the
    # withdrawn hosted ones under withdrawn.hosted.*, and each of its entries is
    # classified on its own. Following it would mark every analysis that reads a
    # base rate or a threshold from the register as hosted, which is how the first
    # version of this walk flagged 1,158 local and public values. Skipping it is
    # safe because a generator that read a withdrawn.hosted.* key would have to
    # name that key, and the gate refuses any such key as citable.
    return tuple(dict.fromkeys(f for f in found
                               if f != path and not f.endswith("VERIFIED_FACTS.json")))


@functools.lru_cache(maxsize=None)
def _derives_from_hosted(path, _depth=0):
    """True if this artifact was computed from a hosted one, transitively.

    A token test on the artifact's own path is not enough, and this project has
    already shipped the proof: frontier_label_noise.json holds a hosted model's
    per-pair cells, but neither its filename nor any of its 100 register key
    names contains a channel token, so a name-based test called it local and
    left every one of those values citable. Its provenance block says plainly
    that its input was frontier_anthropic_metrics.json. The same failure is
    recorded in CLAUDE.md from an earlier round, where 96 labelnoise keys
    survived the correction that withdrew the script behind them. So the test
    follows what an artifact says it was computed from, and a derived value is
    withdrawn if anything upstream of it is.

    Depth-limited because a provenance field is free text and could cycle.
    """
    if _depth > 6:
        return False
    lower = str(path).lower()
    if any(tok in lower for tok in HOSTED_ARTIFACT_TOKENS):
        return True
    return any(_derives_from_hosted(src, _depth + 1)
               for src in _provenance_inputs(path))


def is_hosted(key, rec):
    """True if this registered value came through a third-party hosted service."""
    if key in STRATUM_KEYS:
        return False
    if key.startswith(ALREADY_WITHDRAWN_PREFIXES):
        return False
    lower_key = key.lower()
    if any(tok in lower_key for tok in LOCAL_KEY_TOKENS):
        return False
    artifact = str((rec or {}).get("artifact") or "")
    haystack = (artifact + " " + lower_key).lower()
    if any(tok in haystack for tok in HOSTED_ARTIFACT_TOKENS):
        return True
    # Follow the artifact's declared provenance. The file part is everything
    # before the "::" node path, which is what can be opened.
    return _derives_from_hosted(artifact.split("::")[0].strip())


def withdrawn_name(key):
    return f"{WITHDRAWN_NS}.{key}"
