"""Download the off-the-shelf NLI baseline model. Network. Never opens MIMIC data.

WHY THIS EXISTS
---------------
C100 needs FacebookAI/roberta-large-mnli, pinned at a commit SHA, served fully
offline and under a network-egress sandbox once the run starts. Model
downloads must happen BEFORE and SEPARATELY from any process that opens a
MIMIC file (see CLAUDE.md and the compliance rules this session runs under),
so the download is its own script, has no CSV path anywhere in it, and writes
only a manifest: the commit SHA, a SHA-256 per downloaded file, and the
licence string, none of which is patient data.

What is fetched, and nothing else: config.json, model.safetensors,
tokenizer.json, tokenizer_config.json, vocab.json, merges.txt. No optional
weights format, no training checkpoint.

Writes:
  ~/models_public/roberta-large-mnli/          the model files
  results/accord/differential/nli_model_manifest.json

    python3 accord/differential/fetch_nli_model.py
"""

import hashlib
import json
import pathlib

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "FacebookAI/roberta-large-mnli"
LOCAL_DIR = pathlib.Path.home() / "models_public" / "roberta-large-mnli"
ALLOW_PATTERNS = ["config.json", "model.safetensors", "tokenizer.json",
                  "tokenizer_config.json", "vocab.json", "merges.txt"]
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "accord" / "differential" / "nli_model_manifest.json"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    api = HfApi()
    info = api.model_info(REPO_ID)
    commit_sha = info.sha
    licence = None
    for tag in (info.tags or []):
        if tag.startswith("license:"):
            licence = tag.split(":", 1)[1]
            break
    if licence is None:
        card = api.model_info(REPO_ID, files_metadata=False)
        licence = getattr(card, "cardData", {}).get("license") if getattr(
            card, "cardData", None) else None

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPO_ID, revision=commit_sha,
                      local_dir=str(LOCAL_DIR), allow_patterns=ALLOW_PATTERNS)

    files = {}
    for pattern in ALLOW_PATTERNS:
        p = LOCAL_DIR / pattern
        if p.exists():
            files[pattern] = sha256_of(p)
    missing = [p for p in ALLOW_PATTERNS if p not in files]

    manifest = {
        "repo_id": REPO_ID,
        "commit_sha": commit_sha,
        "licence": licence,
        "local_dir": str(LOCAL_DIR),
        "files_sha256": files,
        "files_missing": missing,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  repo={REPO_ID}  commit_sha={commit_sha}  licence={licence}")
    print(f"  {len(files)} file(s) downloaded, {len(missing)} missing")
    for name, digest in files.items():
        print(f"    {name:24s} {digest}")
    if missing:
        print(f"  MISSING: {missing}")


if __name__ == "__main__":
    main()
