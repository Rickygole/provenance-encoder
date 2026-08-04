"""Train one arm. Fresh process per run -- MPS leaks memory across configs.

Compute-matched by construction: identical data order, identical token budget,
identical steps and optimizer across arms. The ONLY difference is the config
flags, which is the entire point of the arm design.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.bundles import BundleDataset  # noqa: E402
from src.data.lineage import hash_file, hash_task_splits  # noqa: E402
from src.model.config import preset  # noqa: E402
from src.model.encoder import ProvenanceEncoder  # noqa: E402
from src.train.instrument import Instrumentor  # noqa: E402

LOCAL = ROOT / "local_only"


def check_lineage(task_dir: Path, bundles_dir: Path, tokenizer_dir: Path) -> str:
    """Refuse to pretrain on a stale or mismatched corpus. See
    src/data/lineage.py and scripts/build_bundles.py's docstring for the
    contamination this closes: bundles used to be built from an independent
    subject split and could contain the task's val/test patients' discharge
    summaries. Every hash here is recomputed fresh, never cached.
    """
    bundles_meta_path = bundles_dir / "meta.json"
    if not bundles_meta_path.exists():
        raise SystemExit(
            f"{bundles_meta_path} missing -- run scripts/build_bundles.py "
            f"(after scripts/build_task.py) before pretraining."
        )
    bundles_meta = json.loads(bundles_meta_path.read_text())
    stored_task_hash = bundles_meta.get("task_split_hash")
    if stored_task_hash is None:
        raise SystemExit(
            "local_only/bundles/meta.json has no task_split_hash -- this "
            "corpus predates the contamination fix (it was built with an "
            "independent subject shuffle instead of the task's split). "
            "Rebuild with the current scripts/build_bundles.py, then retrain "
            "the tokenizer with scripts/train_tokenizer.py, before pretraining."
        )
    fresh_task_hash, task_counts = hash_task_splits(task_dir)
    if fresh_task_hash != stored_task_hash:
        raise SystemExit(
            f"STALE PRETRAINING CORPUS: local_only/bundles was built from "
            f"task_split_hash={stored_task_hash}, but {task_dir} currently "
            f"hashes to {fresh_task_hash} (subjects: {task_counts}). The "
            f"downstream task split has changed since the bundles were built "
            f"-- rerun scripts/build_bundles.py, then scripts/train_tokenizer.py, "
            f"before pretraining. Refusing to silently pretrain on a corpus "
            f"that may now include the task's current val/test patients."
        )

    tok_lineage_path = tokenizer_dir / "lineage.json"
    if not tok_lineage_path.exists():
        raise SystemExit(
            f"{tok_lineage_path} missing -- retrain the tokenizer with the "
            f"current scripts/train_tokenizer.py (it now stamps a lineage.json)."
        )
    tok_lineage = json.loads(tok_lineage_path.read_text())
    train_jsonl = bundles_dir / "train.jsonl"
    fresh_bundles_hash = hash_file(train_jsonl)
    if tok_lineage.get("bundles_content_hash") != fresh_bundles_hash:
        raise SystemExit(
            f"STALE TOKENIZER: {tok_lineage_path} was trained on bundles "
            f"content hash {tok_lineage.get('bundles_content_hash')}, but "
            f"{train_jsonl} currently hashes to {fresh_bundles_hash}. The "
            f"bundle corpus was rebuilt since the tokenizer was trained -- "
            f"rerun scripts/train_tokenizer.py before pretraining."
        )
    if tok_lineage.get("task_split_hash") != stored_task_hash:
        raise SystemExit(
            f"STALE TOKENIZER: {tok_lineage_path}'s task_split_hash "
            f"({tok_lineage.get('task_split_hash')}) does not match the "
            f"current bundles' task_split_hash ({stored_task_hash}). Rerun "
            f"scripts/build_bundles.py then scripts/train_tokenizer.py."
        )
    print(f"lineage OK: task_split_hash={stored_task_hash} "
          f"(subjects {task_counts}) -> bundles -> tokenizer, all matched.")
    return stored_task_hash


def device_of(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


POS_BINS = [(0, 8), (8, 20), (20, 40), (40, 1000)]


def evaluate(model, loader, dev, max_batches: int, by_position: bool = False):
    """Mean MLM loss, optionally broken down by position WITHIN a segment.

    The position breakdown is a free diagnostic for the boilerplate confound: at
    seq 256 with ~6 segments, each note contributes only ~36 tokens, so training
    sits almost entirely in note OPENINGS -- the most template-heavy, most
    role-stereotyped region. If a provenance advantage is concentrated in the
    first few tokens of each segment and decays to zero, the mechanism is
    template identification under aggressive truncation, not provenance.
    """
    model.eval()
    tot, n = 0.0, 0
    bin_sum = [0.0] * len(POS_BINS)
    bin_cnt = [0] * len(POS_BINS)
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= max_batches:
                break
            b = {k: v.to(dev) for k, v in b.items()}
            h = model(b["input_ids"], b["local_positions"], b["attention_mask"],
                      b["role_ids"], b["auth_ids"], b["event_times"])
            logits = model.mlm_logits(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   b["mlm_labels"].view(-1), ignore_index=-100)
            if torch.isfinite(loss):
                tot += float(loss)
                n += 1
            if by_position:
                per_tok = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), b["mlm_labels"].view(-1),
                    ignore_index=-100, reduction="none").view(b["mlm_labels"].shape)
                valid = b["mlm_labels"] != -100
                lp = b["local_positions"]
                for bi, (lo, hi) in enumerate(POS_BINS):
                    sel = valid & (lp >= lo) & (lp < hi)
                    c = int(sel.sum())
                    if c:
                        bin_sum[bi] += float(per_tok[sel].sum())
                        bin_cnt[bi] += c
    model.train()
    mean = tot / max(n, 1)
    if not by_position:
        return mean
    breakdown = {f"pos_{lo}_{hi}": (bin_sum[i] / bin_cnt[i] if bin_cnt[i] else None)
                 for i, (lo, hi) in enumerate(POS_BINS)}
    breakdown_n = {f"pos_{lo}_{hi}_n": bin_cnt[i] for i, (lo, hi) in enumerate(POS_BINS)}
    return mean, {**breakdown, **breakdown_n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A1", "A2", "A3", "A4"])
    ap.add_argument("--size", default="10M")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--seg-tokens", type=int, default=36)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--permute-roles", action="store_true",
                    help="control: fixed random bijection over roles (structure kept, "
                         "semantics destroyed). A gain that survives this is not provenance.")
    ap.add_argument("--random-roles", action="store_true",
                    help="control: independent random role per segment (structure destroyed too)")
    ap.add_argument("--length-roles", action="store_true",
                    help="control: bias keyed on note-length bucket instead of role. An "
                         "informative per-segment category with no authority semantics.")
    ap.add_argument("--segment-index-roles", action="store_true",
                    help="A5 control: key the bias on ordinal position in the bundle "
                         "(zero provenance meaning). Requires --shuffle-segments.")
    ap.add_argument("--shuffle-segments", action="store_true",
                    help="randomize segment order within a bundle, decorrelating role "
                         "from position (bundles are built chronologically)")
    ap.add_argument("--global-positions", action="store_true",
                    help="T0-1: standard absolute positions instead of "
                         "local-within-segment. Without this the vanilla baseline has "
                         "no global position coordinate and every 'segmentation' gain "
                         "may just be the model buying position back.")
    ap.add_argument("--rpe", action="store_true",
                    help="T0-2: T5-style relative position bias, the comparison class")
    ap.add_argument("--hard-block", action="store_true",
                    help="T0-4: fixed block-diagonal mask, the limit of the learned bias")
    ap.add_argument("--misaligned-blocks", action="store_true",
                    help="T0-3: same block count, boundaries NOT at document edges. "
                         "Isolates document structure from any coarse blocking.")
    ap.add_argument("--alibi", action="store_true",
                    help="A6 control: replace the segment bias with a pure ALiBi "
                         "distance bias using NO segment or role information")
    ap.add_argument("--tag", default="", help="suffix for the run directory")
    ap.add_argument("--task-dir", default=str(LOCAL / "task"),
                    help="downstream task split the bundles must be derived from "
                         "(lineage check); override only for smoke-testing")
    ap.add_argument("--bundles-dir", default=str(LOCAL / "bundles"),
                    help="override for smoke-testing against a scratch corpus")
    ap.add_argument("--tokenizer-dir", default=str(LOCAL / "tokenizer"),
                    help="override for smoke-testing against a scratch tokenizer")
    args = ap.parse_args()
    if sum([args.permute_roles, args.random_roles, args.length_roles,
            args.segment_index_roles]) > 1:
        raise SystemExit("role controls are mutually exclusive")
    if args.segment_index_roles and not args.shuffle_segments:
        raise SystemExit(
            "--segment-index-roles without --shuffle-segments is meaningless: bundles "
            "are chronological, so segment index and role are the same variable and "
            "this would just re-run the role condition."
        )
    if args.permute_roles:
        print("WARNING: a global role derangement is VACUOUS -- the model relearns the "
              "permuted table and the loss is unchanged by construction. Do not cite.")

    task_dir = Path(args.task_dir)
    bundles_dir = Path(args.bundles_dir)
    tokenizer_dir = Path(args.tokenizer_dir)
    check_lineage(task_dir, bundles_dir, tokenizer_dir)

    torch.manual_seed(args.seed)
    dev = device_of(args.device)
    tok = Tokenizer.from_file(str(tokenizer_dir / "wordpiece.json"))

    cfg = preset(args.size, arm=args.arm)
    cfg.vocab_size = tok.get_vocab_size()
    cfg.max_seq_len = args.seq_len
    cfg.alibi_bias = args.alibi
    cfg.rpe_bias = args.rpe
    cfg.hard_block_mask = args.hard_block
    if (args.alibi or args.rpe or args.hard_block) and not cfg.use_authority_bias:
        raise SystemExit("bias controls need an arm with a bias module (A3 or A4)")
    if sum([args.alibi, args.rpe, args.hard_block]) > 1:
        raise SystemExit("bias controls are mutually exclusive")
    model = ProvenanceEncoder(cfg).to(dev)

    # Step count is part of the identity: a 1000-step and a 4000-step run of the
    # same arm are different experiments and must not overwrite each other.
    ctrl = ("_permuted" if args.permute_roles else "_randroles" if args.random_roles
            else "_lenroles" if args.length_roles
            else "_segidx" if args.segment_index_roles else "")
    ctrl += "_alibi" if args.alibi else ""
    ctrl += "_rpe" if args.rpe else ""
    ctrl += "_hardblk" if args.hard_block else ""
    ctrl += "_misalign" if args.misaligned_blocks else ""
    ctrl += "_globalpos" if args.global_positions else ""
    ctrl += "_shuf" if args.shuffle_segments else ""
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = (LOCAL / "runs" /
               f"{args.arm}_{args.size}_st{args.steps}_s{args.seed}{ctrl}{tag}")
    inst = Instrumentor(model, run_dir, args.arm)

    mk = lambda split, seed: BundleDataset(  # noqa: E731
        bundles_dir / f"{split}.jsonl", tok, arm=args.arm,
        max_seq_len=args.seq_len, max_tokens_per_segment=args.seg_tokens, seed=seed,
        permute_roles=args.permute_roles, random_roles=args.random_roles,
        length_roles=args.length_roles, segment_index_roles=args.segment_index_roles,
        shuffle_segments=args.shuffle_segments,
        global_positions=args.global_positions,
        misaligned_blocks=args.misaligned_blocks)
    train_ds, val_ds = mk("train", args.seed), mk("val", 12345)
    # Same seed => same data order across arms. Compute-matching requires it.
    g = torch.Generator().manual_seed(args.seed)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                           generator=g, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))

    def lr_at(s: int) -> float:
        if s < args.warmup:
            return args.lr * s / max(args.warmup, 1)
        p = (s - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    print(f"arm={args.arm} size={args.size} device={dev} | {cfg.summary()}")
    print(f"non-embedding params: {model.n_params(True):,} | total: {model.n_params(False):,}")
    print(f"train bundles: {len(train_ds):,}  steps: {args.steps}  "
          f"tokens/step: {args.batch * args.seq_len:,}")

    step, t0, tokens = 0, time.time(), 0
    hist = []
    model.train()
    while step < args.steps:
        for b in train_dl:
            if step >= args.steps:
                break
            b = {k: v.to(dev) for k, v in b.items()}

            if step < 20 or step % 500 == 0:
                stats = inst.check_inputs(b)
                stats.update(inst.check_bias_effective(b))
                inst.check_gauge()
                if step % 500 == 0:
                    inst.log(step, phase="guard", **stats)

            for gparam in opt.param_groups:
                gparam["lr"] = lr_at(step)

            h = model(b["input_ids"], b["local_positions"], b["attention_mask"],
                      b["role_ids"], b["auth_ids"], b["event_times"])
            logits = model.mlm_logits(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   b["mlm_labels"].view(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            inst.accumulate_grads()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tokens += args.batch * args.seq_len
            step += 1

            if step % 50 == 0:
                el = time.time() - t0
                print(f"  step {step:>5}/{args.steps}  loss {float(loss):.4f}  "
                      f"{tokens/el:,.0f} tok/s", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                vl = evaluate(model, val_dl, dev, args.eval_batches)
                mv = inst.movement()
                inst.log(step, phase="eval", train_loss=float(loss), val_loss=vl,
                         val_ppl=math.exp(min(vl, 20)), movement=mv)
                hist.append({"step": step, "val_loss": vl})
                print(f"  [eval] step {step}  val_loss {vl:.4f}  ppl {math.exp(min(vl,20)):.1f}",
                      flush=True)
                inst.assert_alive(step / args.steps)

    rep = inst.final_report()
    final_val, pos_breakdown = evaluate(model, val_dl, dev, args.eval_batches * 2,
                                        by_position=True)
    summary = {
        "arm": args.arm, "size": args.size, "seed": args.seed, "steps": args.steps,
        "tokens": tokens, "non_emb_params": model.n_params(True),
        "final_val_loss": final_val, "final_val_ppl": math.exp(min(final_val, 20)),
        "wall_seconds": time.time() - t0, "history": hist,
        "provenance_params": rep["provenance_params"],
        "loss_by_segment_position": pos_breakdown,
        "controls": {"permute_roles": args.permute_roles, "random_roles": args.random_roles,
                     "length_roles": args.length_roles,
                     "segment_index_roles": args.segment_index_roles,
                     "shuffle_segments": args.shuffle_segments,
                     "global_positions": args.global_positions},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # Save weights. Every run before this one produced scalars and nothing else,
    # so no pretrained encoder existed to fine-tune. Optimizer state is omitted
    # deliberately: disk is tight and these runs are cheap to repeat.
    torch.save({"model": model.state_dict(),
                "cfg": {k: getattr(cfg, k) for k in
                        ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff",
                         "max_seq_len", "n_roles", "n_auth_buckets", "arm",
                         "alibi_bias", "rpe_bias", "hard_block_mask")}},
               run_dir / "model.pt")
    print(f"saved weights -> {run_dir / 'model.pt'}")
    print(f"\nFINAL arm={args.arm} val_loss={final_val:.4f} "
          f"ppl={math.exp(min(final_val,20)):.2f} ({time.time()-t0:.0f}s)")
    for k, m in rep["provenance_params"].items():
        print(f"  {k:<12} moved {m['abs_movement']:.4e}  final_norm {m['final_norm']:.4e}  "
              f"mean_grad {m['mean_grad_norm']:.2e}")


if __name__ == "__main__":
    main()
