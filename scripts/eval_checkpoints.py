"""Sample every checkpoint on the eval pool and append graded rows to the log.

Examples:
  # every saved GRPO checkpoint of one seed, R = 4 samples per problem
  python scripts/eval_checkpoints.py --checkpoints outputs/grpo/seed0 --model-tag grpo --seed 0
  # base / SFT / final models with 64 samples for pass@k
  python scripts/eval_checkpoints.py --model Qwen/Qwen2.5-1.5B --model-tag base --final
"""

import argparse
import re
from pathlib import Path

from genrl.config import load_config
from genrl.data import load_eval_pool
from genrl.eval import save_log
from genrl.eval.sample import sample_and_grade


def find_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    ckpts = []
    for d in run_dir.iterdir():
        m = re.fullmatch(r"checkpoint-(\d+)", d.name)
        if m and d.is_dir():
            ckpts.append((int(m.group(1)), d))
    return sorted(ckpts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval/pool.yaml")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoints", type=Path, help="run dir with checkpoint-N subdirs")
    g.add_argument("--model", help="single model path or hub id")
    ap.add_argument("--model-tag", required=True, help="base / sft / grpo")
    ap.add_argument("--seed", type=int, default=-1, help="training seed of the run")
    ap.add_argument("--final", action="store_true", help="use samples_per_problem_final")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pool = load_eval_pool(cfg["datasets"])
    n_samples = cfg["samples_per_problem_final" if args.final else "samples_per_problem"]
    targets = (
        find_checkpoints(args.checkpoints) if args.checkpoints else [(0, Path(args.model))]
    )
    for step, path in targets:
        ckpt_id = f"{args.model_tag}/seed{args.seed}/step{step}"
        print(f"evaluating {ckpt_id} ({path}) with {n_samples} samples/problem")
        df = sample_and_grade(
            str(path), pool, n_samples,
            temperature=cfg["temperature"], top_p=cfg["top_p"], max_tokens=cfg["max_tokens"],
            seed=cfg["seed"], save_raw_text=cfg["save_raw_text"],
            meta={"model": args.model_tag, "seed": args.seed, "step": step,
                  "checkpoint": ckpt_id},
        )
        out = cfg["output"].replace(".parquet", "_final.parquet") if args.final else cfg["output"]
        save_log(df, out)
