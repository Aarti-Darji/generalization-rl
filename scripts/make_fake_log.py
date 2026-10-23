"""Write a synthetic log with the real schema, to develop the analyses without a GPU.

Checkpoint correlation comes from a shared per-problem difficulty; seeds add a
run-level offset and a seed x problem effect.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.special import expit

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/logs/fake_samples.parquet")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20, help="checkpoints per seed")
    ap.add_argument("--problems", type=int, default=1800)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--rng", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.rng)
    difficulty = rng.normal(0, 1.5, args.problems)
    rows = []
    for s in range(args.seeds):
        seed_offset = rng.normal(0, 0.1)
        seed_prob = rng.normal(0, 0.4, args.problems)
        for t in range(1, args.steps + 1):
            step = 25 * t
            learn = 0.6 * (1 - np.exp(-t / 6))
            logit = -difficulty + learn + seed_offset + seed_prob + rng.normal(0, 0.3, args.problems)
            c = rng.random((args.problems, args.samples)) < expit(logit)[:, None]
            pid, r = np.meshgrid(np.arange(args.problems), np.arange(args.samples), indexing="ij")
            rows.append(pd.DataFrame({
                "model": "grpo", "seed": s, "step": step,
                "checkpoint": f"grpo/seed{s}/step{step}",
                "problem_id": [f"p{i:05d}" for i in pid.ravel()],
                "source": "fake", "sample_idx": r.ravel(),
                "extracted": "", "reference": "", "correct": c.ravel(), "text": "",
            }))
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df):,} rows to {args.out}")
