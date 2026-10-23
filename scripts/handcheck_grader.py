"""Export a random sample of graded completions for manual checking of the grader.

Fill in the `human_correct` column, then rerun with --score to get the agreement rate.
"""

import argparse

import pandas as pd

from genrl.eval import load_log

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="results/logs/samples.parquet")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/handcheck.csv")
    ap.add_argument("--score", action="store_true", help="score a filled-in handcheck file")
    args = ap.parse_args()

    if args.score:
        df = pd.read_csv(args.out)
        agree = (df["correct"].astype(bool) == df["human_correct"].astype(bool)).mean()
        print(f"grader agrees with hand labels on {agree:.1%} of {len(df)} samples")
        print(df[df["correct"].astype(bool) != df["human_correct"].astype(bool)])
    else:
        df = load_log(args.log).sample(n=args.n, random_state=0)
        df = df[["checkpoint", "problem_id", "reference", "extracted", "correct", "text"]]
        df["human_correct"] = ""
        df.to_csv(args.out, index=False)
        print(f"wrote {len(df)} samples to {args.out}")
