"""Part B: variance components of final-checkpoint scores, and pass@k curves."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from genrl.analysis.passk import pass_at_k_curve
from genrl.analysis.stats import bootstrap_problems, variance_chi2_ci
from genrl.analysis.variance_components import (
    estimate_components,
    per_problem_split,
    seed_disagreement,
)
from genrl.eval import correctness_tensor, load_log

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="results/logs/samples.parquet")
    ap.add_argument("--final-log", help="64-sample log of base / sft / final grpo, for pass@k")
    ap.add_argument("--steps", type=int, nargs="*", help="training steps to decompose "
                    "(default: last step only)")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", default="results/part_b")
    args = ap.parse_args()

    df = load_log(args.log, columns=["model", "seed", "step", "problem_id", "sample_idx",
                                     "correct"])
    df = df[df["model"] == "grpo"]
    steps = args.steps or [int(df["step"].max())]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for step in steps:
        c, seeds, problems = correctness_tensor(df[df["step"] == step], by="seed")  # (S, n, R)
        vc = estimate_components(c)
        _, lo, hi = bootstrap_problems(lambda x: estimate_components(x).problem, c,
                                       n_boot=args.n_boot, axis=1)
        p = c.mean(axis=2)
        split = per_problem_split(p)
        results[step] = {
            "components": {"seed": vc.seed, "problem": vc.problem,
                           "interaction": vc.interaction, "sampling": vc.sampling},
            "n_truncated": vc.n_truncated,
            "seed_ci_chi2": variance_chi2_ci(vc.seed, vc.S - 1),
            "problem_ci_boot": (lo, hi),
            "run_score_sd": float(np.sqrt(vc.run_score_variance())),
            "run_score_sd_fixed_benchmark": float(np.sqrt(vc.run_score_variance(fixed_benchmark=True))),
            "n_star": vc.n_star,
            "between_seed_share": float(split["between_seed"].sum() / split["total"].sum()),
            "disagreement_problems": [problems[i] for i in seed_disagreement(p)][:50],
        }
    (out / "variance_components.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({s: r["components"] | {"n_star": r["n_star"]} for s, r in results.items()},
                     indent=2, default=float))

    if args.final_log:
        fl = load_log(args.final_log, columns=["model", "seed", "checkpoint", "problem_id",
                                               "sample_idx", "correct"])
        ks = [1, 2, 4, 8, 16, 32, 64]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for tag in ["base", "sft"]:
            c, _, _ = correctness_tensor(fl[fl["model"] == tag])
            ax.plot(ks, pass_at_k_curve(c[0], ks), "o-", label=tag)
        c, _, _ = correctness_tensor(fl[fl["model"] == "grpo"])
        curves = np.array([pass_at_k_curve(ci, ks) for ci in c])
        ax.plot(ks, curves.mean(axis=0), "o-", label="GRPO (mean over seeds)")
        ax.fill_between(ks, curves.min(axis=0), curves.max(axis=0), alpha=0.2)
        ax.set(xscale="log", xlabel="k", ylabel="pass@k", title="pass@k: base vs SFT vs GRPO")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "pass_at_k.png", dpi=150)
