"""Part A: measured optimism of best-checkpoint selection vs. three predictions."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from genrl.analysis import selection_bias as sb
from genrl.data import split_pool
from genrl.eval import correctness_tensor, load_log

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="results/logs/samples.parquet")
    ap.add_argument("--ns", type=int, nargs="+", default=[30, 100, 300])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="keep every k-th checkpoint")
    ap.add_argument("--seed-filter", type=int, nargs="*", help="restrict to these seeds")
    ap.add_argument("--out", default="results/part_a")
    args = ap.parse_args()

    df = load_log(args.log, columns=["checkpoint", "seed", "step", "problem_id", "sample_idx",
                                     "correct", "model"])
    df = df[df["model"] == "grpo"]
    if args.seed_filter:
        df = df[df["seed"].isin(args.seed_filter)]
    steps = np.sort(df["step"].unique())[:: args.every]
    df = df[df["step"].isin(steps)]

    c, ckpts, problems = correctness_tensor(df)
    c_bar = c.mean(axis=2)                                  # (K, N)
    fit_ids, split_ids = split_pool(problems, seed=args.split_seed)
    pos = {p: i for i, p in enumerate(problems)}
    fit = c_bar[:, [pos[p] for p in fit_ids]]
    split = c_bar[:, [pos[p] for p in split_ids]]

    A, Sigma = sb.fit_moments(fit)
    sigma = float(np.sqrt(np.mean(np.diag(Sigma))))
    corr = np.corrcoef(fit)
    K = len(ckpts)
    ns = [n for n in args.ns if n < split.shape[1]]
    if len(ns) < len(args.ns):
        print(f"skipping n >= split-half size {split.shape[1]}")
    results = {"K": K, "meff_nyholt": sb.meff_nyholt(corr), "meff_li_ji": sb.meff_li_ji(corr),
               "mean_pairwise_corr": float(corr[np.triu_indices(K, 1)].mean()), "by_n": []}
    for n in ns:
        res = sb.split_simulation(split, n, n_reps=args.reps)
        results["by_n"].append({
            "n": n,
            "optimism_vs_oracle": res.optimism_vs_oracle,
            "optimism_vs_pool": res.optimism_vs_pool,
            "hoeffding_bound": sb.hoeffding_bound(K, n),
            "independent_k": sb.independent_k_formula(sigma, K, n),
            "gaussian_plugin": sb.gaussian_plugin(A, Sigma, n),
        })

    meta = df.drop_duplicates("checkpoint").set_index("checkpoint").loc[ckpts]
    results["selection_rules"] = {
        n: sb.selection_rules(split, meta["seed"].to_numpy(), meta["step"].to_numpy(), n,
                              n_reps=args.reps)
        for n in ns
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    rows = results["by_n"]
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ns, [r["optimism_vs_pool"] for r in rows], "ko-", label="measured (vs pool)")
    for key, label in [("hoeffding_bound", "Hoeffding bound"),
                       ("independent_k", "independent-K"), ("gaussian_plugin", "Gaussian plug-in")]:
        ax.plot(ns, [r[key] for r in rows], "--", label=label)
    ax.set(xscale="log", xlabel="selection set size n", ylabel="optimism",
           title=f"Winner's curse, K = {K}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "optimism_vs_n.png", dpi=150)
    print(json.dumps(rows, indent=2))
