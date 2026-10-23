"""Synthetic sanity check of the Part A math (proposal, "Numeric check").

Correlated checkpoints share a per-problem latent difficulty. True accuracies
come from a large pool, so the measured optimism can be compared with the
Hoeffding bound, the independent-K formula and the Gaussian plug-in. This
checks the algebra and the estimators, not real model behaviour.
"""

import argparse

import numpy as np
from scipy.special import expit

from genrl.analysis import selection_bias as sb

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--problems", type=int, default=20_000)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--ns", type=int, nargs="+", default=[30, 100, 300])
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--rng", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.rng)
    K, N = args.K, args.problems
    difficulty = rng.normal(0, 1.5, N)
    skill = rng.normal(0, 0.1, K)
    logit = skill[:, None] - difficulty[None, :] + rng.normal(0, 0.8, (K, N))
    c_bar = (rng.random((K, N, args.samples)) < expit(logit)[..., None]).mean(axis=2)

    A, Sigma = sb.fit_moments(c_bar)       # with N this large, A is the true accuracy
    corr = np.corrcoef(c_bar)
    sigma = float(np.sqrt(np.mean(np.diag(Sigma))))
    print(f"K={K}  mean pairwise corr={corr[np.triu_indices(K, 1)].mean():.2f}  "
          f"Meff Nyholt={sb.meff_nyholt(corr):.0f}  Li-Ji={sb.meff_li_ji(corr):.0f}")
    print(f"{'n':>5} {'measured':>9} {'hoeffding':>10} {'indep-K':>8} {'plug-in':>8}")
    for n in args.ns:
        opt = []
        for _ in range(args.reps):
            idx = rng.choice(N, n, replace=False)
            scores = c_bar[:, idx].mean(axis=1)
            k = int(np.argmax(scores))
            opt.append(scores[k] - A[k])
        print(f"{n:>5} {np.mean(opt):>9.3f} {sb.hoeffding_bound(K, n):>10.3f} "
              f"{sb.independent_k_formula(sigma, K, n):>8.3f} "
              f"{sb.gaussian_plugin(A, Sigma, n):>8.3f}")
