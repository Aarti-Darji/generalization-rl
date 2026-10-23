"""Part A: winner's curse in checkpoint selection.

Inputs are per-problem mean correctness c̄_k(x), shape (K, N): K checkpoints,
N problems. The optimism of picking k* = argmax_k Â_k on n problems is
Â_{k*} - A_{k*}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, stats


# --- Predictions -------------------------------------------------------------


def hoeffding_bound(K: int, n: int) -> float:
    """sqrt(log K / 2n): holds for any dependence between checkpoints (Eq. 2.106-2.109)."""
    return float(np.sqrt(np.log(K) / (2 * n)))


def expected_max_std_normal(K: int) -> float:
    """E[max of K i.i.d. standard normals]."""
    if K == 1:
        return 0.0
    f = lambda z: z * K * stats.norm.pdf(z) * stats.norm.cdf(z) ** (K - 1)  # noqa: E731
    return integrate.quad(f, -np.inf, np.inf)[0]


def independent_k_formula(sigma: float, K: int, n: int) -> float:
    """sigma * E[max of K standard normals] / sqrt(n): assumes independent scores."""
    return sigma * expected_max_std_normal(K) / np.sqrt(n)


def gaussian_plugin(
    A: np.ndarray, Sigma: np.ndarray, n: int, n_sims: int = 20_000, seed: int = 0
) -> float:
    """Correlation-aware prediction: Z ~ N(0, Sigma/n), k* = argmax(A + Z), return E[Z_{k*}].

    A and Sigma should come from the fit half only.
    """
    rng = np.random.default_rng(seed)
    K = len(A)
    L = np.linalg.cholesky(Sigma / n + 1e-12 * np.eye(K))
    Z = rng.standard_normal((n_sims, K)) @ L.T
    k_star = np.argmax(A[None, :] + Z, axis=1)
    return float(Z[np.arange(n_sims), k_star].mean())


def fit_moments(c_bar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A_k and Sigma_kl = Cov_x(c̄_k(x), c̄_l(x)) from a (K, N) array."""
    return c_bar.mean(axis=1), np.cov(c_bar)


# --- Effective number of tests (descriptive only) ---------------------------


def meff_nyholt(corr: np.ndarray) -> float:
    lam = np.linalg.eigvalsh(corr)
    M = len(lam)
    return float(1 + (M - 1) * (1 - np.var(lam, ddof=1) / M))


def meff_li_ji(corr: np.ndarray) -> float:
    lam = np.abs(np.linalg.eigvalsh(corr))
    return float(np.sum((lam >= 1).astype(float) + (lam - np.floor(lam))))


# --- Split simulation --------------------------------------------------------


@dataclass
class SplitResult:
    n: int
    selected: np.ndarray      # index of k* in each replicate
    sel_score: np.ndarray     # Â_{k*} on the selection set
    oracle_score: np.ndarray  # k*'s score on the rest of the pool
    pool_score: np.ndarray    # k*'s score on the whole pool

    @property
    def optimism_vs_oracle(self) -> float:
        return float(np.mean(self.sel_score - self.oracle_score))

    @property
    def optimism_vs_pool(self) -> float:
        return float(np.mean(self.sel_score - self.pool_score))


def _argmax_random_ties(x: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(x == x.max())
    return int(rng.choice(best))


def split_simulation(c_bar: np.ndarray, n: int, n_reps: int = 1000, seed: int = 0) -> SplitResult:
    """Repeatedly draw n selection problems, pick the best checkpoint, score it on the rest.

    c_bar: (K, N) per-problem mean correctness on the split half of the pool.
    """
    rng = np.random.default_rng(seed)
    K, N = c_bar.shape
    pool = c_bar.mean(axis=1)
    selected = np.empty(n_reps, dtype=int)
    sel = np.empty(n_reps)
    orc = np.empty(n_reps)
    for r in range(n_reps):
        perm = rng.permutation(N)
        s_idx, o_idx = perm[:n], perm[n:]
        scores = c_bar[:, s_idx].mean(axis=1)
        k = _argmax_random_ties(scores, rng)
        selected[r], sel[r], orc[r] = k, scores[k], c_bar[k, o_idx].mean()
    return SplitResult(n, selected, sel, orc, pool[selected])


# --- Selection rules ---------------------------------------------------------


def selection_rules(
    c_bar: np.ndarray, seeds: np.ndarray, steps: np.ndarray, n: int, n_reps: int = 1000, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Compare checkpoint-selection rules by selection-set and oracle accuracy.

    seeds, steps: length-K arrays labelling each row of c_bar.
    Rules: per-run argmax, last checkpoint, mean of last three, and argmax of the
    across-seed mean curve (Eq. 2.140). Per-run rules are averaged over seeds.
    """
    rng = np.random.default_rng(seed)
    N = c_bar.shape[1]
    uniq_seeds = np.unique(seeds)
    uniq_steps = np.unique(steps)
    out: dict[str, list[tuple[float, float]]] = {
        "per_run_argmax": [], "last": [], "mean_last3": [], "argmax_mean_curve": []
    }
    for _ in range(n_reps):
        perm = rng.permutation(N)
        s_idx, o_idx = perm[:n], perm[n:]
        sel = c_bar[:, s_idx].mean(axis=1)
        orc = c_bar[:, o_idx].mean(axis=1)
        per_run, last, last3 = [], [], []
        for s in uniq_seeds:
            rows = np.flatnonzero(seeds == s)
            rows = rows[np.argsort(steps[rows])]
            k = rows[_argmax_random_ties(sel[rows], rng)]
            per_run.append((sel[k], orc[k]))
            last.append((sel[rows[-1]], orc[rows[-1]]))
            last3.append((sel[rows[-3:]].mean(), orc[rows[-3:]].mean()))
        out["per_run_argmax"].append(tuple(np.mean(per_run, axis=0)))
        out["last"].append(tuple(np.mean(last, axis=0)))
        out["mean_last3"].append(tuple(np.mean(last3, axis=0)))
        curve = np.array([sel[steps == t].mean() for t in uniq_steps])
        t_star = uniq_steps[_argmax_random_ties(curve, rng)]
        rows = steps == t_star
        out["argmax_mean_curve"].append((sel[rows].mean(), orc[rows].mean()))
    return {
        rule: {"selection": float(np.mean([v[0] for v in vals])),
               "oracle": float(np.mean([v[1] for v in vals]))}
        for rule, vals in out.items()
    }
