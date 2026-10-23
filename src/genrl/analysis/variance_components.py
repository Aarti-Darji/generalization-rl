"""Part B: random-effects decomposition of run-to-run variation.

    c_{s,i,r} = mu + a_s + b_i + g_si + e_sir

a_s: training seed, b_i: problem, g_si: seed x problem, e_sir: sampling noise.
Estimated from balanced two-way ANOVA mean squares (method of moments).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VarianceComponents:
    seed: float          # sigma_a^2
    problem: float       # sigma_b^2
    interaction: float   # sigma_g^2
    sampling: float      # sigma_e^2
    n_truncated: int     # how many raw estimates were negative and set to 0
    S: int
    n: int
    R: int

    def run_score_variance(self, n: int | None = None, R: int | None = None,
                           fixed_benchmark: bool = False) -> float:
        """Var(one run's score) = sa2 + (sb2 + sg2)/n + se2/(nR).

        With fixed_benchmark=True the problem term is dropped.
        """
        n = n or self.n
        R = R or self.R
        sb2 = 0.0 if fixed_benchmark else self.problem
        return self.seed + (sb2 + self.interaction) / n + self.sampling / (n * R)

    @property
    def n_star(self) -> float:
        """Benchmark size beyond which more problems barely help against the seed floor."""
        if self.seed == 0:
            return float("inf")
        return (self.problem + self.interaction) / self.seed


def mean_squares(c: np.ndarray) -> dict[str, float]:
    """Mean squares for a balanced (S seeds, n problems, R replicates) 0/1 array."""
    S, n, R = c.shape
    grand = c.mean()
    m_s = c.mean(axis=(1, 2))
    m_i = c.mean(axis=(0, 2))
    m_si = c.mean(axis=2)
    ss_seed = n * R * np.sum((m_s - grand) ** 2)
    ss_prob = S * R * np.sum((m_i - grand) ** 2)
    ss_int = R * np.sum((m_si - m_s[:, None] - m_i[None, :] + grand) ** 2)
    ss_err = np.sum((c - m_si[..., None]) ** 2)
    return {
        "seed": ss_seed / (S - 1),
        "problem": ss_prob / (n - 1),
        "interaction": ss_int / ((S - 1) * (n - 1)),
        "error": ss_err / (S * n * (R - 1)),
    }


def estimate_components(c: np.ndarray) -> VarianceComponents:
    """Solve the expected-mean-square equations; truncate negatives at zero.

    E[MS_seed] = se2 + R sg2 + nR sa2
    E[MS_int]  = se2 + R sg2
    E[MS_prob] = se2 + R sg2 + SR sb2
    E[MS_err]  = se2
    """
    S, n, R = c.shape
    if S < 2 or n < 2 or R < 2:
        raise ValueError("need S, n, R >= 2")
    ms = mean_squares(c)
    raw = {
        "sampling": ms["error"],
        "interaction": (ms["interaction"] - ms["error"]) / R,
        "seed": (ms["seed"] - ms["interaction"]) / (n * R),
        "problem": (ms["problem"] - ms["interaction"]) / (S * R),
    }
    n_neg = sum(v < 0 for v in raw.values())
    est = {k: max(v, 0.0) for k, v in raw.items()}
    return VarianceComponents(**est, n_truncated=n_neg, S=S, n=n, R=R)


def per_problem_split(p: np.ndarray) -> dict[str, np.ndarray]:
    """Split each problem's pooled Bernoulli variance into within- and between-seed parts.

    p: (S, n) solve probabilities per seed and problem.
    p̄(1 - p̄) = E_s[p_s(1 - p_s)] + Var_s(p_s)   (exact, with population variance)
    """
    p_bar = p.mean(axis=0)
    return {
        "total": p_bar * (1 - p_bar),
        "within_seed": np.mean(p * (1 - p), axis=0),
        "between_seed": np.var(p, axis=0),
    }


def seed_disagreement(p: np.ndarray, hi: float = 0.75, lo: float = 0.25) -> np.ndarray:
    """Indices of problems that some seed solves reliably (>= hi) and another rarely (<= lo)."""
    return np.flatnonzero((p.max(axis=0) >= hi) & (p.min(axis=0) <= lo))
