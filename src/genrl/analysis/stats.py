"""Statistics helpers: CLT intervals, bootstrap over problems, chi-square intervals."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_ci(per_problem: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and CLT interval for a score that averages per-problem values (Eq. 2.125-2.129).

    Pass per-problem means c̄(x) so that samples of one problem are clustered
    (Miller, 2024).
    """
    x = np.asarray(per_problem, dtype=float)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    z = stats.norm.ppf(0.5 + level / 2)
    return m, m - z * se, m + z * se


def paired_diff_ci(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Interval for the difference of two models scored on the same problems."""
    return mean_ci(np.asarray(a) - np.asarray(b), level)


def bootstrap_problems(
    fn, *arrays: np.ndarray, n_boot: int = 1000, level: float = 0.95, axis: int = -1, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap of fn(*arrays) resampling problems along `axis` (Eq. 2.138)."""
    rng = np.random.default_rng(seed)
    n = arrays[0].shape[axis]
    point = fn(*arrays)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = fn(*(np.take(a, idx, axis=axis) for a in arrays))
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return point, lo, hi


def variance_chi2_ci(estimate: float, dof: int, level: float = 0.95) -> tuple[float, float]:
    """Chi-square interval for a variance estimated with `dof` degrees of freedom.

    With S seeds, dof = S - 1; S = 5 gives roughly 0.36x to 8.3x the estimate.
    """
    a = (1 - level) / 2
    return (
        dof * estimate / stats.chi2.ppf(1 - a, dof),
        dof * estimate / stats.chi2.ppf(a, dof),
    )
