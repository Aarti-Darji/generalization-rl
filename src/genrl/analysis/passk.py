"""Unbiased pass@k estimator (Chen et al., 2021) and curves for Part B."""

from __future__ import annotations

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """1 - C(n-c, k) / C(n, k), computed stably."""
    if k > n:
        raise ValueError("k must be <= n")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def pass_at_k_curve(correct: np.ndarray, ks: list[int]) -> np.ndarray:
    """Mean pass@k over problems.

    correct: (n_problems, n_samples) 0/1 array. Returns an array aligned with ks.
    """
    correct = np.asarray(correct)
    n = correct.shape[1]
    counts = correct.sum(axis=1).astype(int)
    return np.array([np.mean([pass_at_k(n, c, k) for c in counts]) for k in ks])
