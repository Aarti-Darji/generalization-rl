from itertools import combinations
from math import comb

import numpy as np

from genrl.analysis.passk import pass_at_k, pass_at_k_curve


def test_pass_at_k_matches_combinatorial_formula():
    for n in (4, 10):
        for c in range(n + 1):
            for k in range(1, n + 1):
                expected = 1 - comb(n - c, k) / comb(n, k)
                assert abs(pass_at_k(n, c, k) - expected) < 1e-12


def test_pass_at_k_matches_brute_force_subsets():
    outcomes = [1, 0, 0, 1, 0, 0]
    k = 3
    subsets = list(combinations(outcomes, k))
    brute = np.mean([any(s) for s in subsets])
    assert abs(pass_at_k(len(outcomes), sum(outcomes), k) - brute) < 1e-12


def test_pass_at_1_is_accuracy():
    rng = np.random.default_rng(0)
    correct = rng.random((50, 16)) < 0.3
    assert abs(pass_at_k_curve(correct, [1])[0] - correct.mean()) < 1e-12
