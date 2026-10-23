import numpy as np

from genrl.analysis.stats import bootstrap_problems, mean_ci, variance_chi2_ci


def test_chi2_interval_ratios_from_proposal():
    for S, lo, hi in [(3, 0.27, 39.5), (5, 0.36, 8.3), (8, 0.44, 4.1), (10, 0.47, 3.3)]:
        a, b = variance_chi2_ci(1.0, S - 1)
        assert abs(a - lo) < 0.01 and abs(b - hi) < 0.1


def test_mean_ci_and_bootstrap_agree():
    rng = np.random.default_rng(0)
    x = rng.random(500)
    m, lo, hi = mean_ci(x)
    bm, blo, bhi = bootstrap_problems(np.mean, x, n_boot=2000)
    assert m == bm
    assert abs(lo - blo) < 0.01 and abs(hi - bhi) < 0.01
