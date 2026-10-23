import numpy as np

from genrl.analysis import selection_bias as sb


def test_hoeffding_values_from_proposal():
    assert abs(sb.hoeffding_bound(60, 30) - 0.261) < 1e-3
    assert abs(sb.hoeffding_bound(60, 100) - 0.143) < 1e-3
    assert abs(sb.hoeffding_bound(60, 300) - 0.083) < 1e-3


def test_expected_max_std_normal():
    assert sb.expected_max_std_normal(1) == 0.0
    assert abs(sb.expected_max_std_normal(2) - 1 / np.sqrt(np.pi)) < 1e-6


def test_plugin_matches_independent_formula_when_independent():
    K, n, sigma = 20, 50, 0.4
    pred = sb.gaussian_plugin(np.zeros(K), sigma**2 * np.eye(K), n, n_sims=100_000)
    assert abs(pred - sb.independent_k_formula(sigma, K, n)) < 2e-3


def test_plugin_is_zero_for_perfectly_correlated():
    K = 10
    Sigma = 0.25 * np.ones((K, K))
    A = np.linspace(0.3, 0.5, K)  # clear winner, all noise shared
    assert abs(sb.gaussian_plugin(A, Sigma, 100, n_sims=50_000)) < 3e-3


def test_meff_bounds():
    K = 8
    assert abs(sb.meff_nyholt(np.eye(K)) - K) < 1e-9
    assert abs(sb.meff_li_ji(np.eye(K)) - K) < 1e-9
    assert sb.meff_nyholt(np.ones((K, K))) < 1.5


def test_pool_split_identity():
    rng = np.random.default_rng(0)
    c_bar = rng.random((15, 400))
    N, n = c_bar.shape[1], 30
    res = sb.split_simulation(c_bar, n=n, n_reps=200)
    lhs = res.sel_score - res.oracle_score
    rhs = N / (N - n) * (res.sel_score - res.pool_score)
    assert np.allclose(lhs, rhs, atol=1e-12)


def test_split_simulation_optimism_positive_and_below_bound():
    rng = np.random.default_rng(1)
    K, N = 30, 1500
    p = rng.uniform(0.4, 0.6, size=(1, N)) + rng.normal(0, 0.05, size=(K, N))
    c_bar = (rng.random((K, N, 4)) < np.clip(p, 0, 1)[..., None]).mean(axis=2)
    res = sb.split_simulation(c_bar, n=100, n_reps=500)
    assert 0 < res.optimism_vs_pool < sb.hoeffding_bound(K, 100)


def test_selection_rules_shapes():
    rng = np.random.default_rng(2)
    seeds = np.repeat(np.arange(3), 6)
    steps = np.tile(np.arange(25, 175, 25), 3)
    c_bar = rng.random((18, 300))
    out = sb.selection_rules(c_bar, seeds, steps, n=50, n_reps=50)
    assert set(out) == {"per_run_argmax", "last", "mean_last3", "argmax_mean_curve"}
    assert out["per_run_argmax"]["selection"] >= out["per_run_argmax"]["oracle"]


def test_split_simulation_rejects_empty_oracle():
    import pytest

    with pytest.raises(ValueError):
        sb.split_simulation(np.zeros((3, 50)), n=50)
