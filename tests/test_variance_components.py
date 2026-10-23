import numpy as np

from genrl.analysis.variance_components import (
    estimate_components,
    per_problem_split,
    seed_disagreement,
)


def _simulate(S, n, R, sa, sb, sg, rng):
    # Gaussian outcomes so the true components are known exactly.
    a = rng.normal(0, sa, (S, 1, 1))
    b = rng.normal(0, sb, (1, n, 1))
    g = rng.normal(0, sg, (S, n, 1))
    e = rng.normal(0, 0.5, (S, n, R))
    return 0.5 + a + b + g + e


def test_anova_estimators_unbiased():
    rng = np.random.default_rng(0)
    truth = dict(sa=0.05, sb=0.3, sg=0.1)
    est = [estimate_components(_simulate(5, 300, 4, **truth, rng=rng)) for _ in range(300)]
    assert abs(np.mean([e.sampling for e in est]) - 0.25) < 0.005
    assert abs(np.mean([e.interaction for e in est]) - 0.01) < 0.003
    assert abs(np.mean([e.problem for e in est]) - 0.09) < 0.005
    assert abs(np.mean([e.seed for e in est]) - 0.0025) < 0.0008


def test_run_score_variance_and_n_star():
    rng = np.random.default_rng(1)
    vc = estimate_components(_simulate(5, 200, 4, 0.05, 0.3, 0.1, rng))
    full = vc.run_score_variance()
    fixed = vc.run_score_variance(fixed_benchmark=True)
    assert full >= fixed >= vc.seed
    assert vc.n_star > 0


def test_per_problem_identity_exact():
    rng = np.random.default_rng(2)
    p = rng.random((6, 100))
    parts = per_problem_split(p)
    assert np.allclose(parts["total"], parts["within_seed"] + parts["between_seed"], atol=1e-15)


def test_seed_disagreement():
    p = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.1]])
    assert seed_disagreement(p).tolist() == [0]
