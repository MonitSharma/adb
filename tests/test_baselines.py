import numpy as np
import pytest

from adbq.baselines import CONDITIONED_KEY, PathSpace
from adbq.model import SCENARIOS, TARGETS, exact


@pytest.fixture(scope="module")
def ps():
    return PathSpace(SCENARIOS["adb-base"])


def test_path_space_mean_is_exact(ps):
    ex = exact("adb-base")
    for k in TARGETS:
        assert ps.mu(k) == pytest.approx(ex[k], rel=1e-12)


@pytest.mark.parametrize("method", ["mc", "is", "strat"])
def test_sampler_variance_matches_formula(ps, method):
    rng = np.random.default_rng(7)
    k, N, R = TARGETS[2], 2048, 400
    est = np.array([getattr(ps, f"estimate_{method}")(k, N, rng) for _ in range(R)])
    var = {"mc": ps.var_mc(k) / N, "is": ps.var_is(k) / N, "strat": ps.var_strat(k, N)}[method]
    assert est.mean() == pytest.approx(ps.mu(k), abs=4 * np.sqrt(var / R))
    assert est.var() == pytest.approx(var, rel=0.25)


@pytest.mark.parametrize("k", [TARGETS[2], TARGETS[3]])
def test_tilt_sampler_variance_matches_formula(ps, k):
    rng = np.random.default_rng(11)
    N, R = 2048, 400
    est = np.array([ps.estimate_tilt(k, N, rng) for _ in range(R)])
    var = ps.var_tilt(k) / N
    assert est.mean() == pytest.approx(ps.mu(k), abs=4 * np.sqrt(var / R))
    assert est.var() == pytest.approx(var, rel=0.25)


def test_tuned_tilt_never_worse_than_mc(ps):
    # theta = 0 is plain Monte Carlo and is on the tuning grid
    for k in TARGETS:
        assert ps.var_tilt(k) <= ps.var_mc(k) * (1 + 1e-12)


def test_conditioned_factorisation_is_exact(ps):
    k = CONDITIONED_KEY
    pq = ps.p_quake
    P = np.einsum("i,j,k->ijk", pq, pq, pq)
    assert ps.p_rain_both * (P * ps.f_quake_exhausted).sum() == pytest.approx(ps.mu(k), rel=1e-12)


def test_conditioned_rain_factor_matches_oracle_prefactor(ps):
    from adbq.oracle import build_oracle
    orc = build_oracle(SCENARIOS["adb-base"], CONDITIONED_KEY, condition_rain=True)
    assert ps.p_rain_both == pytest.approx(orc.prefactor, rel=1e-12)


def test_cond_tilt_sampler_variance_matches_formula(ps):
    rng = np.random.default_rng(13)
    N, R = 4096, 400
    est = np.array([ps.estimate_cond_tilt(N, rng) for _ in range(R)])
    var = ps.var_cond_tilt() / N
    mu = ps.mu(CONDITIONED_KEY)
    assert est.mean() == pytest.approx(mu, abs=4 * np.sqrt(var / R))
    assert est.var() == pytest.approx(var, rel=0.25)
    # far below the brief's importance sampler on this rare target
    assert ps.var_cond_tilt() < 0.05 * ps.var_is(CONDITIONED_KEY)
