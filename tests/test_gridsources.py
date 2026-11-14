
import numpy as np
import pytest
from scipy.linalg import expm

from adbq.grid_estimators import (TARGETS, chunked_is, exact_targets, iqae_queries, target_functions,
                                  tilted_laws)
from adbq.gridsources import (EXACT_MAX_ROWS, draw, exact_loss_pmf, exact_mean_var, grid_portfolio,
                              log_weight, losses, source_laws)
from adbq.model import SCENARIOS, STEPS
from adbq.portfolio import chain_portfolio
from adbq.resources import estimate
from adbq.risk import loss_pmf_chain, loss_pmf_two_country


@pytest.mark.parametrize("name", ["adb-base", "adb-common-shock", "adb-rain-cover"])
def test_one_by_two_grid_is_the_brief(name):
    v, p = exact_loss_pmf(grid_portfolio(1, 2, SCENARIOS[name]))
    v2, p2 = loss_pmf_two_country(SCENARIOS[name])
    assert np.array_equal(v, v2) and np.abs(p - p2).max() < 1e-15


@pytest.mark.parametrize("n", [3, 5])
def test_one_by_n_grid_is_the_chain(n):
    """The 1 x n grid and portfolio.py's chain are the same model: same edge rates, same law."""
    cp = chain_portfolio(n)
    gp = grid_portfolio(1, n)
    assert np.allclose(gp.c, cp.c, rtol=1e-15)
    v, p = exact_loss_pmf(gp)
    vc, pc = loss_pmf_chain(cp)
    assert np.abs(p - pc).max() < 1e-14


def _brute_force_2x2(gp):
    """Joint CTMC over 5^4 quake states (own + four shared streams) and enumerated rain."""
    n, H = gp.n, gp.scen.years
    shape = (STEPS + 1,) * n
    N = 5**n
    G = np.zeros((N, N))
    own = gp.own_quake_rate
    for flat in range(N):
        st = np.array(np.unravel_index(flat, shape))
        for i in range(n):
            for k in range(1, STEPS + 1):
                nx = st.copy(); nx[i] = min(STEPS, nx[i] + k)
                G[flat, np.ravel_multi_index(nx, shape)] += own[i] * gp.w[i, k - 1]
        for e, (i, j) in enumerate(gp.edges):
            for k in range(1, STEPS + 1):
                for l in range(1, STEPS + 1):
                    nx = st.copy(); nx[i] = min(STEPS, nx[i] + k); nx[j] = min(STEPS, nx[j] + l)
                    G[flat, np.ravel_multi_index(nx, shape)] += gp.c[e] * gp.w[i, k - 1] * gp.w[j, l - 1]
        G[flat, flat] -= G[flat].sum()
    pq = expm(G * H)[0]
    po = 1 - np.exp(-gp.own_rain_rate * H)
    ps = 1 - np.exp(-gp.cr * H)
    qu, ru, u = gp.units
    pmf = np.zeros(n * gp.full_units + 1)
    qs = np.array(np.unravel_index(np.arange(N), shape)).T
    for bits in np.ndindex(*(2,) * (n + len(gp.edges))):
        ob, eb = bits[:n], bits[n:]
        pr = np.prod([po[i] if ob[i] else 1 - po[i] for i in range(n)])
        pr *= np.prod([ps[e] if eb[e] else 1 - ps[e] for e in range(len(gp.edges))])
        rain = np.array(ob)
        for e, (i, j) in enumerate(gp.edges):
            rain[i] |= eb[e]; rain[j] |= eb[e]
        L = (qs * qu).sum(1) + (rain * ru).sum()
        pmf += pr * np.bincount(L, weights=pq, minlength=len(pmf))
    return pmf


def test_two_by_two_matches_joint_ctmc():
    gp = grid_portfolio(2, 2)
    _, p = exact_loss_pmf(gp)
    assert np.abs(p - _brute_force_2x2(gp)).max() < 1e-13


def test_exact_mean_and_variance():
    gp = grid_portfolio(2, 3)
    v, p = exact_loss_pmf(gp)
    m, var = exact_mean_var(gp)
    assert m == pytest.approx(v @ p, rel=1e-12)
    assert var == pytest.approx(v**2 @ p - (v @ p) ** 2, rel=1e-10)


def test_sampler_reproduces_exact_law():
    gp = grid_portfolio(2, 2)
    v, p = exact_loss_pmf(gp)
    rng = np.random.default_rng(0)
    L = losses(gp, draw(gp, rng.random((200_000, gp.n_sources))))
    emp = np.bincount(L, minlength=len(p)) / len(L)
    assert 0.5 * np.abs(emp - p).sum() < 0.01


def test_tilted_is_is_unbiased_and_weights_normalise():
    gp = grid_portfolio(2, 3)
    v, p = exact_loss_pmf(gp)
    ex = exact_targets(gp, v, p)
    tf = target_functions(gp)
    rng = np.random.default_rng(1)
    laws = tilted_laws(gp, 0.06)
    X = draw(gp, rng.random((100_000, gp.n_sources)), laws)
    assert np.exp(log_weight(gp, X, laws)).mean() == pytest.approx(1.0, abs=0.02)
    for k in ("P(L>T3)", "stop-loss"):
        m, var = chunked_is(gp, tf[k][0], laws, 200_000, rng)
        assert m == pytest.approx(ex[k], abs=4 * np.sqrt(var / 200_000))


def test_iqae_emulated_queries_follow_one_over_sqrt_a():
    rng = np.random.default_rng(2)
    q1 = iqae_queries(1e-2, 0.05, rng, reps=60)["queries"]
    q2 = iqae_queries(1e-4, 0.05, rng, reps=60)["queries"]
    assert 5 < q2 / q1 < 20                                 # ideal ratio 10


def test_resource_model_default_is_the_chain():
    a, b = estimate(10), estimate(10, edges=9, degree=2)
    assert a == b
    assert estimate(16, edges=24, degree=4).toffoli > estimate(16).toffoli


def test_exact_limit():
    with pytest.raises(ValueError):
        exact_loss_pmf(grid_portfolio(EXACT_MAX_ROWS + 1, 1))
