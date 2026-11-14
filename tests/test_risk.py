import numpy as np
import pytest

from adbq.model import SCENARIOS, exact, term_distribution_brief
from adbq.portfolio import brute_force_joint, chain_portfolio, exact_chain
from adbq.risk import (ALPHAS, loss_pmf_chain, loss_pmf_two_country, loss_units_two_country,
                       risk_summary, ru_cvar_bruteforce, var_cvar)


def test_var_cvar_conventions_small_example():
    # F(0)=0.5, F(1)=0.9, F(3)=1: VaR_0.9 = 1 (atom reached exactly), CVaR_0.9 = 1 + 0.1*2/0.1 = 3
    v, c = var_cvar([0, 1, 3], [0.5, 0.4, 0.1], 0.9)
    assert (v, c) == (1.0, pytest.approx(3.0))
    v, c = var_cvar([3, 0, 1], [0.1, 0.5, 0.4], 0.95)      # unsorted input
    assert v == 3.0 and c == pytest.approx(3.0)


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("alpha", ALPHAS)
def test_ru_minimum_is_attained_at_var(name, alpha):
    v, p = loss_pmf_two_country(SCENARIOS[name])
    assert var_cvar(v, p, alpha)[1] == pytest.approx(ru_cvar_bruteforce(v, p, alpha), rel=1e-12)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_two_country_pmf_consistent_with_table5_targets(name):
    s = SCENARIOS[name]
    v, p = loss_pmf_two_country(s)
    ex = exact(name)
    assert p.sum() == pytest.approx(1, abs=1e-14)
    assert np.dot(v, p) == pytest.approx(ex["term_expected_principal_loss_usd"], rel=1e-12)
    assert p[v > s.threshold].sum() == pytest.approx(ex["term_portfolio_loss_exceedance_probability"], rel=1e-12)
    assert np.dot(np.maximum(v - s.threshold, 0), p) == pytest.approx(ex["term_portfolio_stop_loss_usd"], rel=1e-12)
    # and the brief's annual-CTMC route gives the same loss law
    Lu, u = loss_units_two_country(s)
    p_brief = np.bincount(Lu.reshape(-1), weights=term_distribution_brief(s).reshape(-1), minlength=len(p))
    assert np.abs(p_brief - p).max() < 1e-15


def test_expected_base_values():
    """The headline numbers quoted in the project brief / README."""
    r = {n: risk_summary(*loss_pmf_two_country(SCENARIOS[n])) for n in
         ("adb-base", "adb-independent", "adb-common-shock")}
    assert r["adb-base"]["VaR99"] == 87.5e6
    assert r["adb-base"]["CVaR99"] == pytest.approx(106.70e6, abs=0.01e6)
    for n in r:
        assert r[n]["VaR95"] == 70e6                      # VaR95 is blind to dependence ...
    cv = [r[n]["CVaR99"] for n in ("adb-independent", "adb-base", "adb-common-shock")]
    assert cv == pytest.approx([97.13e6, 106.70e6, 113.96e6], abs=0.01e6)   # ... CVaR is not


@pytest.mark.parametrize("name", ["adb-base", "adb-common-shock", "adb-rain-cover"])
def test_chain_n2_loss_law_is_the_brief_model(name):
    v2, p2 = loss_pmf_two_country(SCENARIOS[name])
    vc, pc = loss_pmf_chain(chain_portfolio(2, SCENARIOS[name]))
    assert np.array_equal(v2, vc)
    assert np.abs(p2 - pc).max() < 1e-15


@pytest.mark.parametrize("n", [3, 4])
def test_chain_loss_law_matches_full_ctmc(n):
    pf = chain_portfolio(n)
    prob, Li = brute_force_joint(pf)
    vals, pmf = loss_pmf_chain(pf)
    u = vals[1]
    idx = np.rint(Li.sum(-1) / u).astype(int)
    ref = np.bincount(idx.reshape(-1), weights=prob.reshape(-1), minlength=len(pmf))
    assert np.abs(ref - pmf).max() < 1e-13


@pytest.mark.parametrize("n", [5, 10])
def test_chain_loss_law_consistent_with_exact_chain(n):
    pf = chain_portfolio(n)
    vals, pmf = loss_pmf_chain(pf)
    ex = exact_chain(pf)
    T = pf.scen.threshold
    assert pmf.sum() == pytest.approx(1, abs=1e-13)
    assert np.dot(vals, pmf) == pytest.approx(ex["E[L]"], rel=1e-12)
    assert pmf[vals > T].sum() == pytest.approx(ex["P(L>T)"], rel=1e-10)
    assert np.dot(np.maximum(vals - T, 0), pmf) == pytest.approx(ex["stop-loss"], rel=1e-10)
