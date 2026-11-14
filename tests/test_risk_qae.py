import numpy as np
import pytest

from adbq.circuits import apply, grover_iterate, prob_good
from adbq.model import SCENARIOS
from adbq.oracle import build_oracle
from adbq.risk import ALPHAS, loss_pmf_two_country, loss_units_two_country, var_cvar
from adbq.risk_qae import (cdf_amplitude, excess_amplitude, quantum_cvar, quantum_var)


@pytest.mark.parametrize("name", ["adb-base", "adb-rain-cover"])
def test_gate_level_cdf_and_excess_amplitudes(name):
    """The emulator's amplitudes for 1{L <= x} and (L - v)+ are those of the circuit A."""
    s = SCENARIOS[name]
    vals, pmf = loss_pmf_two_country(s)
    Lu, u = loss_units_two_country(s)
    L = Lu * u
    for alpha in ALPHAS:
        v, _ = var_cvar(vals, pmf, alpha)
        for x in (v - u, v):                                     # the two atoms bisection must separate
            o = build_oracle(s, "cdf", payoff=(L <= x).astype(float))
            a = prob_good(apply({0: 1.0}, o.A), o.good)
            assert a == pytest.approx(cdf_amplitude(vals, pmf, x), abs=1e-13)
        o = build_oracle(s, "excess", payoff=np.maximum(L - v, 0.0))
        a = prob_good(apply({0: 1.0}, o.A), o.good)
        a_ref, f_max = excess_amplitude(vals, pmf, v)
        assert o.f_max == pytest.approx(f_max)
        assert a == pytest.approx(a_ref, rel=1e-11)


def test_grover_dynamics_on_cdf_indicator():
    s = SCENARIOS["adb-base"]
    Lu, u = loss_units_two_country(s)
    o = build_oracle(s, "cdf", payoff=(Lu * u <= 87.5e6).astype(float))
    st = apply({0: 1.0}, o.A)
    th = np.arcsin(np.sqrt(prob_good(st, o.good)))
    Ainv = o.A.inverse()
    for k in (1, 2):
        st = grover_iterate(st, o.A, Ainv, o.good)
        assert prob_good(st, o.good) == pytest.approx(np.sin((2 * k + 1) * th) ** 2, abs=1e-10)


@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_bisection_finds_var_with_stated_confidence(alpha):
    vals, pmf = loss_pmf_two_country(SCENARIOS["adb-base"])
    v, _ = var_cvar(vals, pmf, alpha)
    rng = np.random.default_rng(4)
    hits = [quantum_var(vals, pmf, alpha, rng).var == v for _ in range(100)]
    assert np.mean(hits) >= 0.97


def test_cvar_error_shrinks_with_epsilon():
    vals, pmf = loss_pmf_two_country(SCENARIOS["adb-base"])
    _, c = var_cvar(vals, pmf, 0.99)
    rng = np.random.default_rng(9)
    rmse, queries = [], []
    for eps in (0.3, 0.05):
        rs = [quantum_cvar(vals, pmf, 0.99, eps, rng) for _ in range(60)]
        rmse.append(np.sqrt(np.mean([(r.cvar - c) ** 2 for r in rs])))
        queries.append(np.mean([r.queries for r in rs]))
    assert rmse[1] < rmse[0] / 2 and queries[1] > queries[0]
    assert rmse[1] / c < 0.005


def test_matched_mode_is_cheaper_at_995():
    vals, pmf = loss_pmf_two_country(SCENARIOS["adb-base"])
    rng = np.random.default_rng(1)
    q = {m: np.mean([quantum_cvar(vals, pmf, 0.995, 0.3, rng, var_mode=m).queries
                     for _ in range(20)]) for m in ("certify", "matched")}
    assert q["matched"] < q["certify"] / 2
