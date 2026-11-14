import numpy as np
import pytest

from adbq.model import SCENARIOS
from adbq.risk import var_cvar
from adbq.risk_baselines import METHODS, RiskPathSpace


@pytest.fixture(scope="module")
def ps():
    return RiskPathSpace(SCENARIOS["adb-base"])


def test_path_losses_reproduce_exact_pmf(ps):
    pmf = np.bincount(ps.Lflat, weights=ps.P, minlength=ps.nb)
    assert np.abs(pmf - ps.pmf).max() < 1e-15


@pytest.mark.parametrize("method", ["is", "tilt"])
def test_grouped_atoms_preserve_law(ps, method):
    """Grouping paths by (loss, weight) keeps the proposal mass and the target law."""
    ps.sample_pmf(method, 1, np.random.default_rng(0), 0.99)
    L, w, mass = ps._cache[("is",) if method == "is" else ("tilt", 0.99)]
    assert mass.sum() == pytest.approx(1)
    assert np.abs(np.bincount(L, weights=mass * w, minlength=ps.nb) - ps.pmf).max() < 1e-12


@pytest.mark.parametrize("method", ["mc", "strat", "is", "tilt"])
def test_cvar_sampler_matches_asymptotic_variance(ps, method):
    alpha, N, R = 0.99, 8192, 600
    rng = np.random.default_rng(11)
    v, c = var_cvar(ps.values, ps.pmf, alpha)
    est = np.array([ps.estimate(method, N, rng, alpha) for _ in range(R)])
    sd = np.sqrt(ps.cvar_asymptotic_var(method, alpha) / N)
    rmse = np.sqrt(np.mean((est[:, 1] - c) ** 2))
    assert rmse == pytest.approx(sd, rel=0.2)
    assert np.mean(est[:, 0] == v) > 0.95


def test_sobol_is_consistent(ps):
    rng = np.random.default_rng(5)
    v, c = var_cvar(ps.values, ps.pmf, 0.95)
    est = np.array([ps.estimate("sobol", 2**14, rng, 0.95) for _ in range(20)])
    assert np.all(est[:, 0] == v)
    assert abs(est[:, 1].mean() - c) / c < 0.01


def test_tilt_beats_brief_proposals_on_cvar(ps):
    for a in (0.95, 0.99, 0.995):
        av = {m: ps.cvar_asymptotic_var(m, a) for m in ("mc", "strat", "is", "tilt")}
        assert av["tilt"] < min(av["mc"], av["strat"], av["is"])
