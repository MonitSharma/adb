import numpy as np
import pytest

from adbq.baselines import PathSpace
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
