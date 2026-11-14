import numpy as np
import pytest

from adbq.model import (SCENARIOS, TABLE5, TARGETS, exact, term_distribution_brief,
                        term_distribution_sources, two_country_params, annual_distribution)


def test_annual_base_case_numbers():
    s = SCENARIOS["adb-base"]
    p = two_country_params(s)
    assert p.c == pytest.approx(0.0029667, abs=5e-8)
    ann = annual_distribution(s)
    assert ann[0, 0].sum() == pytest.approx(0.9262, abs=5e-5)   # no quake payout
    assert ann[0, 0, 0, 0] == pytest.approx(0.8940, abs=5e-5)   # no payout at all


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_source_route_equals_brief_route(name):
    s = SCENARIOS[name]
    assert np.abs(term_distribution_brief(s) - term_distribution_sources(s)).max() < 1e-14


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_table5(name):
    ex = exact(name)
    ref = TABLE5[name]
    for k, r in zip(TARGETS, ref):
        # Table 5 is printed to 3-4 significant figures
        assert ex[k] == pytest.approx(r, rel=0.02, abs=6e-7)


def test_var_cvar_base_and_dependence_sensitivity():
    from adbq.model import loss_distribution, var_cvar
    out = {n: var_cvar(*loss_distribution(SCENARIOS[n]), 0.99) for n in
           ("adb-independent", "adb-base", "adb-common-shock")}
    assert out["adb-base"][0] == pytest.approx(87.5e6)
    assert out["adb-base"][1] == pytest.approx(106.7e6, rel=1e-3)
    # CVaR rises with dependence
    assert out["adb-independent"][1] < out["adb-base"][1] < out["adb-common-shock"][1]
    # VaR95 is identical across the three dependence settings
    v95 = {var_cvar(*loss_distribution(SCENARIOS[n]), 0.95)[0] for n in out}
    assert v95 == {70e6}
