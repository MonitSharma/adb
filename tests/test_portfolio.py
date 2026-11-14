import pytest

from adbq.model import SCENARIOS, SHORT, TARGETS, exact
from adbq.portfolio import GEN_TARGETS, brute_force_chain, chain_portfolio, exact_chain


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_chain_n2_is_the_brief_model(name):
    got = exact_chain(chain_portfolio(2, SCENARIOS[name]))
    ref = exact(name)
    for g, k in zip(GEN_TARGETS, TARGETS):
        assert got[g] == pytest.approx(ref[k], rel=1e-12)


@pytest.mark.parametrize("n", [3, 4])
def test_transfer_matrix_matches_full_ctmc(n):
    pf = chain_portfolio(n)
    a, b = exact_chain(pf), brute_force_chain(pf)
    for k in GEN_TARGETS:
        assert a[k] == pytest.approx(b[k], rel=1e-10)
