import numpy as np
import pytest

from adbq.grid import (ENUM_MAX_BITS, TM_MAX_ROWS, GridModel, TransferMatrix, brief_trigger_law,
                       calibrated_grid, calibration_residual, enumerate_law, exact_loss_pmf,
                       grid_edges, zone_types)


@pytest.mark.parametrize("name", ["adb-base", "adb-common-shock"])
def test_two_zone_grid_is_the_brief_trigger_law(name):
    m = calibrated_grid(1, 2, name)
    pi = enumerate_law(m)["pi"].reshape(2, 2)          # state id = 2 b0 + b1
    assert np.abs(pi - brief_trigger_law(name)["joint"]).max() < 1e-9


def test_independent_scenario_is_a_product_law():
    m = calibrated_grid(3, 3, "adb-independent")
    assert np.all(m.J == 0)
    d = enumerate_law(m)
    assert np.abs(d["marginals"] - m.p_target).max() < 1e-14
    for (i, j), pij in zip(m.edge_list, d["pair"]):
        assert pij == pytest.approx(m.p_target[i] * m.p_target[j], rel=1e-12)


def test_energy_and_state_id_follow_logistics_convention():
    """E(b) = theta . b - sum J b_i b_j and zone k is bit n-1-k of the state ID."""
    m = calibrated_grid(2, 3, kappa=2.0)
    e = enumerate_law(m)["energies"]
    for k in range(m.n):
        assert e[1 << (m.n - 1 - k)] == pytest.approx(m.theta[k])
    i, j, J = m.edges[0]
    s = (1 << (m.n - 1 - i)) | (1 << (m.n - 1 - j))
    assert e[s] == pytest.approx(m.theta[i] + m.theta[j] - J)


@pytest.mark.parametrize("shape,kappa", [((3, 4), 1.0), ((4, 4), 1.0), ((4, 4), 3.0), ((4, 5), 4.0)])
def test_transfer_matrix_matches_enumeration(shape, kappa):
    m = calibrated_grid(*shape, kappa=kappa)
    d = enumerate_law(m)
    tm = TransferMatrix(m)
    mg, pr = tm.moments()
    assert tm.logZ == pytest.approx(d["logZ"], abs=1e-10)
    assert np.abs(mg - d["marginals"]).max() < 1e-13
    assert np.abs(pr - d["pair"]).max() < 1e-13
    v, p = tm.loss_pmf()
    pe = np.bincount(d["loss_units"], weights=d["pi"], minlength=len(p))
    assert np.abs(pe - p).max() < 1e-13


@pytest.mark.parametrize("shape", [(4, 4), (6, 6), (8, 8)])
@pytest.mark.parametrize("name", ["adb-base", "adb-common-shock"])
def test_calibration_matches_brief_moments(shape, name):
    m = calibrated_grid(*shape, name)
    assert calibration_residual(m) < 1e-7
    assert len(m.J) == len(grid_edges(*shape))


@pytest.mark.parametrize("kappa", [2.0, 4.0])
def test_contagion_knob_keeps_marginals(kappa):
    base = calibrated_grid(4, 4)
    m = calibrated_grid(4, 4, kappa=kappa)
    assert np.allclose(m.J, kappa * base.J)
    assert calibration_residual(m) < 1e-7
    # stronger contagion fattens the tail at fixed marginals
    v, p0 = exact_loss_pmf(base)
    _, p1 = exact_loss_pmf(m)
    assert np.dot(v, p0) == pytest.approx(np.dot(v, p1), rel=1e-6)
    assert p1[v > 0.5 * v.max()].sum() > 2 * p0[v > 0.5 * v.max()].sum()


def test_payouts_and_types():
    law = brief_trigger_law()
    m = calibrated_grid(2, 2)
    assert list(zone_types(2, 2)) == [0, 1, 1, 0]
    assert m.payout_units.tolist() == [17, 15, 15, 17]          # 42.5m and 37.5m
    assert abs(17 * 2.5e6 - law["payout"][0]) <= 1.25e6


def test_engine_limits():
    with pytest.raises(ValueError):
        TransferMatrix(calibrated_grid(TM_MAX_ROWS + 1, 1, "adb-independent"))
    assert ENUM_MAX_BITS >= 20
