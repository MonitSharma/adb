import numpy as np
import pytest

from adbq.circuits import apply, grover_iterate, prob_good
from adbq.model import SCENARIOS, TARGETS, exact
from adbq.oracle import build_oracle
from adbq.qae import iqae


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("key", TARGETS)
def test_circuit_amplitude_is_exact(name, key):
    o = build_oracle(SCENARIOS[name], key)
    a = prob_good(apply({0: 1.0}, o.A), o.good)
    assert o.f_max * a == pytest.approx(exact(name)[key], rel=1e-12)


@pytest.mark.parametrize("cond", [False, True])
def test_grover_dynamics(cond):
    o = build_oracle(SCENARIOS["adb-base"], TARGETS[4], condition_rain=cond)
    st = apply({0: 1.0}, o.A)
    a = prob_good(st, o.good)
    assert o.prefactor * o.f_max * a == pytest.approx(exact("adb-base")[TARGETS[4]], rel=1e-12)
    th = np.arcsin(np.sqrt(a))
    Ainv = o.A.inverse()
    for k in range(1, 4):
        st = grover_iterate(st, o.A, Ainv, o.good)
        assert prob_good(st, o.good) == pytest.approx(np.sin((2 * k + 1) * th) ** 2, rel=1e-10)


def test_iqae_interval_covers():
    rng = np.random.default_rng(3)
    a = 3.163e-5
    hits = sum(lo <= a <= hi for _, _, (lo, hi) in (iqae(a, 0.05 * a, rng) for _ in range(100)))
    assert hits >= 93
