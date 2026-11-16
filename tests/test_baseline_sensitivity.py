"""The open baseline settings: Neyman bounds every allocation, mix 0.5 is the shipped IS."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from adbq.baselines import PathSpace  # noqa: E402
from adbq.model import SCENARIOS, TARGETS  # noqa: E402
import baseline_sensitivity as bs  # noqa: E402


def test_neyman_bounds_allocations_and_mix_half_is_the_brief():
    ps = PathSpace(SCENARIOS["adb-base"])
    for key in (TARGETS[1], TARGETS[4]):
        st = bs.strat_queries(ps, key, 0.01)
        assert st["neyman"] <= min(st["proportional"], st["equal"]) * (1 + 1e-12)
        mu = ps.mu(key)
        assert np.isclose(bs.is_queries(ps, key, 0.5, 0.01), ps.var_is(key) / (0.01 * mu) ** 2, rtol=1e-9)
        n = 10**5
        assert np.isclose(st["proportional"], ps.var_strat(key, n) * n / (0.01 * mu) ** 2, rtol=1e-9)
