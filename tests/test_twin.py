"""Digital twin: every building block against exact values."""
from dataclasses import replace

import numpy as np
import pytest

from adbq.model import SCENARIOS, TARGETS, exact
from adbq.twin import (Ledger, ScenarioLibrary, SourceLaw, allocate, exact_ratio, paired_sign, plan_calibration,
                       ratio_bounds, sweep, weighted_error)

BASE = SCENARIOS["adb-base"]


def test_source_law_reproduces_table5():
    law = SourceLaw(BASE)
    ex = exact("adb-base")
    for k in TARGETS:
        assert law.mean(law.payoff(k)) == pytest.approx(ex[k], rel=1e-12)


@pytest.mark.parametrize("m", [0.5, 1.05, 2.0])
def test_source_ratio_bounds_contain_the_exact_ratio(m):
    a, b = SourceLaw(BASE), SourceLaw(replace(BASE, quake_mult=m))
    lo, hi = ratio_bounds(a, b)
    r = exact_ratio(a, b)[b.joint > 0]
    assert lo <= r.min() * (1 + 1e-12) and r.max() <= hi * (1 + 1e-12)


def test_new_support_forbids_transport():
    a, b = SourceLaw(replace(BASE, quake_mult=0.0)), SourceLaw(BASE)
    assert ratio_bounds(a, b)[1] == np.inf


def test_transported_intervals_cover_and_save_queries():
    rng = np.random.default_rng(3)
    grid = np.round(np.arange(0.8, 1.2001, 0.05), 2)
    runs = [sweep(BASE, "quake_mult", grid, TARGETS[1], 0.1, rng) for _ in range(10)]
    cov = np.mean([r["covered"] for run in runs for r in run["rows"]])
    assert cov >= 0.98
    assert np.mean([x["twin_queries"] for x in runs]) < np.mean([x["baseline_queries"] for x in runs])


def test_plan_never_covers_less_than_one_version():
    laws = [SourceLaw(replace(BASE, rain_shared_share=s)) for s in (0.05, 0.5)]
    h, n = plan_calibration(laws, 0, 0.1)
    assert n >= 1 and 0 < h <= 0.1


def test_thinned_library_is_an_exact_sample_of_the_new_conditional_law():
    rng = np.random.default_rng(5)
    a, b = SourceLaw(BASE), SourceLaw(replace(BASE, quake_mult=1.5))
    ev = a.payoff(TARGETS[4]) > 0
    lib = ScenarioLibrary.generate(a, ev, 20000, rng, Ledger())
    lib2, kept = lib.move_to(b, 0, rng, Ledger())             # thinning only, no top-up
    assert 0 < kept < 20000
    target = np.where(ev, b.joint, 0).ravel()
    target /= target.sum()
    top = np.argsort(target)[::-1][:5]                          # most likely conditional configurations
    for i in top:
        p = target[i]
        assert np.mean(lib2.samples == i) == pytest.approx(p, abs=4 * np.sqrt(p * (1 - p) / kept))


def test_paired_sign_is_correct():
    rng = np.random.default_rng(11)
    law = SourceLaw(BASE)
    g1 = law.payoff(TARGETS[3], replace(BASE, quake_limit=60e6, rain_limit=20e6))
    g2 = law.payoff(TARGETS[3])
    signs = [paired_sign(law, g1, g2, rng)[0] for _ in range(20)]
    assert all(s == -1 for s in signs)                          # 60/20 has the smaller stop-loss


def test_allocation_beats_equal_split():
    kap, w = np.array([1.0, 10.0, 100.0]), np.array([1.0, 1.0, 4.0])
    opt = allocate(kap, w, 300.0)
    assert opt.sum() == pytest.approx(300.0)
    assert weighted_error(kap, w, opt) < weighted_error(kap, w, np.full(3, 100.0))
    # first-order optimality: w kappa^2 / n^3 equal across targets
    g = w * kap ** 2 / opt ** 3
    assert np.allclose(g, g[0])


def test_twin_cli_runs(tmp_path):
    from pathlib import Path
    from adbq.twin import load_twin_config, run_twin
    cfg = load_twin_config(Path(__file__).resolve().parents[1] / "configs" / "twin-adb-base.yaml")
    cfg["sweep"].update({"from": 0.9, "to": 1.1})
    out = run_twin(cfg, tmp_path)
    assert (out / "summary.md").exists() and (out / "report.json").exists()


def test_twin_defaults_to_chernoff_hoeffding_and_validates_method(tmp_path):
    import yaml
    from pathlib import Path
    from adbq import twin
    assert twin.METHOD == "ch"
    src = Path(__file__).resolve().parents[1] / "configs" / "twin-adb-base.yaml"
    cfg = twin.load_twin_config(src)
    assert cfg["interval"] == "ch"
    bad = yaml.safe_load(src.read_text())
    bad["interval"] = "wald"
    bad["portfolio"] = str(src.parent / bad["portfolio"])
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError):
        twin.load_twin_config(p)


@pytest.mark.parametrize("method", ["ch", "cp"])
def test_both_interval_methods_certify(method):
    rng = np.random.default_rng(17)
    law = SourceLaw(BASE)
    f = law.payoff(TARGETS[1])
    mu = law.mean(f)
    from adbq.twin import certified
    hits = [lo <= mu <= hi for _, lo, hi in (certified(mu, 1.0, 0.1, rng, method=method) for _ in range(100))]
    assert np.mean(hits) >= 0.97
