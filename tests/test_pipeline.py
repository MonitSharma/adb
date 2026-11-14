import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")

from adbq.cli import main                                                     # noqa: E402
from adbq.model import SCENARIOS, TABLE5, TARGETS, targets, term_distribution_sources  # noqa: E402
from adbq.pipeline import load_config, run, scenario, validate                  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_shipped_brief_configs_reproduce_table5(name):
    s = scenario(load_config(CONFIGS / f"{name}.yaml"))
    got = targets(term_distribution_sources(s), s)
    ref = targets(term_distribution_sources(SCENARIOS[name]), SCENARIOS[name])
    for k in TARGETS:
        assert got[k] == pytest.approx(ref[k], rel=1e-14)
    assert [round(got[k], 6) for k in TARGETS[1:3]] == pytest.approx(TABLE5[name][1:3], abs=1e-5)


def test_custom_country_names_give_the_same_law():
    cfg = load_config(CONFIGS / "adb-base.yaml")
    cfg["countries"] = [{"name": "A", "exceedance": c["exceedance"]} for c in cfg["countries"]]
    cfg["countries"][1]["name"] = "B"
    s = scenario(validate(cfg))
    base = SCENARIOS["adb-base"]
    assert np.abs(term_distribution_sources(s) - term_distribution_sources(base)).max() < 1e-15


@pytest.mark.parametrize("bad", [
    {"model": "nonsense"},
    {"model": "two_country", "perils": {"quake": {"steps": 5}}},
    {"model": "two_country", "countries": [{"name": "A", "exceedance": [0.1, 0.05, 0.02, 0.01]},
                                           {"name": "B", "exceedance": [0.1, 0.05, 0.02, 0.01]}]},
])
def test_validation_rejects(bad):
    with pytest.raises(ValueError):
        validate(bad)


def test_two_country_run_end_to_end(tmp_path):
    cfg = load_config(CONFIGS / "adb-base.yaml")
    cfg.update(name="mini", targets=["E[L]", "P(L>T)", "CVaR95"])
    out = run(cfg, tmp_path)
    rep = json.loads((out / "report.json").read_text())
    assert rep["exact"]["E[L]"]["value"] == pytest.approx(10.06e6, rel=1e-3)
    assert rep["exact"]["CVaR95"]["source"] == "exact"
    assert rep["quantum"]["P(L>T)"]["iqae"] > 0 and rep["classical"]["P(L>T)"]["is"] > 0
    for f in ("summary.md", "loss_distribution.png", "queries.png"):
        assert (out / f).exists()


def test_chain_run_matches_exact_chain(tmp_path):
    from adbq.portfolio import chain_portfolio, exact_chain
    cfg = validate({"name": "chain3", "model": "chain", "dependence": {"type": "chain", "n": 3},
                    "targets": ["P(L>T)", "VaR99"], "seed": 3})
    rep = json.loads((run(cfg, tmp_path) / "report.json").read_text())
    assert rep["exact"]["P(L>T)"]["value"] == pytest.approx(exact_chain(chain_portfolio(3))["P(L>T)"], rel=1e-12)


def test_cli_validate(capsys):
    assert main(["validate", str(CONFIGS / "chain-5.yaml")]) == 0
    assert "ok" in capsys.readouterr().out
