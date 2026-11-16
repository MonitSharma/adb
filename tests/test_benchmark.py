"""Benchmark interface: problem IDs, exact targets on general cases, IQAE within the
evaluator's tolerances, and team_solver.py end to end."""
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from adbq.benchmark import (case_from_config, check_result, exact_targets, iqae_targets,
                            tolerance)
from adbq.model import SCENARIOS, TARGETS, exact, targets, term_distribution_brief

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "benchmark_inputs"
# problem_id of adb-base published in joint-loss-baseline v0.5.0 examples/adb-base.json
BASE_ID = "1e05d3e5593e9f38955e8ff4271a15d65e538ed6555955aff2f84e2d3f8e5fda"


def _case(tmp_path, config: dict):
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "cases").mkdir(exist_ok=True)
    shutil.copyfile(INPUTS / "data/adb_terms.json", tmp_path / "data/adb_terms.json")
    p = tmp_path / "cases/case.json"
    p.write_text(json.dumps({**config, "terms_file": "../data/adb_terms.json"}))
    return p


def test_problem_id_matches_the_benchmark():
    assert case_from_config(INPUTS / "scenarios/adb-base.json").problem_id == BASE_ID


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_exact_targets_match_the_model(name):
    e = exact_targets(case_from_config(INPUTS / f"scenarios/{name}.json"))
    for k in TARGETS:
        assert abs(e[k] - exact(name)[k]) <= 1e-12 * abs(exact(name)[k])


@pytest.mark.parametrize("years", [1, 10])
def test_other_terms_match_the_brief_route(tmp_path, years):
    cfg = json.loads((INPUTS / "scenarios/adb-common-shock.json").read_text())
    cfg["term_years"] = years
    e = exact_targets(case_from_config(_case(tmp_path, cfg)))
    s = replace(SCENARIOS["adb-common-shock"], years=years)
    ref = targets(term_distribution_brief(s), s)
    for k in TARGETS:
        assert abs(e[k] - ref[k]) <= 1e-10 * max(abs(ref[k]), 1e-12)


def test_iqae_meets_the_tolerances_on_an_asymmetric_case(tmp_path):
    cfg = json.loads((INPUTS / "scenarios/adb-base.json").read_text())
    cfg.update(term_years=10, earthquake_sublimit_usd=[50e6, 90e6], precipitation_sublimit_usd=[20e6, 5e6],
               precipitation_annual_trigger_probability=[0.01, 0.035], portfolio_threshold_usd=95e6,
               earthquake_common_rate_multiplier=2.3)
    case = case_from_config(_case(tmp_path, cfg))
    ref = exact_targets(case)
    runs = [iqae_targets(case, seed=s) for s in range(10)]
    for k in TARGETS:
        rmse = np.sqrt(np.mean([(r["estimates"][k] - ref[k]) ** 2 for r in runs]))
        assert rmse <= tolerance(case, k, ref[k])


def test_team_solver_end_to_end(tmp_path):
    p = _case(tmp_path, json.loads((INPUTS / "scenarios/adb-rain-cover.json").read_text()))
    case = case_from_config(p)
    request = {"schema_version": 1, "run_id": "ab" * 16, "problem_id": case.problem_id, "track": "adb_drb",
               "config_file": "cases/case.json", "seed": 5, "samples": 8192, "targets": list(TARGETS)}
    (tmp_path / "request.json").write_text(json.dumps(request))
    for method, execution in (("iqae", "quantum_simulator"), ("exact", "classical_cpu")):
        subprocess.run([sys.executable, str(ROOT / "team_solver.py"), "--request", "request.json",
                        "--output", f"{method}.json", "--method", method], cwd=tmp_path, check=True)
        sub = json.loads((tmp_path / f"{method}.json").read_text())
        check_result(sub, case, request)
        assert sub["execution"] == execution


def test_world_bank_case_is_refused_cleanly(tmp_path):
    (tmp_path / "cases").mkdir()
    (tmp_path / "cases/case.json").write_text(json.dumps({"schema_version": 2, "track": "worldbank_elt"}))
    (tmp_path / "request.json").write_text(json.dumps({"config_file": "cases/case.json", "problem_id": "x",
                                                       "seed": 1, "run_id": "r"}))
    r = subprocess.run([sys.executable, str(ROOT / "team_solver.py"), "--request", "request.json",
                        "--output", "out.json"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 3 and "worldbank_elt" in r.stderr


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_shipped_result_files_are_valid(name):
    sub = json.loads((ROOT / f"results/benchmark_submission/{name}.json").read_text())
    check_result(sub, case_from_config(INPUTS / f"scenarios/{name}.json"))
