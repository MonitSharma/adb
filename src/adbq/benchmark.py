"""Adapter to the challenge's benchmark interface (joint-loss-baseline, ``qjlb``, v0.5.0).

The organisers' evaluator (EVALUATION.md of the benchmark) runs a team solver as

    team_solver.py --request request.json --output result.json

where the request names one case file (``config_file``) plus ``problem_id``, ``seed``,
``samples``, ``targets`` and ``run_id``. Hold-out ADB cases vary the term (1 to 10 years),
the per-country sub-limits, the rain probabilities and the portfolio threshold, so this
module solves the general ``adb_drb`` case rather than only the brief's five scenarios:

* :func:`case_from_config` reads a case and its terms file and computes the
  ``problem_id`` exactly as the benchmark does (SHA-256 of the canonical definition).
* :func:`exact_targets` gives the five term targets by the brief's route (annual CTMC,
  precipitation law, capped convolution over the term).
* :func:`iqae_targets` estimates each target with iterative QAE (Chernoff-Hoeffding
  intervals) on the amplitude a = mu / f_max of the state preparation A, with the interval
  half-width set from the evaluator's tolerance for that target.

Execution label: ``quantum_simulator``. The amplitude of A is computed classically from the
verified circuit model (``oracle.py`` matches it to 1e-13 for the brief's scenarios) and
IQAE draws every shot from the exact measurement law. ``evidence.circuit_sha256`` hashes a
canonical specification of A (source laws, register layout, payoff table); no hardware is
involved. The ``worldbank_elt`` track is not covered yet.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from .model import STEPS, TARGETS, TwoCountryParams, poisson_rates, quake_generator, rain_joint
from .qae import iqae

DEFINITION_VERSION = "adb-quarter-step-principal-v1"   # qjlb BondModel.definition_version
CAP = np.array([STEPS, STEPS, 1, 1])
SHAPE = (STEPS + 1, STEPS + 1, 2, 2)
# Evaluator tolerances for the ADB track (qjlb.evaluation.tolerances, protocol evaluation-v1):
# RMSE over repeats <= max(relative * |reference|, absolute); absolute floors in units of
# the combined principal P where marked.
TOL_RELATIVE = (0.01, 0.05, 0.05, 0.05, 0.10)
TOL_ABSOLUTE = (("P", 1e-4), (None, 1e-4), (None, 1e-4), ("P", 1e-5), (None, 1e-6))
N_QUBITS_A = 24        # register layout of oracle.py (sources, totals, objective)


@dataclass(frozen=True)
class Case:
    config: dict
    terms: dict
    problem_id: str
    years: int
    eq_limit: np.ndarray
    rain_limit: np.ndarray
    threshold: float

    @property
    def principal(self) -> np.ndarray:
        return self.eq_limit + self.rain_limit


def case_from_config(path: str | Path) -> Case:
    path = Path(path)
    raw_config = path.read_bytes()
    config = json.loads(raw_config)
    if config.get("schema_version") != 1 or config.get("track") != "adb_drb":
        raise NotImplementedError(f"track {config.get('track')!r} is not supported by this solver")
    if config.get("countries") != ["KGZ", "TJK"]:
        raise ValueError("the public ADB calibration covers KGZ and TJK in that order")
    terms_path = path.parent / config["terms_file"]
    terms_bytes = terms_path.read_bytes()
    canonical = json.dumps(
        {"definition_version": DEFINITION_VERSION, "config": config,
         "data_sha256": {"adb_terms": hashlib.sha256(terms_bytes).hexdigest()}},
        sort_keys=True, separators=(",", ":"), allow_nan=False)
    years = config["term_years"]
    if isinstance(years, bool) or not isinstance(years, int) or not 1 <= years <= 10:
        raise ValueError("term_years must be an integer from 1 to 10")
    return Case(config, json.loads(terms_bytes), hashlib.sha256(canonical.encode()).hexdigest(),
                years, np.asarray(config["earthquake_sublimit_usd"], float),
                np.asarray(config["precipitation_sublimit_usd"], float),
                float(config["portfolio_threshold_usd"]))


def annual_law(case: Case) -> np.ndarray:
    """Annual law over (a, b, r1, r2), shape (5, 5, 2, 2): brief steps 2-6, per-country inputs."""
    c, t = case.config, case.terms
    ex = t["annual_earthquake_exceedance_probabilities"]
    cond = t["same_event_conditional_trigger_probabilities"]
    L1, l1 = poisson_rates(tuple(ex["KGZ"]))
    L2, l2 = poisson_rates(tuple(ex["TJK"]))
    shared = c["earthquake_common_rate_multiplier"] * 0.5 * (
        cond["TJK_given_KGZ"] * L1 + cond["KGZ_given_TJK"] * L2)
    q = np.asarray(c["precipitation_annual_trigger_probability"], float)
    mu = -np.log1p(-q) * c["precipitation_rate_multiplier"]
    cr = c["precipitation_common_rate_share"] * mu.min()
    p = TwoCountryParams((L1, L2), (l1 / L1, l2 / L2), shared, (mu[0], mu[1]), cr)
    quake = expm(quake_generator(p))[0].reshape(STEPS + 1, STEPS + 1)
    quake = np.maximum(quake, 0.0)
    quake /= quake.sum()
    return np.einsum("ab,rs->abrs", quake, rain_joint(mu[0], mu[1], cr))


def term_law(case: Case) -> np.ndarray:
    """Capped sum of independent annual states over the term (brief step 7)."""
    ann = annual_law(case).ravel()
    states = np.array(list(np.ndindex(SHAPE)))
    dest = np.ravel_multi_index(np.minimum(states[:, None, :] + states[None, :, :], CAP)
                                .reshape(-1, 4).T, SHAPE)
    term = np.zeros(ann.size)
    term[0] = 1.0
    for _ in range(case.years):
        term = np.bincount(dest, weights=np.outer(term, ann).ravel(), minlength=ann.size)
    return term.reshape(SHAPE)


def payoff_tables(case: Case) -> dict[str, np.ndarray]:
    a, b, r1, r2 = np.indices(SHAPE)
    L1 = case.eq_limit[0] * a / STEPS + case.rain_limit[0] * r1
    L2 = case.eq_limit[1] * b / STEPS + case.rain_limit[1] * r2
    L = L1 + L2
    return {
        TARGETS[0]: L,
        TARGETS[1]: ((L1 > 0) & (L2 > 0)).astype(float),
        TARGETS[2]: (L > case.threshold).astype(float),
        TARGETS[3]: np.maximum(L - case.threshold, 0.0),
        TARGETS[4]: ((L1 >= case.principal[0]) & (L2 >= case.principal[1])).astype(float),
    }


def exact_targets(case: Case) -> dict[str, float]:
    term = term_law(case)
    return {k: float((term * f).sum()) for k, f in payoff_tables(case).items()}


def tolerance(case: Case, key: str, reference: float) -> float:
    i = TARGETS.index(key)
    unit, floor = TOL_ABSOLUTE[i]
    absolute = floor * (case.principal.sum() if unit == "P" else 1.0)
    return max(TOL_RELATIVE[i] * abs(reference), absolute)


def circuit_spec_sha256(case: Case) -> str:
    """SHA-256 of a canonical description of A for this case: term law of the loaded
    sources (rounded to 1e-15), register layout and the five payoff tables."""
    spec = {"register_qubits": N_QUBITS_A, "steps": STEPS,
            "term_law": np.round(term_law(case), 15).ravel().tolist(),
            "payoffs": {k: np.round(f, 6).ravel().tolist() for k, f in payoff_tables(case).items()}}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def iqae_targets(case: Case, seed: int, shots: int = 100, alpha: float = 0.01,
                 safety: float = 0.5) -> dict:
    """IQAE estimate of every target. The interval half-width on mu is ``safety`` times
    the evaluator tolerance, so the RMSE over repeats stays inside the tolerance whenever
    the 1 - alpha interval holds. Returns estimates, 1 - alpha intervals and query counts."""
    rng = np.random.default_rng(seed)
    term = term_law(case)
    out = {"estimates": {}, "intervals": {}, "queries": 0, "shots": 0}
    for key, f in payoff_tables(case).items():
        mu = float((term * f).sum())
        f_max = float(f.max())
        if f_max <= 0:                                     # payoff identically zero
            out["estimates"][key], out["intervals"][key] = 0.0, [0.0, 0.0]
            continue
        a = min(max(mu / f_max, 0.0), 1.0)
        eps = min(0.25, safety * tolerance(case, key, mu) / f_max)
        est, q, (lo, hi), n_shots = iqae(a, eps, rng, alpha=alpha, shots=shots, method="ch",
                                          return_shots=True)
        out["estimates"][key] = est * f_max
        out["intervals"][key] = [lo * f_max, hi * f_max]
        out["queries"] += q
        out["shots"] += n_shots
    return out


# ---------------------------------------------------------------------------
# Result files in the benchmark's submission format, and the evaluator's checks
# ---------------------------------------------------------------------------
EXECUTIONS = {"classical_cpu", "quantum_simulator", "quantum_hardware", "hybrid"}
BACKEND = "adbq IQAE emulator: exact amplitude of A, binomial shots of Q^k A (Chernoff-Hoeffding IQAE)"


def solve(case: Case, seed: int, method: str = "iqae") -> dict:
    """Estimates, intervals and resource fields for one case (no timing)."""
    if method == "exact":
        ref = exact_targets(case)
        return {"method": "adbq_exact_convolution", "execution": "classical_cpu",
                "estimates": ref, "intervals": {k: [v, v] for k, v in ref.items()},
                "resources": {"shots": 0, "qubits": 0, "model_evaluations": int(np.prod(SHAPE))}}
    r = iqae_targets(case, seed)
    return {"method": "adbq_iqae_chernoff_hoeffding", "execution": "quantum_simulator",
            "estimates": r["estimates"], "intervals": r["intervals"],
            "resources": {"shots": r["shots"], "qubits": N_QUBITS_A, "model_evaluations": r["queries"]},
            "evidence": {"provider": "adbq (this repository)", "backend": BACKEND,
                         "circuit_sha256": circuit_spec_sha256(case),
                         "model_evaluations_unit": "applications of the state preparation A (oracle queries)"}}


def check_result(sub: dict, case: Case, request: dict | None = None) -> None:
    """The checks of qjlb.validation.check_submission and qjlb.evaluation.strict_submission
    that apply to a single result file, re-implemented so the repository needs no qjlb."""
    if sub.get("problem_id") != case.problem_id:
        raise ValueError("problem_id does not match the case")
    if not isinstance(sub.get("method"), str) or not sub["method"].strip():
        raise ValueError("method must be a nonempty name")
    if sub.get("execution") not in EXECUTIONS:
        raise ValueError("invalid execution label")
    est = sub.get("estimates")
    if not isinstance(est, dict) or set(est) != set(TARGETS):
        raise ValueError("estimates must name exactly the five targets")
    for k, v in est.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise ValueError(f"{k}: estimate must be finite")
    for k, iv in sub.get("intervals", {}).items():
        if k not in TARGETS or (iv is not None and (len(iv) != 2 or iv[0] > iv[1])):
            raise ValueError(f"{k}: bad interval")
    res = sub.get("resources", {})
    for k in ("elapsed_seconds", "preparation_seconds", "shots", "qubits", "model_evaluations"):
        v = res.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or not np.isfinite(v):
            raise ValueError(f"resources.{k} must be a nonnegative number")
        if k in ("shots", "qubits", "model_evaluations") and type(v) is not int:
            raise ValueError(f"resources.{k} must be an integer")
    if res["preparation_seconds"] > res["elapsed_seconds"]:
        raise ValueError("preparation_seconds exceeds elapsed_seconds")
    if sub["execution"] == "classical_cpu" and (res["shots"] or res["qubits"]):
        raise ValueError("classical_cpu requires zero shots and qubits")
    if sub["execution"] != "classical_cpu":
        ev = sub.get("evidence", {})
        if not all(isinstance(ev.get(k), str) and ev[k].strip() for k in ("provider", "backend")):
            raise ValueError("quantum execution needs evidence.provider and evidence.backend")
        h = ev.get("circuit_sha256", "")
        if not (isinstance(h, str) and len(h) == 64 and all(ch in "0123456789abcdef" for ch in h)):
            raise ValueError("evidence.circuit_sha256 must be a SHA-256 digest")
        if res["qubits"] <= 0:
            raise ValueError("quantum execution needs positive qubits")
    if request is not None:
        if sub.get("run_id") != request["run_id"] or sub.get("seed") != request["seed"]:
            raise ValueError("run_id or seed does not match the request")
    json.dumps(sub, allow_nan=False)
