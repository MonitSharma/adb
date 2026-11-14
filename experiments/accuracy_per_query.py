"""Accuracy per query: classical baselines from the brief versus iterative QAE.

For every scenario and target this records relative RMSE against the number of queries
(classical: one sampled three-year path; quantum: one application of A or A^dagger).

* Monte Carlo, stratified and importance sampling: exact variance over all 10^6 paths.
* Scrambled Sobol: empirical, 8 scrambles per estimate, 40 repetitions per budget.
* IQAE (alpha = 0.01): shot-level emulation of the verified oracle's amplitude, 400 runs
  per epsilon. RMSE includes the rare runs whose confidence interval misses (<= alpha).
* IQAE + conditioned loading (both-exhausted target only): rain register loaded
  conditioned on r1 = r2 = 1, estimate rescaled by P(r1 = r2 = 1).

Writes results/accuracy_per_query.json. Run: python experiments/accuracy_per_query.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adbq.baselines import PathSpace                       # noqa: E402
from adbq.circuits import apply, prob_good                 # noqa: E402
from adbq.model import SCENARIOS, SHORT, TARGETS, exact    # noqa: E402
from adbq.oracle import build_oracle                       # noqa: E402
from adbq.qae import iqae                                  # noqa: E402

BUDGETS = [2**j for j in range(8, 25)]
SOBOL_BUDGETS = [2**j for j in range(8, 21)]
EPS_REL = np.geomspace(0.5, 3e-4, 16)
SEED = 20261112


def queries_for(curve_q, curve_e, target):
    """Log-log interpolation of the queries needed to reach a relative RMSE."""
    q, e = np.log(np.asarray(curve_q, float)), np.log(np.asarray(curve_e, float))
    order = np.argsort(q)
    q, e = q[order], e[order]
    # running minimum makes the curve monotone for interpolation
    e = np.minimum.accumulate(e)
    if e[-1] > np.log(target):
        slope = (e[-1] - e[-3]) / (q[-1] - q[-3])
        return float(np.exp(q[-1] + (np.log(target) - e[-1]) / slope)), True
    i = int(np.argmax(e <= np.log(target)))
    if i == 0:
        return float(np.exp(q[0])), False
    t = (np.log(target) - e[i - 1]) / (e[i] - e[i - 1])
    return float(np.exp(q[i - 1] + t * (q[i] - q[i - 1]))), False


ALPHA = 0.01   # IQAE confidence level 1 - alpha


def run(scenarios=tuple(SCENARIOS), reps_sobol=40, reps_q=400):
    rng = np.random.default_rng(SEED)
    out = {"iqae_alpha": ALPHA, "budgets": BUDGETS, "sobol_budgets": SOBOL_BUDGETS, "scenarios": {}}
    for name in scenarios:
        t0 = time.time()
        s = SCENARIOS[name]
        ps = PathSpace(s)
        ex = exact(name)
        res = {}
        # Sobol: draw once per rep and budget, evaluate all targets
        sob = {k: [] for k in TARGETS}
        for N in SOBOL_BUDGETS:
            errs = {k: [] for k in TARGETS}
            for _ in range(reps_sobol):
                ests = {k: [] for k in TARGETS}
                from scipy.stats import qmc
                for _sc in range(8):
                    u = qmc.Sobol(d=3, scramble=True, seed=rng).random_base2(int(np.log2(N // 8)))
                    i, j, kk = (ps._draw(ps.p, u[:, d]) for d in range(3))
                    for k in TARGETS:
                        ests[k].append(ps.f[k][i, j, kk].mean())
                for k in TARGETS:
                    errs[k].append(np.mean(ests[k]) - ex[k])
            for k in TARGETS:
                sob[k].append(float(np.sqrt(np.mean(np.square(errs[k]))) / ex[k]))
        for k, short in zip(TARGETS, SHORT):
            mu = ex[k]
            r = {"exact": mu}
            if mu == 0:
                continue
            r["mc"] = [float(np.sqrt(ps.var_mc(k) / N) / mu) for N in BUDGETS]
            r["is"] = [float(np.sqrt(ps.var_is(k) / N) / mu) for N in BUDGETS]
            r["strat"] = [float(np.sqrt(ps.var_strat(k, N)) / mu) for N in BUDGETS]
            r["sobol"] = sob[k]
            variants = [("qae", False)] + ([("qae_cond", True)] if k == TARGETS[4] else [])
            for label, cond in variants:
                orc = build_oracle(s, k, condition_rain=cond)
                a = prob_good(apply({0: 1.0}, orc.A), orc.good)
                assert abs(orc.prefactor * orc.f_max * a - mu) / mu < 1e-10
                qs, es, cov = [], [], []
                for eps in EPS_REL:
                    runs = [iqae(a, eps * a, rng, alpha=ALPHA) for _ in range(reps_q)]
                    est = np.array([x[0] for x in runs])
                    qs.append(float(np.mean([x[1] for x in runs])))
                    es.append(float(np.sqrt(np.mean((est - a) ** 2)) / a))
                    cov.append(float(np.mean([x[2][0] <= a <= x[2][1] for x in runs])))
                r[label] = {"amplitude": a, "f_max": orc.f_max, "prefactor": orc.prefactor,
                            "queries": qs, "rel_rmse": es, "coverage": cov}
            # queries to reach 10% and 1% relative RMSE
            need = {}
            for tgt in (0.1, 0.01):
                row = {}
                for m in ("mc", "is", "strat"):
                    row[m] = queries_for(BUDGETS, r[m], tgt)
                row["sobol"] = queries_for(SOBOL_BUDGETS, r["sobol"], tgt)
                for label, _ in variants:
                    row[label] = queries_for(r[label]["queries"], r[label]["rel_rmse"], tgt)
                need[str(tgt)] = row
            r["queries_needed"] = need
            res[k] = r
        out["scenarios"][name] = res
        print(f"{name}: {time.time() - t0:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    data = run()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "accuracy_per_query.json").write_text(json.dumps(data, indent=1))
    base = data["scenarios"]["adb-base"]
    print("\nQueries to reach 1% relative RMSE, adb-base (extrapolated marked *)")
    for k, short in zip(TARGETS, SHORT):
        row = base[k]["queries_needed"]["0.01"]
        print(f"  {short:18s}", "  ".join(f"{m}={v[0]:.2e}{'*' if v[1] else ''}" for m, v in row.items()))
