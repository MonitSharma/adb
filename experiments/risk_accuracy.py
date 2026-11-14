"""VaR and CVaR: exact references, classical estimators and quantum bisection + IQAE.

Part 1 (exact): VaR / CVaR at alpha = 0.95, 0.99, 0.995 for the brief's five scenarios
and the n-country chain, with the CDF gap min |F(l) - alpha| that sets how hard VaR is
to identify.

Part 2 (accuracy per query, two-country model): relative RMSE of CVaR-hat and the VaR
hit rate against queries, for
  classical  mc, sobol, strat, is (the brief's), tilt (loss-tilted IS tuned for CVaR);
             empirical over REPS replications + exact asymptotic variance lines;
  quantum    q_certify  bisection decisions at 99% joint confidence, then IQAE on (L - v)+;
             q_matched  bisection floor matched to the CVaR tolerance (risk_qae.py);
             q_varknown CVaR stage only, VaR given (lower bound: shows the VaR cost).
Quantum queries include every bisection step and every IQAE rerun.

Writes results/risk_accuracy.json.
Run: python experiments/risk_accuracy.py [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from accuracy_per_query import queries_for                       # noqa: E402
from adbq.model import SCENARIOS                                 # noqa: E402
from adbq.portfolio import chain_portfolio                       # noqa: E402
from adbq.qae import iqae                                        # noqa: E402
from adbq.risk import (ALPHAS, _tag, cdf_gap, loss_pmf_chain,    # noqa: E402
                       loss_pmf_two_country, risk_summary, var_cvar)
from adbq.risk_baselines import RiskPathSpace                    # noqa: E402
from adbq.risk_qae import excess_amplitude, quantum_cvar         # noqa: E402

SEED = 20261113
CHAIN_N = (2, 3, 5, 10, 20, 50)
TARGETS_REL = (0.01, 0.003, 0.001)


def exact_part():
    out = {"scenarios": {}, "chain": {}}
    for name, s in SCENARIOS.items():
        v, p = loss_pmf_two_country(s)
        r = risk_summary(v, p)
        r["cdf_gap"] = {_tag(a): cdf_gap(v, p, a) for a in ALPHAS}
        out["scenarios"][name] = r
    for n in CHAIN_N:
        t0 = time.time()
        v, p = loss_pmf_chain(chain_portfolio(n))
        r = risk_summary(v, p)
        r["cdf_gap"] = {_tag(a): cdf_gap(v, p, a) for a in ALPHAS}
        r["seconds"] = time.time() - t0
        r["support_size"] = int((p > 0).sum())
        out["chain"][str(n)] = r
    return out


def classical_part(ps, budgets, sobol_budgets, reps, reps_sobol, rng):
    """{alpha: {method: {"budgets", "cvar_rel_rmse", "var_hit", "asymptotic"}}}"""
    ex = {a: var_cvar(ps.values, ps.pmf, a) for a in ALPHAS}
    res = {_tag(a): {} for a in ALPHAS}
    for m in ("mc", "sobol", "strat", "is", "tilt"):
        B = sobol_budgets if m == "sobol" else budgets
        R = reps_sobol if m == "sobol" else reps
        err = {a: np.zeros((len(B), R)) for a in ALPHAS}
        hit = {a: np.zeros((len(B), R), bool) for a in ALPHAS}
        for bi, N in enumerate(B):
            for r in range(R):
                if m == "tilt":                       # proposal depends on alpha
                    for a in ALPHAS:
                        v, c = var_cvar(ps.values, ps.sample_pmf(m, N, rng, a), a)
                        err[a][bi, r], hit[a][bi, r] = c - ex[a][1], v == ex[a][0]
                else:                                 # one sample serves every alpha
                    pmf = ps.sample_pmf(m, N, rng)
                    for a in ALPHAS:
                        v, c = var_cvar(ps.values, pmf, a)
                        err[a][bi, r], hit[a][bi, r] = c - ex[a][1], v == ex[a][0]
        for a in ALPHAS:
            rel = np.sqrt((err[a] ** 2).mean(1)) / ex[a][1]
            d = {"budgets": list(map(int, B)), "cvar_rel_rmse": rel.tolist(),
                 "var_hit": hit[a].mean(1).tolist()}
            if m != "sobol":
                av = ps.cvar_asymptotic_var(m, a)
                d["asymptotic_rel_rmse"] = [float(np.sqrt(av / N) / ex[a][1]) for N in B]
            if m == "tilt":
                _, th, lam = ps.tuned_tilt(a)
                d["theta_per_10m"], d["lambda"] = th, lam
            res[_tag(a)][m] = d
    return res


def quantum_part(values, pmf, eps_grid, reps, rng):
    res = {}
    for a in ALPHAS:
        V, C = var_cvar(values, pmf, a)
        row = {}
        for mode in ("certify", "matched"):
            q, qv, rel, hit = [], [], [], []
            for eps in eps_grid:
                runs = [quantum_cvar(values, pmf, a, eps, rng, var_mode=mode) for _ in range(reps)]
                q.append(float(np.mean([r.queries for r in runs])))
                qv.append(float(np.mean([r.queries_var for r in runs])))
                rel.append(float(np.sqrt(np.mean([(r.cvar - C) ** 2 for r in runs])) / C))
                hit.append(float(np.mean([r.var == V for r in runs])))
            row[f"q_{mode}"] = {"eps_rel": list(map(float, eps_grid)), "queries": q,
                                "queries_var_stage": qv, "cvar_rel_rmse": rel, "var_hit": hit}
        amp, f_max = excess_amplitude(values, pmf, V)
        q, rel = [], []
        for eps in eps_grid:
            runs = [iqae(amp, eps * amp, rng, alpha=0.01) for _ in range(reps)]
            q.append(float(np.mean([r[1] for r in runs])))
            rel.append(float(np.sqrt(np.mean([(V + f_max * r[0] / (1 - a) - C) ** 2 for r in runs])) / C))
        row["q_varknown"] = {"eps_rel": list(map(float, eps_grid)), "queries": q, "cvar_rel_rmse": rel,
                             "excess_amplitude": amp, "f_max": f_max}
        res[_tag(a)] = row
    return res


def needed(cl, qu):
    """Queries to reach each relative-RMSE target; (value, extrapolated?)."""
    out = {}
    for tgt in TARGETS_REL:
        row = {}
        for m, d in cl.items():
            row[m] = queries_for(d["budgets"], d["cvar_rel_rmse"], tgt)
        for m, d in qu.items():
            row[m] = queries_for(d["queries"], d["cvar_rel_rmse"], tgt)
        out[str(tgt)] = row
    return out


def main(quick: bool = False):
    rng = np.random.default_rng(SEED)
    budgets = [2**j for j in range(8, 17 if quick else 21)]
    sobol_budgets = [2**j for j in range(8, 15 if quick else 19)]
    reps, reps_sobol, reps_q = (40, 20, 40) if quick else (300, 100, 300)
    eps_grid = np.geomspace(1.0, 0.003, 6 if quick else 13)
    scen = ("adb-base",) if quick else tuple(SCENARIOS)
    out = {"seed": SEED, "alphas": list(ALPHAS), "reps": reps, "reps_sobol": reps_sobol,
           "reps_quantum": reps_q, "exact": exact_part(), "accuracy": {}}
    for name in scen:
        t0 = time.time()
        ps = RiskPathSpace(SCENARIOS[name])
        cl = classical_part(ps, budgets, sobol_budgets, reps, reps_sobol, rng)
        qu = quantum_part(ps.values, ps.pmf, eps_grid, reps_q, rng)
        out["accuracy"][name] = {
            _tag(a): {"exact": dict(zip(("VaR", "CVaR"), var_cvar(ps.values, ps.pmf, a))),
                      "classical": cl[_tag(a)], "quantum": qu[_tag(a)],
                      "queries_needed": needed(cl[_tag(a)], qu[_tag(a)])}
            for a in ALPHAS}
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)
    return out


def table(out):
    print("\nQueries to 1% relative RMSE on CVaR (* = extrapolated)")
    for name, d in out["accuracy"].items():
        for a, r in d.items():
            row = r["queries_needed"]["0.01"]
            print(f"  {name:17s} {a:>5s}  " + "  ".join(
                f"{m}={v[0]:.2e}{'*' if v[1] else ''}" for m, v in row.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    data = main(args.quick)
    fn = ROOT / "results" / ("risk_accuracy_quick.json" if args.quick else "risk_accuracy.json")
    fn.write_text(json.dumps(data, indent=1))
    table(data)
