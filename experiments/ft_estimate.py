"""Fault-tolerant cost of the quantum route on a surface-code machine (order of magnitude).

Inputs (all measured in this repository):
* per-query cost of A from the compiled Qiskit circuit (results/real_circuits.json, n = 2:
  logical qubits and T count with Gidney adders);
* the IQAE schedule for a target (mean total queries and deepest Grover power k_max at a
  relative accuracy), emulated with Clopper-Pearson bounds at 99% confidence.

Model (rules of thumb, stated so they can be replaced by a proper estimator in Phase 2):
* physical error rate p = 1e-3, surface-code threshold 1e-2; logical error per patch per code
  cycle p_L(d) = 0.1 (p / 0.01)^((d + 1) / 2)  (Fowler et al. 2012);
* each logical qubit is one d x d patch of 2 d^2 physical qubits, plus 50% routing space;
* T states from two-level 15-to-1 distillation (output error ~ 35 (35 p^3)^3, far below
  1 / T count); one factory is about 142 d^2 physical qubits and emits one T state every
  6 d code cycles; T gates are consumed one after another (no T-parallelism assumed);
* code distance d is the smallest odd d with total logical failure <= 1% over the run;
* code cycle 1 microsecond (superconducting-class); trapped-ion cycles are slower.

Writes results/ft_estimate.json and results/ft_estimate.md.
Run: python experiments/ft_estimate.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import adbq.qae as qae                                              # noqa: E402
from adbq.circuits import apply, prob_good                          # noqa: E402
from adbq.model import SCENARIOS, SHORT, TARGETS                    # noqa: E402
from adbq.oracle import build_oracle                                # noqa: E402
from adbq.twin import METHOD                                        # noqa: E402  (interval method the twin uses)

METHOD_NAME = {"ch": "Chernoff-Hoeffding", "cp": "Clopper-Pearson"}[METHOD]
SHOTS_PER_ROUND = 100                                               # qae.iqae default

P_PHYS, P_TH = 1e-3, 1e-2
CYCLE_S = 1e-6
FACTORIES = 4
FAIL = 0.01
SEED = 20261113


def schedule(a: float, eps_rel: float, rng, reps: int = 200):
    """Mean total queries, the largest Grover power, and the mean number of circuit executions
    (shots) used by IQAE with the twin's interval method."""
    ks = []
    orig = qae._find_next_k

    def spy(*args, **kw):
        out = orig(*args, **kw)
        ks.append(out[0])
        return out
    qae._find_next_k = spy
    try:
        qs, kmax, shots = [], [], []
        for _ in range(reps):
            ks.clear()
            _, n, _ = qae.iqae(a, eps_rel * a, rng, alpha=0.01, method=METHOD, shots=SHOTS_PER_ROUND)
            qs.append(n)
            kmax.append(max(ks) if ks else 0)
            shots.append(len(ks) * SHOTS_PER_ROUND)        # one round = SHOTS_PER_ROUND executions of Q^k A
    finally:
        qae._find_next_k = orig
    return float(np.mean(qs)), int(np.max(kmax)), float(np.mean(shots))


def surface_code(n_logical: int, t_count: float) -> dict:
    patches = 1.5 * n_logical
    for d in range(3, 101, 2):
        cycles = t_count * 6 * d / FACTORIES
        p_l = 0.1 * (P_PHYS / P_TH) ** ((d + 1) / 2)
        area = patches + FACTORIES * 142 / 2           # factories in units of 2 d^2 patches
        if area * cycles * p_l <= FAIL:
            phys = 2 * d * d * patches + FACTORIES * 142 * d * d
            return {"distance": d, "physical_qubits": float(phys), "code_cycles": float(cycles),
                    "seconds": float(cycles * CYCLE_S)}
    return {"distance": None}


def human(sec: float) -> str:
    for unit, s in (("years", 3.156e7), ("days", 86400), ("hours", 3600), ("minutes", 60)):
        if sec >= s:
            return f"{sec / s:.1f} {unit}"
    return f"{sec:.1f} s"


def main():
    rng = np.random.default_rng(SEED)
    rc = json.loads((ROOT / "results" / "real_circuits.json").read_text())
    a2 = [x for x in rc["resources"] if x["n"] == 2][0]
    n_log, t_a = a2["qubits"], a2["t_estimate_gidney"]
    s = SCENARIOS["adb-base"]
    rows = []
    for key, short in ((TARGETS[0], SHORT[0]), (TARGETS[2], SHORT[2]), (TARGETS[4], SHORT[4])):
        o = build_oracle(s, key)
        a = prob_good(apply({0: 1.0}, o.A), o.good)
        for eps in (0.1, 0.01):
            q, kmax, shots = schedule(a, eps, rng)
            deep = (2 * kmax + 1) * t_a
            total = q * t_a
            d1, d2 = surface_code(n_log, deep), surface_code(n_log, total)
            rows.append({"target": short, "eps_rel": eps, "amplitude": a, "queries": q, "k_max": kmax, "shots": shots,
                         "t_deepest_circuit": deep, "t_total": total, "deepest": d1, "whole_run": d2})
            print(f"{short:18s} {eps:.0%}: queries {q:.3g}, k_max {kmax}, T total {total:.2e}, d {d2['distance']}, "
                  f"{d2['physical_qubits']:.2e} qubits, {human(d2['seconds'])}", flush=True)
    out = {"assumptions": {"p_phys": P_PHYS, "threshold": P_TH, "cycle_s": CYCLE_S, "factories": FACTORIES,
                           "total_failure": FAIL, "logical_qubits": n_log, "t_per_query": t_a,
                           "interval_method": METHOD, "shots_per_round": SHOTS_PER_ROUND,
                           "cx_per_query": a2["cx"], "depth_per_query": a2["depth"],
                           "source": "compiled n = 2 chain circuit, results/real_circuits.json"},
           "rows": rows}
    (ROOT / "results" / "ft_estimate.json").write_text(json.dumps(out, indent=1))
    L = ["# Fault-tolerant cost (order of magnitude)", "",
         f"Generated by `experiments/ft_estimate.py`. Per query: {n_log} logical qubits and {t_a:,} T gates (compiled "
         f"circuit, n = 2). Surface code at physical error {P_PHYS:g}, {FACTORIES} two-level distillation factories, "
         f"code cycle {CYCLE_S * 1e6:g} microsecond, total failure at most {FAIL:.0%}. IQAE with {METHOD_NAME} bounds "
         "at 99% confidence. Rules of thumb; see the script for the formulas.", "",
         "| target | interval half-width | IQAE queries | shots | deepest Grover power | T gates, whole run | code distance | physical qubits | run time |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        w = r["whole_run"]
        L.append(f"| {r['target']} | {r['eps_rel']:.0%} | {r['queries']:,.0f} | {r['shots']:,.0f} | {r['k_max']:,} | {r['t_total']:.1e} | "
                 f"{w['distance']} | {w['physical_qubits']:.1e} | {human(w['seconds'])} |")
    L += ["", "A classical query is microseconds on a laptop, so the quantum route is far slower in wall-clock time at "
          "this size; the comparison in this project is accuracy per query, as the brief asks."]
    (ROOT / "results" / "ft_estimate.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
