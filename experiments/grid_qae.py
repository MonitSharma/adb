"""High-width comparison: IQAE vs the strongest classical samplers on the grid source model.

For square grids 2x2 ... 16x16 (src/adbq/gridsources.py):
* exact loss law by the zone-by-zone sweep where it fits (rows <= EXACT_MAX_ROWS); beyond
  that the reference values come from a large tilted-IS run and are labelled 'estimate'
  with their relative standard error;
* queries to 1% relative RMSE for MC (exact or estimated variance), randomized QMC
  (empirical, power-law extrapolation), exponential-tilt IS and cross-entropy IS (pilot
  and adaptation costs included; each IS cost validated by independent repeats, and an
  IS estimator whose mean misses the reference is marked failed), and IQAE (emulated at
  a = mu / f_max);
* logical resources per quantum query (resources.estimate with grid edges, degree 4),
  so queries can be turned into T gates. Queries are not time: see the claim boundary.

Exactness frontier: wall-clock of the exact sweep for growing row counts, and the memory
the frontier would need beyond EXACT_MAX_ROWS.

Writes results/grid_qae.json and results/grid_qae.md.
Run: python experiments/grid_qae.py [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adbq.grid_estimators import (TARGETS, chunked_is, cross_entropy, iqae_queries,  # noqa: E402
                                  rqmc_queries, target_functions, tune_tilt)
from adbq.gridsources import (EXACT_MAX_ROWS, exact_loss_pmf, exact_mean_var,         # noqa: E402
                              grid_portfolio, source_laws)
from adbq.resources import estimate                                                     # noqa: E402

SEED = 20261116
REL = 0.01
RES_TARGET = {"E[L]": "E[L]", "P(L>T3)": "P(L>T)", "P(L>T5)": "P(L>T)", "stop-loss": "stop-loss"}


def log(m):
    print(m, flush=True)


def frontier():
    rows = []
    for R, C in ((1, 8), (2, 4), (2, 8), (3, 3), (3, 6), (4, 4)):
        gp = grid_portfolio(R, C)
        t0 = time.time()
        exact_loss_pmf(gp)
        rows.append({"rows": R, "cols": C, "seconds": time.time() - t0,
                     "frontier_floats": 10 ** (R + 1) * (gp.n * gp.full_units + 1)})
        log(f"  exact {R}x{C}: {rows[-1]['seconds']:.1f}s")
    for R in (5, 6, 8):
        nb = R * R * grid_portfolio(1, 1).full_units + 1
        rows.append({"rows": R, "cols": R, "seconds": None, "frontier_floats": 10 ** (R + 1) * nb})
    return rows


def one(R, rng, quick):
    gp = grid_portfolio(R, R)
    E = len(gp.edges)
    tf = target_functions(gp)
    mu_L, var_L = exact_mean_var(gp)
    out = {"rows": R, "cols": R, "n": gp.n, "edges": E, "sources": gp.n_sources,
           "mean": mu_L, "sd": float(np.sqrt(var_L)), "T3": tf["_T"][0], "T5": tf["_T"][1], "targets": {}}
    exact = R <= EXACT_MAX_ROWS
    if exact:
        t0 = time.time()
        v, p = exact_loss_pmf(gp)
        out["exact_seconds"] = time.time() - t0
        Lu = np.rint(v / gp.units[2]).astype(int)
    pilot, n_var = (3000, 20_000) if quick else (5000, 100_000)
    n_ref = 50_000 if quick else 1_000_000
    for k in TARGETS:
        t0 = time.time()
        f, f_max = tf[k]
        kind = "value" if k in ("E[L]", "stop-loss") else "indicator"
        T_units = int(round(tf["_T"][1 if k == "P(L>T5)" else 0] / gp.units[2]))
        laws_t, theta, c_t = tune_tilt(gp, f, rng, pilot=pilot)
        row = {"kind": kind, "f_max": f_max, "tilt_theta": theta}
        if exact:
            mu = float(p @ f(Lu)); m2 = float(p @ f(Lu) ** 2)
            row.update(mu=mu, mu_source="exact")
        else:
            mu, v_ref = chunked_is(gp, f, laws_t, n_ref, rng)
            g2 = lambda L, f=f: f(L) ** 2
            m2, _ = chunked_is(gp, g2, laws_t, n_ref // 5, rng)
            row.update(mu=mu, mu_source="estimate (tilted IS)", mu_rel_se=float(np.sqrt(v_ref / n_ref) / mu))
        q = {}
        q["mc"] = (m2 - mu**2) / (REL * mu) ** 2
        _, vt = chunked_is(gp, f, laws_t, n_var, rng)
        q["tilt"] = c_t + vt / (REL * mu) ** 2
        se_mu = row.get("mu_rel_se", 0.0) * mu
        laws_ce, c_ce = cross_entropy(gp, f, rng, kind, T_units, N=pilot * 2)
        mc_, vc = chunked_is(gp, f, laws_ce, n_var, rng)
        q["ce"] = c_ce + vc / (REL * mu) ** 2
        mt_, _ = chunked_is(gp, f, laws_t, n_var, rng)
        # failure detector: an IS estimator whose mean misses the reference is not usable
        for m_, est_, var_ in (("ce", mc_, vc), ("tilt", mt_, None)):
            tol = max(0.05 * mu, 5 * np.sqrt((se_mu**2) + (var_ if var_ else 0.0) / n_var))
            if not np.isfinite(est_) or abs(est_ - mu) > tol:
                q[m_] = float("inf"); row.setdefault("failed", []).append(m_)
        # diagnostic: the per-source CE (a free categorical law per source)
        laws_ps, _ = cross_entropy(gp, f, rng, kind, T_units, N=pilot * 2, family="per_source")
        m_ps, _ = chunked_is(gp, f, laws_ps, n_var // 2, rng)
        row["ce_per_source_mean_over_ref"] = float(m_ps / mu)
        rq = rqmc_queries(gp, f, mu, REL, rng, budgets=(2**10, 2**12, 2**14) if quick else (2**11, 2**13, 2**15),
                          reps=10 if quick else 20)
        q["rqmc"] = rq["queries"]
        a = mu / f_max
        qa = iqae_queries(a, REL, rng, reps=60 if quick else 200)
        q["iqae"] = qa["queries"]
        best = min(("mc", "tilt", "ce", "rqmc"), key=lambda m: q[m])
        # Validate the predicted IS cost: realized RMSE of R independent estimates at a budget
        # N (<= 2e5); predicted = 1% * sqrt(q / N). Pilot variances of heavy-tailed weights
        # can be badly underestimated, so the realized cost replaces the prediction when it is
        # clearly (> 1.35x, about 2.5 sigma of a 20-run RMSE) worse.
        if best in ("tilt", "ce"):
            laws = laws_t if best == "tilt" else laws_ce
            Nv = int(min(2e5, q[best] - (c_t if best == "tilt" else c_ce)))
            reps_v = 20 if R <= 6 else 10
            ests = [chunked_is(gp, f, laws, max(Nv, 1000), rng)[0] for _ in range(reps_v)]
            realized = float(np.sqrt(np.mean((np.array(ests) - mu) ** 2)) / mu)
            predicted = REL * np.sqrt((q[best] - (c_t if best == "tilt" else c_ce)) / max(Nv, 1000))
            row["validation"] = {"method": best, "budget": max(Nv, 1000), "reps": reps_v, "realized_rel_rmse": realized,
                                 "predicted_rel_rmse": float(predicted), "ratio": realized / predicted}
            if realized > (1.35 if reps_v == 20 else 1.5) * predicted:   # beyond RMSE noise (~2.5 sigma)
                q[best] = (c_t if best == "tilt" else c_ce) + (q[best] - (c_t if best == "tilt" else c_ce)) \
                    * (realized / predicted) ** 2
                best = min(("mc", "tilt", "ce", "rqmc"), key=lambda m: q[m])
        res = estimate(gp.n, RES_TARGET[k], edges=E, degree=4)
        row.update(amplitude=a, amplitude_source=row["mu_source"], queries_1pct=q, rqmc=rq,
                   best_classical=best, quantum_over_best=q[best] / q["iqae"],
                   t_count_per_query=res.t_count, logical_qubits=res.logical_qubits,
                   t_gates_1pct=res.t_count * q["iqae"])
        out["targets"][k] = row
        log(f"  {R}x{R} {k}: mu={mu:.4g} ({row['mu_source']}), best classical {best} {q[best]:.3g}, "
            f"IQAE {q['iqae']:.3g} -> {row['quantum_over_best']:.2f}x  [{time.time() - t0:.0f}s]")
    return out


def render(d):
    L = ["# High-width comparison on the grid source model", "",
         "Generated by `experiments/grid_qae.py`. Queries to 1% relative RMSE. Classical: one "
         "query = one draw of every source; quantum: one application of A (IQAE emulated at "
         "a = mu / f_max). Thresholds T_k = E[L] + k sd (exact moments).", "",
         "## Where exact computation stops", "",
         "| grid | exact sweep seconds | frontier size (floats) |", "|---|---:|---:|"]
    for f in d["frontier"]:
        s = f"{f['seconds']:.1f}" if f["seconds"] is not None else "not run"
        L.append(f"| {f['rows']}x{f['cols']} | {s} | {f['frontier_floats']:.2g} |")
    L += ["", f"Exact sweep limit here: {d['exact_max_rows']} rows. A frontier of 1e9 floats is 8 GB "
          "per copy and the sweep needs about three, so 6 x 6 and wider are out of reach; "
          "numbers there are estimates.", "",
          "## Queries to 1% relative RMSE", "",
          "| grid | sources | target | mu | mu from | MC | RQMC | exp. tilt IS | CE-IS | IQAE | best classical / IQAE | T gates per query |",
          "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in d["grids"]:
        for k, r in g["targets"].items():
            q = r["queries_1pct"]
            src = "exact" if r["mu_source"] == "exact" else f"est. (+-{r['mu_rel_se']:.2%})"
            L.append(f"| {g['rows']}x{g['cols']} | {g['sources']} | {k} | {r['mu']:.4g} | {src} | "
                     + " | ".join(f"{q[m]:.2g}" for m in ("mc", "rqmc", "tilt", "ce", "iqae"))
                     + f" | **{r['quantum_over_best']:.2f}x** | {r['t_count_per_query']:.2g} |")
    L += ["", "## Checks on the classical numbers", "",
          "| grid | target | validated method | realized / predicted RMSE | failed estimators | per-source CE mean / reference |",
          "|---|---|---|---:|---|---:|"]
    for g in d["grids"]:
        for k, r in g["targets"].items():
            v = r.get("validation")
            L.append(f"| {g['rows']}x{g['cols']} | {k} | {v['method'] if v else '-'} | "
                     + (f"{v['ratio']:.2f} ({v['reps']} runs)" if v else "-") + " | "
                     + (", ".join(r.get("failed", [])) or "none") + " | "
                     + f"{r['ce_per_source_mean_over_ref']:.3g} |")
    L += ["", "A free categorical CE law per source (thousands of parameters on large grids) "
          "degenerates: its estimate collapses far below the reference (last column), with an "
          "estimated variance near zero that would wrongly suggest a tiny query count. The CE "
          "used in the main table is the exponential family with one tilt per source kind."]
    L += ["", "Ratio above 1: IQAE needs fewer queries than the best classical sampler. IS pilot "
          "and cross-entropy adaptation costs are included. RQMC beyond its measured budgets is a "
          "power-law extrapolation (never better than its fitted slope, never worse than MC's).",
          "", "Claim boundary: noiseless emulation, accuracy per query. One classical query costs "
          "microseconds; one quantum query is a fault-tolerant circuit with the T count shown."]
    return "\n".join(L) + "\n"


def main(quick):
    rng = np.random.default_rng(SEED)
    sizes = (2, 3, 4, 6) if quick else (2, 3, 4, 5, 6, 8, 12, 16)
    out = {"seed": SEED, "rel": REL, "exact_max_rows": EXACT_MAX_ROWS, "grids": []}
    log("frontier")
    out["frontier"] = frontier() if not quick else []
    for R in sizes:
        log(f"grid {R}x{R}")
        out["grids"].append(one(R, rng, quick))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    d = main(a.quick)
    sfx = "_quick" if a.quick else ""
    (ROOT / "results" / f"grid_qae{sfx}.json").write_text(json.dumps(d, indent=1))
    (ROOT / "results" / f"grid_qae{sfx}.md").write_text(render(d), encoding="utf-8")
    print(render(d))
