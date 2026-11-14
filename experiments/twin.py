"""Digital-twin experiments on the two-country portfolio (src/adbq/twin.py).

1. Dependence sweeps with interval transport: shared-quake multiplier m from 0.25 to 2.0 and
   shared-rain share from 0.05 to 0.5; every point certified to a relative half-width;
   twin (transport + recalibrate) versus recomputing every point. Coverage checked exactly.
2. Conditional tail-scenario library for 'both bonds exhausted': generated at m = 1, moved
   along m = 1.0 ... 2.0 by exact thinning and top-up, versus regenerating; driver shares
   from the library versus exact.
3. Structure choice: the earthquake / rain split of each USD 80m bond, ranked by stop-loss
   above USD 80m. Paired one-circuit comparison versus two separate certified estimates
   versus classical common-random-numbers Monte Carlo.
4. Budget allocation over 25 certified queries (5 targets x 5 scenarios).

Writes results/twin.json, results/twin.md and results/figures/twin_sweep.png.
Run: python experiments/twin.py      (about 2 minutes)
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adbq.model import SCENARIOS, SHORT, TARGETS, Scenario, exact          # noqa: E402
from adbq.twin import (METHOD, Ledger, ScenarioLibrary, SourceLaw, allocate, amplification_cost,  # noqa: E402
                       crn_classical_samples, drivers, paired_sign, ratio_bounds, separate_sign, sweep,
                       weighted_error)

SEED = 20261113
REPS = 20
METHOD_NAME = {"ch": "Chernoff-Hoeffding", "cp": "Clopper-Pearson"}[METHOD]
TOLS = (0.10, 0.05)
SWEEP_TARGETS = TARGETS[1:]                      # the four tail targets
M_GRID = np.round(np.arange(0.25, 2.0001, 0.05), 2)
SHARE_GRID = np.round(np.arange(0.05, 0.5001, 0.025), 3)
SPLITS = (50, 55, 60, 65, 70, 75)                # earthquake sub-limit, USD m (rain = 80 - eq)


def sweeps(rng):
    base = SCENARIOS["adb-base"]
    out = []
    for param, grid in (("quake_mult", M_GRID), ("rain_shared_share", SHARE_GRID)):
        for key, short in zip(SWEEP_TARGETS, SHORT[1:]):
            for tol in TOLS:
                runs = [sweep(base, param, grid, key, tol, rng) for _ in range(REPS)]
                rows = runs[0]["rows"]
                cov = np.mean([r["covered"] for run in runs for r in run["rows"]])
                bcov = np.mean([r["base_covered"] for run in runs for r in run["rows"]])
                tq, bq = np.mean([x["twin_queries"] for x in runs]), np.mean([x["baseline_queries"] for x in runs])
                out.append({"param": param, "target": short, "tol": tol, "points": len(grid),
                            "twin_queries": float(tq), "baseline_queries": float(bq), "saving": float(1 - tq / bq),
                            "recalibrations": float(np.mean([x["recalibrations"] for x in runs])),
                            "coverage": float(cov), "baseline_coverage": float(bcov),
                            "example": rows if (param == "quake_mult" and key == TARGETS[4] and tol == TOLS[0]) else None})
                print(f"{param:18s} {short:18s} tol {tol:.0%}: twin {tq:.3g} vs {bq:.3g} "
                      f"({1 - tq / bq:+.0%}), recal {out[-1]['recalibrations']:.1f}/{len(grid)}, cov {cov:.4f}", flush=True)
    return out


def library(rng):
    base = SCENARIOS["adb-base"]
    grid = np.round(np.arange(1.0, 2.0001, 0.1), 1)
    n = 200
    res = {"n": n, "grid": grid.tolist(), "steps": []}
    reuse_q, regen_q, kept = [], [], []
    errs = []
    exact_drivers = {float(m): drivers(SourceLaw(replace(base, quake_mult=float(m)))) for m in grid}
    for _ in range(REPS):
        led, led_regen = Ledger(), Ledger()
        law = SourceLaw(base)
        ev = law.payoff(TARGETS[4]) > 0
        lib = ScenarioLibrary.generate(law, ev, n, rng, led)
        ScenarioLibrary.generate(law, ev, n, rng, led_regen)
        k_run = []
        for m in grid[1:]:
            new = SourceLaw(replace(base, quake_mult=float(m)))
            lib, k = lib.move_to(new, n, rng, led)
            ScenarioLibrary.generate(new, ev, n, rng, led_regen)
            k_run.append(k / n)
            est = drivers(new, lib.samples)
            errs.append(max(abs(est[c] - exact_drivers[float(m)][c]) for c in est))
        reuse_q.append(led.total()); regen_q.append(led_regen.total()); kept.append(np.mean(k_run))
    p_exh = SourceLaw(base).mean(SourceLaw(base).payoff(TARGETS[4]))
    res.update({"reuse_queries": float(np.mean(reuse_q)), "regenerate_queries": float(np.mean(regen_q)),
                "saving": float(1 - np.mean(reuse_q) / np.mean(regen_q)), "mean_kept_fraction": float(np.mean(kept)),
                "max_driver_share_error": float(np.max(errs)),
                "binomial_sd_at_n": float(np.sqrt(0.25 / n)),
                "queries_per_sample_m1": amplification_cost(p_exh),
                "classical_rejection_per_sample_m1": float(1 / p_exh),
                "drivers_exact": {str(k): v for k, v in exact_drivers.items()}})
    print(f"library: reuse {res['reuse_queries']:.3g} vs regenerate {res['regenerate_queries']:.3g} "
          f"({res['saving']:+.0%}), kept {res['mean_kept_fraction']:.2f}, max share error {res['max_driver_share_error']:.3f}")
    return res


def structures(rng):
    base = SCENARIOS["adb-base"]
    law = SourceLaw(base)
    key = TARGETS[3]                                     # stop-loss above the threshold
    g = {eq: law.payoff(key, replace(base, quake_limit=eq * 1e6, rain_limit=(80 - eq) * 1e6)) for eq in SPLITS}
    ex = {eq: law.mean(g[eq]) for eq in SPLITS}
    pairs = []
    for e1, e2 in zip(SPLITS[:-1], SPLITS[1:]):
        truth = int(np.sign(ex[e1] - ex[e2]))
        P = [paired_sign(law, g[e1], g[e2], rng) for _ in range(100)]
        S = [separate_sign(law, g[e1], g[e2], rng) for _ in range(100)]
        pairs.append({"split_1": f"{e1}/{80 - e1}", "split_2": f"{e2}/{80 - e2}",
                      "stop_loss_1": ex[e1], "stop_loss_2": ex[e2],
                      "paired_queries": float(np.mean([x[1] for x in P])),
                      "paired_correct": float(np.mean([x[0] == truth for x in P])),
                      "separate_queries": float(np.mean([x[1] for x in S])),
                      "separate_correct": float(np.mean([x[0] == truth for x in S])),
                      "crn_classical_samples": crn_classical_samples(law, g[e1], g[e2])})
        p = pairs[-1]
        print(f"{p['split_1']} vs {p['split_2']}: paired {p['paired_queries']:.3g} ({p['paired_correct']:.2f}) "
              f"separate {p['separate_queries']:.3g} ({p['separate_correct']:.2f}) CRN {p['crn_classical_samples']:.3g}")
    return {"metric": "stop-loss above USD 80m", "exact": {f"{e}/{80 - e}": v for e, v in ex.items()}, "pairs": pairs}


def allocation():
    cov = json.loads((ROOT / "results" / "coverage.json").read_text())
    c = float(np.median([cell["iqae_constant"][METHOD] for cell in cov["targets"].values()]))
    kap, w, labels = [], [], []
    for name in SCENARIOS:
        ex = exact(name)
        from adbq.model import payoff_arrays
        fm = {k: float(v.max()) for k, v in payoff_arrays(SCENARIOS[name]).items()}
        for k, short in zip(TARGETS, SHORT):
            a = ex[k] / fm[k]
            kap.append(c * np.sqrt((1 - a) / a))
            w.append(4.0 if k in (TARGETS[2], TARGETS[3], TARGETS[4]) else 1.0)
            labels.append(f"{name}:{short}")
    kap, w = np.array(kap), np.array(w)
    B = 2.0e6
    eq = np.full(kap.size, B / kap.size)
    opt = allocate(kap, w, B)
    e_eq, e_opt = weighted_error(kap, w, eq), weighted_error(kap, w, opt)
    prop = B * kap / kap.sum()                         # equal relative error everywhere
    e_prop = weighted_error(kap, w, prop)
    print(f"allocation: weighted error equal {e_eq:.3g}, difficulty-proportional {e_prop:.3g}, "
          f"optimal {e_opt:.3g} ({1 - e_opt / e_eq:.0%} below equal, {1 - e_opt / e_prop:.0%} below proportional)")
    return {"budget": B, "iqae_constant": c, "weights": "4 for tail targets (exceedance, stop-loss, exhaustion), 1 otherwise",
            "weighted_error_equal": e_eq, "weighted_error_proportional": e_prop, "weighted_error_optimal": e_opt,
            "reduction": float(1 - e_opt / e_eq), "reduction_vs_proportional": float(1 - e_opt / e_prop),
            "max_share": float(opt.max() / B), "min_share": float(opt.min() / B)}


def figure(sw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ex = [x for x in sw if x["example"]][0]["example"]
    m = np.array([r["value"] for r in ex])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(m, [r["lo"] * 1e5 for r in ex], [r["hi"] * 1e5 for r in ex], color="#2a6fdb", alpha=0.18,
                    label="certified interval (twin)")
    ax.plot(m, [r["exact"] * 1e5 for r in ex], color="#1f2328", lw=1.5, label="exact")
    rc = [r for r in ex if r["mode"] == "recalibrate"]
    ax.scatter([r["value"] for r in rc], [r["exact"] * 1e5 for r in rc], color="#2a6fdb", s=18, zorder=3,
               label="fresh quantum estimate")
    ax.set_xlabel("shared-earthquake multiplier m (1 = the brief)")
    ax.set_ylabel("P(both exhausted) per 100,000 terms")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results" / "figures" / "twin_sweep.png", dpi=160)


def report(out):
    L = ["# Digital twin experiments", "", "Generated by `experiments/twin.py`. Certified estimates are IQAE with "
         f"{METHOD_NAME} bounds at 99% confidence, emulated at the exact amplitude; coverage is checked against "
         f"exact values. {REPS} repetitions per cell.", "", "## Dependence sweeps with interval transport", "",
         "| parameter | target | half-width | points | twin queries | recompute queries | saving | fresh estimates | coverage |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in out["sweeps"]:
        L.append(f"| {x['param']} | {x['target']} | {x['tol']:.0%} | {x['points']} | {x['twin_queries']:,.0f} | "
                 f"{x['baseline_queries']:,.0f} | {x['saving']:.0%} | {x['recalibrations']:.1f} | {x['coverage']:.4f} |")
    lib = out["library"]
    L += ["", "## Conditional tail scenarios (both bonds exhausted)", "",
          f"Library of {lib['n']} exact conditional samples, moved from m = 1.0 to 2.0 in steps of 0.1. "
          f"Reuse by exact thinning: {lib['reuse_queries']:,.0f} queries versus {lib['regenerate_queries']:,.0f} to "
          f"regenerate ({lib['saving']:.0%} saving); on average {lib['mean_kept_fraction']:.0%} of the library is kept "
          f"per step. Largest error of a driver share read from the library: {lib['max_driver_share_error']:.3f} "
          f"(binomial standard deviation at n = {lib['n']}: {lib['binomial_sd_at_n']:.3f}). One sample costs about "
          f"{lib['queries_per_sample_m1']:.0f} quantum queries by amplitude amplification, or "
          f"{lib['classical_rejection_per_sample_m1']:,.0f} classical scenarios by rejection.", "",
          "| m | " + " | ".join(next(iter(lib["drivers_exact"].values())).keys()) + " |",
          "|---:|" + "---:|" * len(next(iter(lib["drivers_exact"].values())))]
    for m, d in lib["drivers_exact"].items():
        L.append(f"| {float(m):.1f} | " + " | ".join(f"{v:.0%}" for v in d.values()) + " |")
    st = out["structures"]
    L += ["", "## Structure choice: earthquake / rain split ranked by stop-loss above USD 80m", "",
          "| split A | split B | stop-loss A | stop-loss B | paired circuit: queries (correct) | "
          "two estimates: queries (correct) | classical paired MC: scenarios |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for p in st["pairs"]:
        L.append(f"| {p['split_1']} | {p['split_2']} | USD {p['stop_loss_1']:,.0f} | USD {p['stop_loss_2']:,.0f} | "
                 f"{p['paired_queries']:,.0f} ({p['paired_correct']:.0%}) | {p['separate_queries']:,.0f} "
                 f"({p['separate_correct']:.0%}) | {p['crn_classical_samples']:,.0f} |")
    al = out["allocation"]
    L += ["", "## Budget allocation", "",
          f"25 certified queries (5 targets x 5 scenarios), budget {al['budget']:,.0f} quantum queries, weights "
          f"{al['weights']}. Error model: relative error = kappa / n with the measured IQAE constant "
          f"{al['iqae_constant']:.1f}. Optimal allocation lowers the weighted squared error by {al['reduction']:.0%} "
          f"against an equal split and by {al['reduction_vs_proportional']:.0%} against a split proportional to "
          f"difficulty (shares from {al['min_share']:.1%} to {al['max_share']:.1%} of the budget)."]
    (ROOT / "results" / "twin.md").write_text("\n".join(L) + "\n")


def main():
    rng = np.random.default_rng(SEED)
    out = {"reps": REPS, "sweeps": sweeps(rng), "library": library(rng), "structures": structures(rng),
           "allocation": allocation()}
    (ROOT / "results" / "twin.json").write_text(json.dumps(out, indent=1))
    figure(out["sweeps"])
    report(out)


if __name__ == "__main__":
    main()
