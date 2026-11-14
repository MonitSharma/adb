"""Interval method for the digital twin: Clopper-Pearson (CP) versus Chernoff-Hoeffding (CH).

Both IQAE variants meet the nominal 99% coverage (experiments/coverage.py), but a CP miss can
be confidently wrong by several times the value. This compares them on the twin's own tasks,
emulated at exact amplitudes and checked against exact values:

A. dependence sweeps with interval transport (shared-quake multiplier m = 0.25 ... 2.0, the
   four tail targets, half-widths 10% and 5%): queries, coverage, misses and the worst miss
   (relative distance from the exact value to the nearest end of a missed interval);
B. certified answers (five base targets, 5% half-width): coverage, worst miss of the
   point estimate, queries;
C. structure ranking by paired one-circuit sign tests (earthquake / rain splits): correct
   rate and queries.

Writes results/twin_intervals.json and results/twin_intervals.md.
Run: python experiments/twin_intervals.py      (about 10 minutes)
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adbq.model import SCENARIOS, SHORT, TARGETS                       # noqa: E402
from adbq.twin import SourceLaw, certified, paired_sign, sweep           # noqa: E402

SEED = 20261114
METHODS = ("cp", "ch")
NAMES = {"cp": "Clopper-Pearson", "ch": "Chernoff-Hoeffding"}
M_GRID = np.round(np.arange(0.25, 2.0001, 0.05), 2)
TOLS = (0.10, 0.05)
REPS_SWEEP = 20
REPS_ANSWER = 400
REPS_SIGN = 50
SPLITS = (50, 55, 60, 65, 70, 75)


def miss_distance(lo, hi, mu):
    """Relative distance from the exact value to the nearest end of an interval that misses it."""
    if lo <= mu <= hi:
        return 0.0
    return max(lo / mu - 1, 1 - hi / mu)


def sweeps(rng):
    out = []
    base = SCENARIOS["adb-base"]
    for key, short in zip(TARGETS[1:], SHORT[1:]):
        for tol in TOLS:
            row = {"target": short, "tol": tol}
            for m in METHODS:
                t0 = time.time()
                runs = [sweep(base, "quake_mult", M_GRID, key, tol, rng, method=m) for _ in range(REPS_SWEEP)]
                pts = [r for run in runs for r in run["rows"]]
                miss = [miss_distance(r["lo"], r["hi"], r["exact"]) for r in pts]
                row[m] = {"twin_queries": float(np.mean([x["twin_queries"] for x in runs])),
                          "recompute_queries": float(np.mean([x["baseline_queries"] for x in runs])),
                          "coverage": float(np.mean([r["covered"] for r in pts])),
                          "misses": int(sum(d > 0 for d in miss)), "points": len(pts),
                          "worst_miss": float(max(miss)),
                          "recompute_coverage": float(np.mean([r["base_covered"] for r in pts])),
                          "seconds": time.time() - t0}
            print(f"  sweep {short} {tol:.0%}: " + "; ".join(
                f"{m} q={row[m]['twin_queries']:.3g} cov={row[m]['coverage']:.4f} worst={row[m]['worst_miss']:.2f}"
                for m in METHODS), flush=True)
            out.append(row)
    return out


def answers(rng, hw=0.05):
    law = SourceLaw(SCENARIOS["adb-base"])
    out = []
    for key, short in zip(TARGETS, SHORT):
        f = law.payoff(key)
        mu, fm = law.mean(f), float(f.max())
        row = {"target": short, "exact": mu}
        for m in METHODS:
            res = []
            for _ in range(REPS_ANSWER):
                from adbq.twin import Ledger
                led = Ledger()
                est, lo, hi = certified(mu, fm, hw, rng, 0.01, led, "answer", m)
                res.append((est, lo, hi, led.total()))
            cov = [lo <= mu <= hi for _, lo, hi, _ in res]
            err = [abs(e / mu - 1) for e, lo, hi, _ in res if not lo <= mu <= hi]
            row[m] = {"coverage": float(np.mean(cov)), "misses": len(err), "worst_miss": float(max(err, default=0.0)),
                      "queries": float(np.mean([x[3] for x in res])),
                      "rel_rmse": float(np.sqrt(np.mean([(e / mu - 1) ** 2 for e, *_ in res])))}
        print(f"  answer {short}: " + "; ".join(f"{m} cov={row[m]['coverage']:.4f} worst={row[m]['worst_miss']:.2f}"
                                               for m in METHODS), flush=True)
        out.append(row)
    return out


def signs(rng):
    base = SCENARIOS["adb-base"]
    law = SourceLaw(base)
    total = base.quake_limit + base.rain_limit
    g = {e: law.payoff(TARGETS[3], replace(base, quake_limit=e * 1e6, rain_limit=total - e * 1e6)) for e in SPLITS}
    out = []
    for a, b in zip(SPLITS[:-1], SPLITS[1:]):
        truth = int(np.sign(law.mean(g[a]) - law.mean(g[b])))
        row = {"A": f"{a}/{int(total / 1e6) - a}", "B": f"{b}/{int(total / 1e6) - b}"}
        for m in METHODS:
            res = [paired_sign(law, g[a], g[b], rng, method=m) for _ in range(REPS_SIGN)]
            row[m] = {"correct": float(np.mean([s == truth for s, _ in res])),
                      "wrong": int(sum(s == -truth for s, _ in res)),
                      "queries": float(np.mean([n for _, n in res]))}
        out.append(row)
    return out


def render(d):
    L = ["# Interval method for the twin: Clopper-Pearson versus Chernoff-Hoeffding", "",
         "Generated by `experiments/twin_intervals.py`. IQAE at 99% confidence, emulated at exact amplitudes; "
         "every interval is checked against the exact value. 'Worst miss' is the relative distance from the exact "
         "value to the nearest end of the worst interval that excluded it (sweeps), or the relative error of the "
         "estimate in the worst missed run (answers).", "",
         f"## A. Dependence sweeps with interval transport ({len(M_GRID)} versions, {REPS_SWEEP} repetitions)", "",
         "| target | half-width | CP queries | CH queries | CH / CP | CP coverage | CH coverage | CP misses | CH misses | CP worst miss | CH worst miss |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in d["sweeps"]:
        cp, ch = r["cp"], r["ch"]
        L.append(f"| {r['target']} | {r['tol']:.0%} | {cp['twin_queries']:,.0f} | {ch['twin_queries']:,.0f} | "
                 f"{ch['twin_queries'] / cp['twin_queries']:.2f} | {cp['coverage']:.4f} | {ch['coverage']:.4f} | "
                 f"{cp['misses']} of {cp['points']} | {ch['misses']} of {ch['points']} | {cp['worst_miss']:.1%} | {ch['worst_miss']:.1%} |")
    L += ["", f"## B. Certified answers (5% half-width, {REPS_ANSWER} repetitions)", "",
          "| target | CP queries | CH queries | CH / CP | CP coverage | CH coverage | CP worst miss | CH worst miss |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in d["answers"]:
        cp, ch = r["cp"], r["ch"]
        L.append(f"| {r['target']} | {cp['queries']:,.0f} | {ch['queries']:,.0f} | {ch['queries'] / cp['queries']:.2f} | "
                 f"{cp['coverage']:.4f} | {ch['coverage']:.4f} | {cp['worst_miss']:.1%} | {ch['worst_miss']:.1%} |")
    L += ["", f"## C. Structure ranking by paired sign tests ({REPS_SIGN} repetitions)", "",
          "| A | B | CP correct | CH correct | CP wrong | CH wrong | CP queries | CH queries |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in d["signs"]:
        L.append(f"| {r['A']} | {r['B']} | {r['cp']['correct']:.0%} | {r['ch']['correct']:.0%} | {r['cp']['wrong']} | "
                 f"{r['ch']['wrong']} | {r['cp']['queries']:,.0f} | {r['ch']['queries']:,.0f} |")
    s = d["summary"]
    L += ["", "## Summary", "",
          f"Worst miss over all tasks: Clopper-Pearson {s['cp_worst']:.0%}, Chernoff-Hoeffding {s['ch_worst']:.0%}. "
          f"Lowest coverage: CP {s['cp_min_cov']:.4f}, CH {s['ch_min_cov']:.4f}. Query cost of CH relative to CP: "
          f"{s['ch_over_cp_lo']:.2f}x-{s['ch_over_cp_hi']:.2f}x (median {s['ch_over_cp_median']:.2f}x). "
          f"The twin uses {NAMES[s['chosen']]} (src/adbq/twin.py, METHOD)."]
    return "\n".join(L) + "\n"


def main():
    rng = np.random.default_rng(SEED)
    d = {"seed": SEED}
    print("sweeps", flush=True); d["sweeps"] = sweeps(rng)
    print("answers", flush=True); d["answers"] = answers(rng)
    print("signs", flush=True); d["signs"] = signs(rng)
    worst = {m: max([r[m]["worst_miss"] for r in d["sweeps"]] + [r[m]["worst_miss"] for r in d["answers"]]) for m in METHODS}
    cov = {m: min([r[m]["coverage"] for r in d["sweeps"]] + [r[m]["coverage"] for r in d["answers"]]) for m in METHODS}
    ratios = [r["ch"]["twin_queries"] / r["cp"]["twin_queries"] for r in d["sweeps"]] + \
             [r["ch"]["queries"] / r["cp"]["queries"] for r in d["answers"]] + \
             [r["ch"]["queries"] / r["cp"]["queries"] for r in d["signs"]]
    from adbq.twin import METHOD
    d["summary"] = {"cp_worst": worst["cp"], "ch_worst": worst["ch"], "cp_min_cov": cov["cp"], "ch_min_cov": cov["ch"],
                    "ch_over_cp_lo": float(min(ratios)), "ch_over_cp_hi": float(max(ratios)),
                    "ch_over_cp_median": float(np.median(ratios)), "chosen": METHOD}
    return d


if __name__ == "__main__":
    d = main()
    (ROOT / "results" / "twin_intervals.json").write_text(json.dumps(d, indent=1))
    (ROOT / "results" / "twin_intervals.md").write_text(render(d), encoding="utf-8")
    print(render(d))
