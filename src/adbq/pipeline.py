"""End-to-end pipeline: portfolio specification (YAML) -> exact references, classical and
quantum cost estimates, logical resources, figures and a Markdown summary.

Config schema (all monetary values in USD; YAML numbers like 70e6 are accepted):

    name: adb-base                    # run name; output goes to <out>/<name>/
    model: two_country                # two_country | chain | grid_sources | grid_boltzmann
    countries:                        # the two calibrations (types A, B), alternating on
      - {name: KGZ, exceedance: [0.03883, 0.02331, 0.01790, 0.01184]}   # chains and grids
      - {name: TJK, exceedance: [0.03925, 0.01743, 0.01332, 0.01043]}
    cond_trigger: [0.075, 0.074]      # P(B triggers | A triggers), P(A | B)
    perils:
      quake: {limit: 70e6, steps: 4, common_shock_multiplier: 1.0}
      rain: {limit: 10e6, annual_prob: 0.02, rate_multiplier: 1.0, shared_share: 0.25}
    term_years: 3
    threshold: 80e6                   # two_country only; chain: 40m x n; grids: E[L] + 3 sd
    dependence: {type: pair}          # chain: {type: chain, n: 5}
                                      # grid_sources: {type: grid, rows: 4, cols: 4}
                                      # grid_boltzmann: {type: boltzmann, rows: 4, cols: 4, kappa: 1}
    targets: [E[L], P(L>T), stop-loss, VaR95, CVaR95, VaR99, CVaR99]
    accuracy: {rel_rmse: 0.01}        # cost metric: queries to this relative RMSE
    seed: 1

Every number in the report says whether it is exact or estimated, and every cost is
'queries to the accuracy target' (classical: one sampled term; quantum: one application
of A, IQAE emulated at the exact or estimated amplitude). See docs/ for the claim boundary.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .model import STEPS, Scenario

MODELS = ("two_country", "chain", "grid_sources", "grid_boltzmann")
RISK = {"VaR95": (0.95, 0), "CVaR95": (0.95, 1), "VaR99": (0.99, 0), "CVaR99": (0.99, 1),
        "VaR99.5": (0.995, 0), "CVaR99.5": (0.995, 1)}
TWO_COUNTRY_KEYS = {"E[L]": "term_expected_principal_loss_usd", "P(both pay)": "term_joint_attachment_probability",
                    "P(L>T)": "term_portfolio_loss_exceedance_probability", "stop-loss": "term_portfolio_stop_loss_usd",
                    "P(both exhausted)": "term_both_bonds_exhausted_probability"}
CHAIN_KEYS = {"E[L]": "E[L]", "P(>=2 pay)": "P(>=2 pay)", "P(L>T)": "P(L>T)", "stop-loss": "stop-loss",
              "P(>=2 exhausted)": "P(>=2 exhausted)"}
GRID_KEYS = ("E[L]", "P(L>T3)", "P(L>T5)", "stop-loss")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _num(x):
    return float(x) if isinstance(x, str) else x


def load_config(path) -> dict:
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return validate(cfg)


def validate(cfg: dict) -> dict:
    if cfg.get("model") not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    cfg.setdefault("name", "run")
    cfg.setdefault("seed", 1)
    cfg.setdefault("term_years", 3)
    cfg.setdefault("accuracy", {"rel_rmse": 0.01})
    cfg.setdefault("dependence", {"type": "pair"})
    cfg.setdefault("targets", ["E[L]", "P(L>T)", "stop-loss", "VaR95", "CVaR95", "VaR99", "CVaR99"])
    q = cfg.get("perils", {}).get("quake", {})
    if int(q.get("steps", STEPS)) != STEPS:
        raise ValueError(f"quake payout steps are fixed at {STEPS} in this model")
    cs = cfg.get("countries")
    if cs is not None and (len(cs) != 2 or any(len(c["exceedance"]) != 4 for c in cs)):
        raise ValueError("countries: exactly two calibrations, each with four exceedance probabilities")
    if cs is not None and "cond_trigger" not in cfg and tuple(c["name"] for c in cs) != ("KGZ", "TJK"):
        raise ValueError("custom countries need cond_trigger: [P(B | A), P(A | B)]")
    return cfg


def scenario(cfg: dict) -> Scenario:
    p = cfg.get("perils", {})
    q, r = p.get("quake", {}), p.get("rain", {})
    kw = dict(name=cfg["name"], years=int(cfg["term_years"]),
              quake_mult=_num(q.get("common_shock_multiplier", 1.0)),
              quake_limit=_num(q.get("limit", 70e6)), rain_limit=_num(r.get("limit", 10e6)),
              rain_annual_prob=_num(r.get("annual_prob", 0.02)), rain_mult=_num(r.get("rate_multiplier", 1.0)),
              rain_shared_share=_num(r.get("shared_share", 0.25)), threshold=_num(cfg.get("threshold", 80e6)))
    if cfg.get("countries"):
        kw["country_types"] = tuple(c["name"] for c in cfg["countries"])
        kw["exceedance"] = tuple(tuple(_num(v) for v in c["exceedance"]) for c in cfg["countries"])
    if cfg.get("cond_trigger"):
        kw["cond_trigger"] = tuple(_num(v) for v in cfg["cond_trigger"])
    return Scenario(**kw)


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------
def _risk_entries(values, pmf, targets, source):
    from .risk import var_cvar
    out = {}
    for t in targets:
        if t in RISK:
            a, i = RISK[t]
            out[t] = {"value": var_cvar(values, pmf, a)[i], "source": source}
    return out


def _iqae(a, rel, rng):
    from .grid_estimators import iqae_queries
    return float(iqae_queries(a, rel, rng, reps=100)["queries"]) if 0 < a < 1 else None


def _quantum_risk(values, pmf, t, rel, rng):
    """Queries for quantum VaR (certified bisection) or CVaR (matched bisection + IQAE) to
    reach relative RMSE ``rel``, by emulation over an epsilon grid (log interpolation)."""
    from .risk import var_cvar
    from .risk_qae import quantum_cvar, quantum_var
    a, i = RISK[t]
    v, c = var_cvar(values, pmf, a)
    if i == 0:
        runs = [quantum_var(values, pmf, a, rng) for _ in range(40)]
        return {"queries": float(np.mean([r.queries for r in runs])),
                "hit_rate": float(np.mean([r.var == v for r in runs])), "mode": "certified bisection"}
    qs, es = [], []
    for eps in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        runs = [quantum_cvar(values, pmf, a, eps, rng, var_mode="matched") for _ in range(40)]
        qs.append(np.mean([r.queries for r in runs]))
        es.append(np.sqrt(np.mean([(r.cvar - c) ** 2 for r in runs])) / c)
    es = np.minimum.accumulate(es)
    if es[-1] > rel:
        return {"queries": None, "mode": "matched bisection + IQAE", "note": "target not reached on the grid"}
    k = int(np.argmax(es <= rel))
    if k == 0:
        return {"queries": float(qs[0]), "mode": "matched bisection + IQAE"}
    tt = (np.log(rel) - np.log(es[k - 1])) / (np.log(es[k]) - np.log(es[k - 1]))
    return {"queries": float(np.exp(np.log(qs[k - 1]) + tt * (np.log(qs[k]) - np.log(qs[k - 1])))),
            "mode": "matched bisection + IQAE"}


def _sobol_queries(ps, key, mu, rel, rng, reps=16, logs=range(10, 19)):
    """Scrambled Sobol (8 scrambles per estimate): empirical relative RMSE on a grid of
    budgets, then a power-law fit on the largest budgets to reach ``rel``."""
    budgets = [2 ** j for j in logs]
    errs = []
    for N in budgets:
        e = [ps.estimate_sobol(key, N, rng) - mu for _ in range(reps)]
        errs.append(np.sqrt(np.mean(np.square(e))) / mu)
    lb, le = np.log(budgets), np.log(np.maximum(errs, 1e-300))
    if errs[-1] <= rel:
        k = int(np.argmax(np.array(errs) <= rel))
        if k == 0:
            return float(budgets[0])
        t = (np.log(rel) - le[k - 1]) / (le[k] - le[k - 1])
        return float(np.exp(lb[k - 1] + t * (lb[k] - lb[k - 1])))
    slope = np.polyfit(lb[-4:], le[-4:], 1)[0]
    slope = min(slope, -0.5)                         # never better than its fit, never worse than MC
    return float(np.exp(lb[-1] + (np.log(rel) - le[-1]) / slope))


def run_two_country(cfg, s, rng):
    from .baselines import PathSpace
    from .model import payoff_arrays, targets as tw_targets, term_distribution_sources
    from .resources import estimate
    from .risk import loss_pmf_two_country, var_cvar
    from .risk_baselines import RiskPathSpace
    rel = cfg["accuracy"]["rel_rmse"]
    term = term_distribution_sources(s)
    ex = tw_targets(term, s)
    values, pmf = loss_pmf_two_country(s)
    res = {"exact": {}, "classical": {}, "quantum": {}, "resources": {}}
    ps = PathSpace(s)
    fmax = {k: float(v.max()) for k, v in payoff_arrays(s).items()}
    for t in cfg["targets"]:
        if t in TWO_COUNTRY_KEYS:
            k = TWO_COUNTRY_KEYS[t]
            mu = ex[k]
            res["exact"][t] = {"value": mu, "source": "exact"}
            if mu <= 0:
                continue
            res["classical"][t] = {"mc": ps.var_mc(k) / (rel * mu) ** 2, "is": ps.var_is(k) / (rel * mu) ** 2,
                                   "strat": ps.var_strat(k, 10**6) * 10**6 / (rel * mu) ** 2,
                                   "sobol": _sobol_queries(ps, k, mu, rel, rng)}
            res["quantum"][t] = {"iqae": _iqae(mu / fmax[k], rel, rng), "amplitude": mu / fmax[k]}
            res["resources"][t] = estimate(2, {"P(both pay)": "P(>=2 pay)", "P(both exhausted)": "P(>=2 exhausted)"}.get(t, t)).as_dict()
    risk_t = [t for t in cfg["targets"] if t in RISK]
    res["exact"].update(_risk_entries(values, pmf, risk_t, "exact"))
    if risk_t:
        rps = RiskPathSpace(s)
        for t in risk_t:
            a, i = RISK[t]
            v, c = var_cvar(values, pmf, a)
            if i == 1:
                res["classical"][t] = {m: rps.cvar_asymptotic_var(m, a) / (rel * c) ** 2
                                       for m in ("mc", "strat", "is", "tilt")}
            res["quantum"][t] = _quantum_risk(values, pmf, t, rel, rng)
    res["note"] = ("Classical costs use exact per-query variances (MC, stratified, the brief's IS; loss-tilted IS "
                   "for CVaR). Scrambled Sobol is measured empirically (16 repetitions per budget up to 2^18 terms, power-law fit beyond); "
                   "experiments/accuracy_per_query.py measures it more precisely and is the reference.")
    return res, (values, pmf)


def run_chain(cfg, s, rng):
    from .portfolio import chain_portfolio, exact_chain
    from .resources import estimate
    from .risk import loss_pmf_chain
    n = int(cfg["dependence"]["n"])
    rel = cfg["accuracy"]["rel_rmse"]
    pf = chain_portfolio(n, s)
    ex = exact_chain(pf)
    values, pmf = loss_pmf_chain(pf)
    T = pf.scen.threshold
    Lmax = float(values[pmf > 0].max())
    fmax = {"E[L]": Lmax, "stop-loss": Lmax - T}
    second = {"E[L]": float(pmf @ values**2), "stop-loss": float(pmf @ np.maximum(values - T, 0) ** 2)}
    res = {"exact": {}, "classical": {}, "quantum": {}, "resources": {}, "threshold": T}
    # the chain is the 1 x n grid source model (tests/test_gridsources.py), so the validated
    # tilted-IS and cross-entropy samplers apply to portfolio-loss targets
    from .grid_estimators import chunked_is, cross_entropy, tune_tilt
    from .gridsources import grid_portfolio
    gp = grid_portfolio(1, n, s)
    u = gp.units[2]
    Tu = int(round(T / u))
    fns = {"E[L]": (lambda L: L * u, "value"), "P(L>T)": (lambda L: (L > Tu).astype(float), "indicator"),
           "stop-loss": (lambda L: np.maximum(L - Tu, 0) * u, "value")}
    for t in cfg["targets"]:
        if t in CHAIN_KEYS:
            mu = ex[CHAIN_KEYS[t]]
            res["exact"][t] = {"value": mu, "source": "exact"}
            if mu <= 0:
                continue
            var = second[t] - mu**2 if t in second else mu * (1 - mu)
            res["classical"][t] = {"mc": var / (rel * mu) ** 2}
            if t in fns:
                f, kind = fns[t]
                laws, _, c_t = tune_tilt(gp, f, rng, pilot=5000)
                m_t, v_t = chunked_is(gp, f, laws, 50_000, rng)
                lc, c_ce = cross_entropy(gp, f, rng, kind, Tu, N=10_000)
                m_c, v_c = chunked_is(gp, f, lc, 50_000, rng)
                ok = lambda m: abs(m / mu - 1) < 0.05                       # failure detector
                res["classical"][t]["tilt"] = c_t + v_t / (rel * mu) ** 2 if ok(m_t) else float("inf")
                res["classical"][t]["ce"] = c_ce + v_c / (rel * mu) ** 2 if ok(m_c) else float("inf")
            else:
                res["classical"][t]["note"] = "MC only (needs per-country losses)"
            a = mu / fmax.get(t, 1.0)
            res["quantum"][t] = {"iqae": _iqae(a, rel, rng), "amplitude": a}
            res["resources"][t] = estimate(n, t).as_dict()
    risk_t = [t for t in cfg["targets"] if t in RISK]
    res["exact"].update(_risk_entries(values, pmf, risk_t, "exact"))
    for t in risk_t:
        a, i = RISK[t]
        if i == 1:
            from .risk import var_cvar
            v, c = var_cvar(values, pmf, a)
            g = np.maximum(values - v, 0)
            var_g = pmf @ g**2 - (pmf @ g) ** 2
            res["classical"][t] = {"mc": float(var_g / (1 - a) ** 2 / (rel * c) ** 2),
                                   "note": "MC only (no tilted CVaR estimator for this model)"}
        res["quantum"][t] = _quantum_risk(values, pmf, t, rel, rng)
    res["note"] = ("Chain: classical costs are MC (exact variance) and, for portfolio-loss targets, exponential-"
                   "tilt and cross-entropy IS on the source model (pilot/adaptation cost included). The count "
                   "targets P(>=2 pay) and P(>=2 exhausted) have MC only, so their ratios overstate the quantum "
                   "gain; RQMC is not run here (results/grid_qae.md).")
    return res, (values, pmf)


def run_grid_sources(cfg, s, rng):
    from .grid_estimators import chunked_is, cross_entropy, target_functions, tune_tilt
    from .gridsources import EXACT_MAX_ROWS, exact_loss_pmf, grid_portfolio
    from .resources import estimate
    d = cfg["dependence"]
    gp = grid_portfolio(int(d["rows"]), int(d["cols"]), s)
    rel = cfg["accuracy"]["rel_rmse"]
    tf = target_functions(gp)
    u = gp.units[2]
    res = {"exact": {}, "classical": {}, "quantum": {}, "resources": {}, "T3": tf["_T"][0], "T5": tf["_T"][1]}
    exact = gp.rows <= EXACT_MAX_ROWS
    values = pmf = None
    if exact:
        values, pmf = exact_loss_pmf(gp)
        Lu = np.rint(values / u).astype(int)
    tmap = {"P(L>T)": "P(L>T3)"}
    for t in cfg["targets"]:
        k = tmap.get(t, t)
        if k not in GRID_KEYS:
            continue
        f, f_max = tf[k]
        laws, theta, c_t = tune_tilt(gp, f, rng, pilot=5000)
        if exact:
            mu = float(pmf @ f(Lu)); m2 = float(pmf @ f(Lu) ** 2); src = "exact"
        else:
            mu, _ = chunked_is(gp, f, laws, 200_000, rng)
            m2, _ = chunked_is(gp, lambda L: f(L) ** 2, laws, 100_000, rng); src = "estimate (tilted IS)"
        res["exact"][t] = {"value": mu, "source": src}
        _, vt = chunked_is(gp, f, laws, 50_000, rng)
        kind = "value" if k in ("E[L]", "stop-loss") else "indicator"
        Tu = int(round(tf["_T"][1 if k == "P(L>T5)" else 0] / u))
        lc, c_ce = cross_entropy(gp, f, rng, kind, Tu, N=10_000)
        m_ce, vc = chunked_is(gp, f, lc, 50_000, rng)
        ce = c_ce + vc / (rel * mu) ** 2 if abs(m_ce / mu - 1) < 0.05 else float("inf")
        res["classical"][t] = {"mc": (m2 - mu**2) / (rel * mu) ** 2, "tilt": c_t + vt / (rel * mu) ** 2, "ce": ce}
        res["quantum"][t] = {"iqae": _iqae(mu / f_max, rel, rng), "amplitude": mu / f_max}
        res["resources"][t] = estimate(gp.n, {"P(L>T3)": "P(L>T)", "P(L>T5)": "P(L>T)"}.get(k, k),
                                       edges=len(gp.edges), degree=4).as_dict()
    risk_t = [t for t in cfg["targets"] if t in RISK]
    if exact:
        from .risk import var_cvar
        res["exact"].update(_risk_entries(values, pmf, risk_t, "exact"))
        for t in risk_t:
            a, i = RISK[t]
            if i == 1:
                v, c = var_cvar(values, pmf, a)
                g = np.maximum(values - v, 0)
                res["classical"][t] = {"mc": float((pmf @ g**2 - (pmf @ g) ** 2) / (1 - a) ** 2 / (rel * c) ** 2),
                                       "note": "MC only (no tilted CVaR estimator for this model)"}
            res["quantum"][t] = _quantum_risk(values, pmf, t, rel, rng)
    res["note"] = ("Grid source model: thresholds T3, T5 = E[L] + 3 / 5 sd (exact moments); 'P(L>T)' means T3. "
                   "Beyond 4 rows the exact sweep is out of reach and references are tilted-IS estimates.")
    return res, (values, pmf)


def run_grid_boltzmann(cfg, s, rng):
    from .grid import calibrated_grid, exact_loss_pmf
    d = cfg["dependence"]
    m = calibrated_grid(int(d["rows"]), int(d["cols"]), s, kappa=_num(d.get("kappa", 1.0)))
    values, pmf = exact_loss_pmf(m)
    rel = cfg["accuracy"]["rel_rmse"]
    mean, sd = float(pmf @ values), float(np.sqrt(pmf @ values**2 - (pmf @ values) ** 2))
    T = float(np.ceil((mean + 3 * sd) / 2.5e6) * 2.5e6)
    res = {"exact": {}, "classical": {}, "quantum": {}, "resources": {}, "threshold": T}
    for t in cfg["targets"]:
        if t == "E[L]":
            res["exact"][t] = {"value": mean, "source": "exact"}
            res["classical"][t] = {"iid_mc": sd**2 / (rel * mean) ** 2}
        elif t == "P(L>T)":
            p = float(pmf[values > T].sum())
            res["exact"][t] = {"value": p, "source": "exact"}
            res["classical"][t] = {"iid_mc": (1 - p) / (p * rel**2)}
    res["exact"].update(_risk_entries(values, pmf, [t for t in cfg["targets"] if t in RISK], "exact"))
    res["note"] = ("Boltzmann (Ising) trigger law: exact by transfer matrix; 'iid_mc' assumes perfect i.i.d. draws, "
                   "which an MCMC sampler does not deliver. QeMCMC comparisons: results/qmcmc_exact.md, "
                   "results/qmcmc_chains.md. No QAE loader is built for this law.")
    return res, (values, pmf)


RUNNERS = {"two_country": run_two_country, "chain": run_chain, "grid_sources": run_grid_sources,
           "grid_boltzmann": run_grid_boltzmann}


# ---------------------------------------------------------------------------
# Orchestration, figures, summary
# ---------------------------------------------------------------------------
def run(cfg: dict, out_root: str | Path = "results/runs") -> Path:
    cfg = validate(dict(cfg))
    s = scenario(cfg)
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    res, (values, pmf) = RUNNERS[cfg["model"]](cfg, s, rng)
    out = Path(out_root) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    report = {"name": cfg["name"], "model": cfg["model"], "config": cfg,
              "scenario": {k: v for k, v in asdict(s).items()}, "seconds": time.time() - t0,
              "accuracy_target": cfg["accuracy"]["rel_rmse"], **res}
    if values is not None:
        report["loss_distribution"] = {"values_usd": values[pmf > 0].tolist(), "probabilities": pmf[pmf > 0].tolist()}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=_json_default))
    figures(report, out)
    (out / "summary.md").write_text(summary(report), encoding="utf-8")
    return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(type(o))


def _best_classical(c: dict):
    vals = {k: v for k, v in c.items() if isinstance(v, (int, float)) and np.isfinite(v)}
    return min(vals.items(), key=lambda kv: kv[1]) if vals else (None, None)


def figures(report, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ACCENT, GREY, INK = "#2a6fdb", "#9aa0a6", "#1f2328"
    ld = report.get("loss_distribution")
    if ld:
        v, p = np.array(ld["values_usd"]) / 1e6, np.array(ld["probabilities"])
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        ax.bar(v, p, width=max(1.5, 0.6 * np.min(np.diff(v)) if len(v) > 1 else 2), color=GREY)
        ax.set_yscale("log"); ax.set_xlabel("term portfolio loss (USD m)"); ax.set_ylabel("probability")
        for t, ls in (("VaR99", "--"), ("CVaR99", "-")):
            if t in report["exact"]:
                ax.axvline(report["exact"][t]["value"] / 1e6, color=ACCENT, ls=ls, lw=1.4, label=t)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"{report['name']}: exact loss distribution", fontsize=10, color=INK, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout(); fig.savefig(out / "loss_distribution.png", dpi=150); plt.close(fig)
    rows = [(t, _best_classical(report["classical"].get(t, {})), report["quantum"].get(t, {}))
            for t in report["exact"]]
    rows = [(t, b, q) for t, b, q in rows if b[1] is not None and (q.get("iqae") or q.get("queries"))]
    if rows:
        fig, ax = plt.subplots(figsize=(7.5, 0.5 + 0.45 * len(rows)))
        y = np.arange(len(rows))
        ax.barh(y - 0.18, [b[1] for _, b, _ in rows], 0.36, color=GREY, label="best classical")
        ax.barh(y + 0.18, [q.get("iqae") or q.get("queries") for _, _, q in rows], 0.36, color=ACCENT, label="quantum (IQAE)")
        ax.set_yticks(y, [t for t, _, _ in rows]); ax.set_xscale("log")
        ax.set_xlabel(f"queries to {report['accuracy_target']:.0%} relative RMSE")
        ax.legend(frameon=False, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout(); fig.savefig(out / "queries.png", dpi=150); plt.close(fig)


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:,.3g}" if abs(v) < 1e4 else f"{v:.3g}"


def summary(r) -> str:
    rel = r["accuracy_target"]
    L = [f"# {r['name']}", "", f"Model: `{r['model']}`. Generated by `adbq run` in {r['seconds']:.0f} s. "
         f"Costs are queries to {rel:.0%} relative RMSE (classical: sampled terms; quantum: applications "
         "of A with IQAE emulated, noiseless).", ""]
    if r.get("note"):
        L += [r["note"], ""]
    L += ["| target | value | source | best classical (queries) | quantum (queries) | quantum / classical |",
          "|---|---:|---|---|---:|---:|"]
    for t, e in r["exact"].items():
        val = e["value"]
        vs = f"USD {val / 1e6:,.2f}m" if abs(val) >= 1e3 else f"{val:.4g}"
        m, b = _best_classical(r["classical"].get(t, {}))
        q = r["quantum"].get(t, {})
        qv = q.get("iqae") if "iqae" in q else q.get("queries")
        ratio = f"{b / qv:.2f}x" if b and qv else "-"
        flag = " (MC only)" if r["classical"].get(t, {}).get("note") else ""
        L.append(f"| {t} | {vs} | {e['source']} | {m + ' ' + _fmt(b) + flag if m else '-'} | {_fmt(qv)} | {ratio} |")
    if r["resources"]:
        L += ["", "Logical resources per application of A (analytic model, resources.py):", "",
              "| target | logical qubits | Toffoli | T count |", "|---|---:|---:|---:|"]
        for t, x in r["resources"].items():
            L.append(f"| {t} | {x['logical_qubits']} | {x['toffoli']:,} | {x['t_count']:,} |")
    L += ["", "Ratio above 1: the quantum route needs fewer queries. A query is not a unit of time; "
          "one quantum query is a fault-tolerant circuit (see the T count). VaR rows use certified bisection "
          "(no classical column); CVaR uses accuracy-matched bisection plus IQAE."]
    if r.get("loss_distribution"):
        L += ["", "![loss distribution](loss_distribution.png)"]
    L += ["", "![queries](queries.png)"]
    return "\n".join(L) + "\n"
