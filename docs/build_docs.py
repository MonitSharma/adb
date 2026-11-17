"""Render README.md and docs/proposal.md from templates and results/*.json.

Every number in the rendered documents comes from a results file written by a script in
experiments/ (or from the brief's inputs coded in src/adbq/model.py). Templates contain
only prose and {{placeholders}}; tests/test_docs.py fails if a rendered file is stale or
if a template carries a hand-typed result number.

Run: python docs/build_docs.py        (after the experiments; see README "Reproduce")
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*\}\}")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def q(x) -> str:
    """Query counts: 940, 8.3k, 113k, 12.1M."""
    if x is None or not np.isfinite(x):
        return "-"
    x = float(x)
    if x < 1e3:
        return f"{x:.0f}"
    if x < 1e4:
        return f"{x / 1e3:.1f}k"
    if x < 1e6:
        return f"{x / 1e3:.0f}k"
    if x < 1e7:
        return f"{x / 1e6:.2f}M"
    return f"{x / 1e6:.1f}M"


def r(x, d=2) -> str:
    """Ratios: 0.27x, 6.5x, 32x."""
    if x is None or not np.isfinite(x):
        return "-"
    if x >= 10:
        return f"{x:.0f}x"
    return f"{x:.{d}g}x" if x < 1 else f"{x:.1f}x" if x < 10 else f"{x:.0f}x"


def usd(x, d=1) -> str:
    return f"USD {x / 1e6:,.{d}f}m"


def pct(x, d=2) -> str:
    return f"{100 * x:.{d}f}%"


def sci(x, d=2) -> str:
    return f"{x:.{d}e}".replace("e-0", "e-").replace("e+0", "e")


def load(name):
    p = RES / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run the experiment that writes it (README, Reproduce)")
    return json.loads(p.read_text())


def table(header, rows, align=None) -> str:
    align = align or ["---"] + ["---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
def context() -> dict:
    from adbq.model import SCENARIOS, SHORT, TABLE5, TARGETS, Scenario
    c = {}
    s0 = Scenario()
    c["bond_principal"] = usd(s0.quake_limit + s0.rain_limit, 0)
    c["portfolio_principal"] = usd(2 * (s0.quake_limit + s0.rain_limit), 0)
    c["term_years"] = str(s0.years)
    c["quake_limit"] = usd(s0.quake_limit, 0)
    c["rain_limit"] = usd(s0.rain_limit, 0)
    c["threshold"] = usd(s0.threshold, 0)

    # --- accuracy per query (two-country) ---
    acc = load("accuracy_per_query.json")
    BRIEF = ("mc", "is", "strat", "sobol")             # the brief's four samplers
    UNCOND = BRIEF + ("tilt",)                         # plus a loss tilt tuned per target
    names = {"mc": "Monte Carlo", "is": "importance (brief)", "strat": "stratified", "sobol": "Sobol",
             "tilt": "tuned tilt", "cond_tilt": "conditioned tilt"}

    def best_of(need, methods):
        m = min((m for m in methods if m in need), key=lambda m: need[m][0])
        return m, need[m][0]

    rows = []
    base = acc["scenarios"]["adb-base"]
    for k, sh in zip(TARGETS, SHORT):
        need = base[k]["queries_needed"]["0.01"]
        m, b = best_of(need, UNCOND)
        qa = need["qae"][0]
        cl_txt, q_txt, r_txt = f"{names[m]}, {q(b)}", q(qa), r(b / qa)
        if "qae_cond" in need:   # like for like: both sides condition on the independent rain triggers
            bc, qc = need["cond_tilt"][0], need["qae_cond"][0]
            cl_txt += f" ({q(bc)} conditioned)"
            q_txt += f" ({q(qc)} conditioned)"
            r_txt += f" ({r(bc / qc)} conditioned)"
        ex = base[k]["exact"]
        val = usd(ex, 2) if ex > 1 else pct(ex, 4 if ex < 1e-3 else 3)
        rows.append([sh, val, cl_txt, q_txt, r_txt])
    c["table_accuracy_base"] = table(["Target", "Exact", "Best classical (queries)", "IQAE (queries)", "Ratio"], rows)
    ratios = {key: [] for key in ("el", "prob", "stop", "rare", "rare_brief")}
    for name, sc in acc["scenarios"].items():
        for k in TARGETS:
            if k not in sc:
                continue
            need = sc[k]["queries_needed"]["0.01"]
            if k == TARGETS[4]:
                best_c = min(need[m][0] for m in UNCOND + ("cond_tilt",))
                best_q = min(need["qae"][0], need["qae_cond"][0])
                ratios["rare"].append(best_c / best_q)
                ratios["rare_brief"].append(best_of(need, BRIEF)[1] / need["qae"][0])
                continue
            key = "el" if k == TARGETS[0] else "stop" if k == TARGETS[3] else "prob"
            ratios[key].append(best_of(need, UNCOND)[1] / need["qae"][0])
    for key, v in ratios.items():
        c[f"acc_{key}_lo"], c[f"acc_{key}_hi"] = r(min(v)), r(max(v))
    exh = base[TARGETS[4]]
    c["cond_amplitude_gain"] = r(exh["qae_cond"]["amplitude"] / exh["qae"]["amplitude"])
    c["cond_query_gain"] = r(exh["queries_needed"]["0.01"]["qae"][0] / exh["queries_needed"]["0.01"]["qae_cond"][0])
    c["cond_prefactor"] = pct(exh["qae_cond"]["prefactor"], 2)
    c["iqae_alpha"] = f"{1 - acc['iqae_alpha']:.0%}"

    # Table 5 reproduction
    from adbq.model import exact
    rows = []
    for name in SCENARIOS:
        e = exact(name)
        rows.append([name] + [f"{usd(e[TARGETS[0]], 2)} ({usd(TABLE5[name][0], 2)})",
                              f"{pct(e[TARGETS[1]], 3)} ({pct(TABLE5[name][1], 3)})",
                              f"{pct(e[TARGETS[2]], 3)} ({pct(TABLE5[name][2], 3)})",
                              f"{usd(e[TARGETS[3]], 3)} ({usd(TABLE5[name][3], 3)})",
                              f"{pct(e[TARGETS[4]], 4)} ({pct(TABLE5[name][4], 4)})"])
    c["table_table5"] = table(["Scenario"] + list(SHORT), rows)

    # --- scaling (chain) ---
    sc = load("scaling.json")
    by_n = {row["n"]: row for row in sc}
    nmax = max(by_n)
    c["chain_nmax"] = str(nmax)
    c["chain_tm_seconds_nmax"] = f"{by_n[nmax]['t_transfer_s']:.0f} s"
    c["chain_naive_states_nmax"] = f"10^{len(str(int(by_n[nmax]['annual_states_naive']))) - 1}"
    rows = []
    for n in sorted(by_n):
        x = by_n[n]["resources"]["P(L>T)"]
        rows.append([n, x["logical_qubits"], f"{x['toffoli']:,}", f"{x['rotations']:,}", f"{x['t_count']:,}",
                     x["toffoli_depth"], q(by_n[n]["iqae_queries_1pct"]), q(by_n[n]["mc_samples_1pct"])])
    c["table_resource_model"] = table(["n", "logical qubits", "Toffoli", "rotations", "T count", "Toffoli depth",
                                       "IQAE queries, P(>=2 exhausted) 1%", "MC samples, same"], rows)
    c["t_per_query_n2"] = f"{by_n[2]['resources']['P(L>T)']['t_count']:,}"

    # --- VaR / CVaR ---
    ra = load("risk_accuracy.json")
    ex = ra["exact"]["scenarios"]
    rows = [[n.replace("adb-", ""), usd(v["VaR95"]), usd(v["CVaR95"]), usd(v["VaR99"]), usd(v["CVaR99"]),
             usd(v["VaR99.5"]), usd(v["CVaR99.5"]), f"{v['cdf_gap']['99']:.1e} / {v['cdf_gap']['99.5']:.1e}"]
            for n, v in ex.items()]
    c["table_var_cvar"] = table(["scenario", "VaR95", "CVaR95", "VaR99", "CVaR99", "VaR99.5", "CVaR99.5",
                                 "CDF gap at 99 / 99.5"], rows)
    c["var99_base"], c["cvar99_base"] = usd(ex["adb-base"]["VaR99"]), usd(ex["adb-base"]["CVaR99"])
    c["var95_base"] = usd(ex["adb-base"]["VaR95"], 0)
    c["cvar99_indep"], c["cvar99_common"] = usd(ex["adb-independent"]["CVaR99"]), usd(ex["adb-common-shock"]["CVaR99"])
    c["var99_indep"] = usd(ex["adb-independent"]["VaR99"])
    cv = load("cvar.json")                                   # second implementation (model.py): must agree
    for n in ex:
        for a in ("0.95", "0.99", "0.995"):
            tag = {"0.95": "95", "0.99": "99", "0.995": "99.5"}[a]
            assert abs(cv[n][a]["cvar"] - ex[n][f"CVaR{tag}"]) < 1e-6 * ex[n][f"CVaR{tag}"], (n, a)
            assert cv[n][a]["var"] == ex[n][f"VaR{tag}"], (n, a)
    rows = []
    rb = ra["accuracy"]["adb-base"]
    for a, rr in rb.items():
        for tgt in ("0.01", "0.001"):
            nd = rr["queries_needed"][tgt]
            cls = {m: nd[m][0] for m in ("mc", "sobol", "strat", "is", "tilt")}
            bc = min(cls, key=cls.get)
            qq = min(nd["q_certify"][0], nd["q_matched"][0])
            rows.append([f"CVaR{a}", f"{float(tgt):.1%}", f"{bc}, {q(cls[bc])}", q(qq), q(nd["q_varknown"][0]),
                         r(cls[bc] / qq)])
    c["table_cvar_queries"] = table(["target", "rel. RMSE", "best classical", "quantum (VaR bisection + CVaR)",
                                     "quantum, VaR given", "classical / quantum"], rows)
    tg = sorted(float(x) for x in next(iter(rb.values()))["queries_needed"])
    c["cvar_loose"], c["cvar_tight"] = f"{tg[-1]:.0%}", f"{tg[0]:.1%}"
    wins = []
    for a, rr in rb.items():
        nd = rr["queries_needed"][str(tg[0])]
        best_c = min(nd[m][0] for m in ("mc", "sobol", "strat", "is", "tilt"))
        if best_c > min(nd["q_certify"][0], nd["q_matched"][0]):
            wins.append(f"CVaR{a}")
    c["cvar_tight_wins"] = " and ".join(wins) if wins else "none of the levels"
    chain = ra["exact"]["chain"]
    c["chain_cvar99_n50"] = usd(chain["50"]["CVaR99"], 0)
    c["chain_loss_seconds_n50"] = f"{chain['50']['seconds']:.2f} s"

    # --- Boltzmann grid ---
    ge = load("grid_exact.json")
    fr = {f["rows"]: f["seconds"] for f in ge["frontier"]}
    c["ising_tm_rows_max"] = str(max(fr))
    c["ising_tm_seconds_max"] = f"{fr[max(fr)]:.1f} s"
    c["ising_tm_seconds_8"] = f"{fr[8]:.2f} s"
    runs = {(g["rows"], g["scenario"], g["kappa"]): g for g in ge["runs"]}
    rows = []
    for key in [(8, "adb-independent", 1.0), (8, "adb-base", 1.0), (8, "adb-common-shock", 1.0),
                (8, "adb-base", 2.0), (8, "adb-base", 3.0), (8, "adb-base", 3.5)]:
        g = runs[key]
        rows.append([f"{g['rows']}x{g['cols']}", g["scenario"].replace("adb-", ""), f"{g['kappa']:g}", usd(g["E[L]"], 0),
                     usd(g["VaR99"], 0), usd(g["CVaR99"], 0), f"{g['p_loss_ge_60pct_max']:.1e}"])
    c["table_ising"] = table(["grid", "calibration", "kappa", "E[L]", "VaR99", "CVaR99", "P(L >= 0.6 max)"], rows)

    # --- QeMCMC ---
    qe = load("qmcmc_exact.json")
    rows = []
    for w, x in qe["worlds"].items():
        bc, bq = x["best_classical"], x["best_quantum"]
        rows.append([w, f"{bc}, {x['best_steps'][bc]:,.0f}", f"{bq}, {x['best_steps'][bq]:,.0f}",
                     f"{x['best_steps']['Local + mismatched quantum']:,.0f}",
                     r(x["quantum_speedup_over_best_classical"])])
    c["table_qmcmc_exact"] = table(["world (3 x 4)", "best classical, steps", "best quantum, steps",
                                    "mismatched-quantum control", "quantum speed-up"], rows)
    qc = load("qmcmc_chains.json")
    B = str(max(qc["budgets"]))
    rows = []
    for w, x in qc["worlds"].items():
        s = x["samplers"]
        v = x["verdict"]
        loc = s["Local MH"][B]["rel_rmse"]
        rows.append([w, f"{v['best_classical']}, {pct(s[v['best_classical']][B]['rel_rmse'])}",
                     f"{v['best_quantum']}, {pct(s[v['best_quantum']][B]['rel_rmse'])}", pct(loc),
                     f"{v['evaluation_ratio']:.3g} ({v['ratio_90'][0]:.3g}-{v['ratio_90'][1]:.3g})",
                     f"{1 / v['evaluation_ratio']:.3g}x"])
    c["table_qmcmc_chains"] = table(["world (4 x 4)", "best classical, RMSE", "best quantum, RMSE", "local MH, RMSE",
                                     "evaluation ratio (90% CI)", "quantum needs"], rows)
    c["qmcmc_budget"] = f"{int(B):,}"
    c["qmcmc_chains"] = str(qc["chains"])
    w4 = [w for w in qc["worlds"] if "4" in w][0]
    c["qmcmc_k4_quantum_needs"] = f"{1 / qc['worlds'][w4]['verdict']['evaluation_ratio']:.0f}x"

    # --- high-width grid ---
    gq = load("grid_qae.json")
    rows = []
    for g in gq["grids"]:
        t = g["targets"]
        rows.append([f"{g['rows']}x{g['cols']}", g["sources"], "exact" if t["E[L]"]["mu_source"] == "exact" else "estimated"]
                    + [r(t[k]["quantum_over_best"]) for k in ("E[L]", "P(L>T3)", "P(L>T5)", "stop-loss")]
                    + [f"{t['P(L>T3)']['t_count_per_query']:.2g}"])
    c["table_grid_qae"] = table(["grid", "sources", "reference", "E[L]", "P(L > 3 sd)", "P(L > 5 sd)", "stop-loss",
                                 "T gates / query"], rows)
    fr = [f for f in gq["frontier"] if f["seconds"] is not None]
    c["grid_exact_rows"] = str(gq["exact_max_rows"])
    c["grid_exact_4x4_seconds"] = f"{[f for f in fr if f['rows'] == 4][0]['seconds']:.0f} s"
    big = [f for f in gq["frontier"] if f["seconds"] is None]
    c["grid_frontier_6x6"] = f"{[f for f in big if f['rows'] == 6][0]['frontier_floats']:.1e}"
    first, last = gq["grids"][0], gq["grids"][-1]
    c["grid_t3_first"], c["grid_t3_last"] = r(first["targets"]["P(L>T3)"]["quantum_over_best"]), r(last["targets"]["P(L>T3)"]["quantum_over_best"])
    c["grid_t5_first"], c["grid_t5_last"] = r(first["targets"]["P(L>T5)"]["quantum_over_best"]), r(last["targets"]["P(L>T5)"]["quantum_over_best"])
    c["grid_last"] = f"{last['rows']}x{last['cols']}"
    c["grid_last_sources"] = f"{last['sources']:,}"
    ps = [g["targets"]["P(L>T5)"]["ce_per_source_mean_over_ref"] for g in gq["grids"]]
    c["ce_collapse_worst"] = f"{min(ps):.0e}"
    vr = [t["validation"]["ratio"] for g in gq["grids"] for t in g["targets"].values() if "validation" in t]
    c["is_validation_lo"], c["is_validation_hi"] = f"{min(vr):.2f}", f"{max(vr):.2f}"

    # --- real circuits ---
    rc = load("real_circuits.json")
    v = rc["verification"]
    c["qiskit_full_A_err"] = f"{max(x['rel_err'] for k, x in v['two_country_full_A'].items() if k != 'grover'):.0e}"
    c["qiskit_grover_err"] = f"{max(v['two_country_full_A']['grover'].values()):.0e}"
    c["qiskit_chain_mismatches"] = str(sum(x["mismatches"] for x in v["chain"].values()))
    rows = [[x["n"], x["qubits"], f"{x['cx']:,}", f"{x['depth']:,}", f"{x['ccx']:,}", f"{x['t_estimate_gidney']:,}",
             f"{x['model_t']:,}"] for x in rc["resources"]]
    c["table_compiled"] = table(["n", "qubits", "CX", "depth", "Toffoli", "T (compiled, Gidney)", "T (analytic model)"], rows)
    rows = []
    for name, x in rc["noisy"].items():
        if name == "full_A_prediction":
            continue
        for dev, y in x["devices"].items():
            rows.append([name.replace("_", " "), dev, f"{y['rows'][0]['cx']}-{y['rows'][-1]['cx']}",
                         f"{x['exact']:.5f}", f"{y['a_mle_k012']:.5f}", f"{100 * y['mle_rel_err']:+.1f}%"])
    c["table_noisy"] = table(["sub-circuit", "noise model", "CX (k = 0..2)", "exact a", "MLE a (k = 0, 1, 2)", "error"], rows)
    fp = rc["noisy"]["full_A_prediction"]
    c["full_A_cx"] = f"{fp['H2-1']['cx']:,}"
    c["full_A_p_ok_h2"] = f"{fp['H2-1']['p_no_error_A']:.0e}"
    c["full_A_p_ok_helios"] = f"{fp['Helios-1']['p_no_error_A']:.0e}"
    jp = rc["job_plan"]
    c["hqc_recommended"] = f"{jp['recommended']['hqc']:.0f}"
    c["hqc_all"] = f"{jp['total_hqc']:,.0f}"
    c["hw_shots"] = str(jp["shots_per_circuit"])

    # --- pipeline runs ---
    rows = []
    for p in sorted((RES / "runs").glob("*/report.json")):
        rep = json.loads(p.read_text())
        rows.append([f"`configs/{rep['name']}.yaml`", rep["model"], ", ".join(list(rep["exact"])[:4]) + (", ..." if len(rep["exact"]) > 4 else ""),
                     f"{rep['seconds']:.0f} s"])
        if rep["name"] == "adb-base":                       # pipeline must agree with the exact reference
            assert abs(rep["exact"]["E[L]"]["value"] - exact("adb-base")[TARGETS[0]]) < 1e-6
    c["table_runs"] = table(["config", "model", "targets", "run time"], rows, ["---", "---", "---", "---:"])

    # --- certified intervals ---
    cv = load("coverage.json")
    c["cov_min"] = f"{cv['min_coverage']:.2%}"
    c["cov_runs"] = f"{cv['runs']:,}"
    c["cov_nominal"] = f"{1 - cv['alpha']:.0%}"
    c["cp_saving_median"] = r(cv["cp_saving_over_ch"]["median"])
    c["cp_saving_lo"], c["cp_saving_hi"] = r(cv["cp_saving_over_ch"]["min"]), r(cv["cp_saving_over_ch"]["max"])
    rows = []
    for short, cell in cv["targets"].items():
        for rr in cell["methods"]["cp"]:
            rows.append([short, f"{rr['eps_rel']:.0%}", f"{rr['coverage']:.2%}", f"{rr['coverage_lower99']:.2%}",
                         q(rr["queries"]), pct(rr["rel_rmse"])])
    c["table_coverage"] = table(["target", "target half-width", "coverage", "99% lower bound", "queries", "rel. RMSE"], rows)
    worst = max(((cell["methods"]["cp"][i].get("worst_miss_rel_err", 0.0), short, rr["eps_rel"])
                 for short, cell in cv["targets"].items() for i, rr in enumerate(cell["methods"]["cp"])),
                key=lambda x: abs(x[0]))
    c["cov_worst_miss"] = f"{abs(worst[0]):.0%}"
    c["cov_worst_miss_cell"] = f"{worst[1]}, {worst[2]:.0%} target half-width"
    ft_method = load("ft_estimate.json")["assumptions"].get("interval_method", "cp")   # same method as the query count
    ex10 = [rr for rr in cv["targets"]["P(both exhausted)"]["methods"][ft_method] if abs(rr["eps_rel"] - 0.1) < 1e-12][0]
    c["exh_target_over_rmse"] = f"{ex10['eps_rel'] / ex10['rel_rmse']:.0f}x"
    ti = load("twin_intervals.json")                           # interval method for the twin
    s_ = ti["summary"]
    c["ti_cp_worst"], c["ti_ch_worst"] = f"{s_['cp_worst']:.0%}", f"{s_['ch_worst']:.0%}"
    c["ti_cost_median"] = f"{s_['ch_over_cp_median']:.2f}x"
    c["ti_cost_lo"], c["ti_cost_hi"] = f"{s_['ch_over_cp_lo']:.2f}x", f"{s_['ch_over_cp_hi']:.2f}x"
    c["ti_cp_wrong"] = str(sum(x["cp"]["wrong"] for x in ti["signs"]))
    c["ti_ch_wrong"] = str(sum(x["ch"]["wrong"] for x in ti["signs"]))
    c["ti_sign_total"] = str(sum(1 for _ in ti["signs"]) * 50)
    c["ti_method"] = {"ch": "Chernoff-Hoeffding", "cp": "Clopper-Pearson"}[s_["chosen"]]
    rows = []
    for x in ti["sweeps"]:
        rows.append([f"sweep, {x['target']}", f"{x['tol']:.0%}", q(x["cp"]["twin_queries"]), q(x["ch"]["twin_queries"]),
                     f"{x['cp']['coverage']:.2%}", f"{x['ch']['coverage']:.2%}",
                     f"{x['cp']['worst_miss']:.0%}", f"{x['ch']['worst_miss']:.0%}"])
    for x in ti["answers"]:
        rows.append([f"answer, {x['target']}", "5%", q(x["cp"]["queries"]), q(x["ch"]["queries"]),
                     f"{x['cp']['coverage']:.2%}", f"{x['ch']['coverage']:.2%}",
                     f"{x['cp']['worst_miss']:.0%}", f"{x['ch']['worst_miss']:.0%}"])
    c["table_twin_intervals"] = table(["task", "half-width", "CP queries", "CH queries", "CP coverage", "CH coverage",
                                       "CP worst miss", "CH worst miss"], rows)
    c["iqae_cp_constant"] = f"{float(np.median([cell['iqae_constant']['cp'] for cell in cv['targets'].values()])):.0f}"

    # --- query counts: idealised versus measured ---
    ft = load("ft_estimate.json")
    ex4 = [x for x in ft["rows"] if x["target"] == SHORT[4]]
    p_exh = ex4[0]["amplitude"]
    ideal = 1 / (0.1 * np.sqrt(p_exh))
    meas = [x for x in ex4 if x["eps_rel"] == 0.1][0]["queries"]
    c["ideal_q_exh_10"] = q(ideal)
    c["measured_q_exh_10"] = q(meas)
    c["measured_over_ideal"] = r(meas / ideal)
    c["mc_exh_10"] = q((1 - p_exh) / (p_exh * 0.01))

    # --- fault tolerance ---
    a = ft["assumptions"]
    c["ft_logical"] = str(a["logical_qubits"])
    c["ft_t_per_query"] = f"{a['t_per_query']:,}"
    c["ft_factories"] = str(a["factories"])
    rows = []
    for x in ft["rows"]:
        w = x["whole_run"]
        sec = w["seconds"]
        hum = f"{sec / 86400:.1f} days" if sec >= 86400 else f"{sec / 3600:.1f} hours"
        rows.append([x["target"], f"{x['eps_rel']:.0%}", q(x["queries"]), f"{x['k_max']:,}", f"{x['t_total']:.1e}",
                     w["distance"], f"{w['physical_qubits']:.1e}", hum])
    c["table_ft"] = table(["target", "interval half-width", "IQAE queries", "deepest Grover power", "T gates",
                           "code distance", "physical qubits", "run time"], rows)
    # --- quantum resources per certified answer (form: qubits, depth, two-qubit gates, shots) ---
    c["shots_per_round"] = str(a["shots_per_round"])
    c["cx_per_query"] = f"{a['cx_per_query']:,}"
    c["depth_per_query"] = f"{a['depth_per_query']:,}"
    rows = []
    for x in ft["rows"]:
        k = x["k_max"]
        rows.append([x["target"], f"{x['eps_rel']:.0%}", q(x["queries"]), q(x["shots"]), f"{k:,}",
                     f"{(2 * k + 1) * a['cx_per_query']:.1e}", f"{(2 * k + 1) * a['depth_per_query']:.1e}",
                     f"{x['t_total']:.1e}"])
    c["table_quantum_resources"] = table(["target", "interval half-width", "queries (applications of A)", "shots",
                                          "largest Grover power k", "two-qubit gates, deepest circuit",
                                          "depth, deepest circuit", "T gates, whole answer"], rows)

    # --- classical compute: the workstation and measured times of the heaviest steps ---
    env = load("environment.json")
    c["env_cpu_line"] = (f"{env['logical_cpus']} logical CPUs, {env['memory_gb']:.0f} GB memory, {env['os']}, "
                         f"Python {env['python']}")
    gq_fr = {(f["rows"], f["cols"]): f["seconds"] for f in load("grid_qae.json")["frontier"] if f["seconds"]}
    ge_fr = load("grid_exact.json")["frontier"]
    fullA = load("real_circuits.json")["verification"]["two_country_full_A"]
    runs = [json.loads(p_.read_text()) for p_ in sorted((RES / "runs").glob("*/report.json"))]
    sc_rows = load("scaling.json")
    rows = [
        ["Exact chain targets, 50 countries (transfer matrix)", f"{sc_rows[-1]['t_transfer_s']:.0f} s"],
        ["Exact shared-peril grid, 4 x 4 (largest exact)", f"{gq_fr[(4, 4)]:.0f} s"],
        [f"Exact Ising grid, {ge_fr[-1]['rows']} x {ge_fr[-1]['rows']} (one contraction)", f"{ge_fr[-1]['seconds']:.0f} s"],
        ["Qiskit state vector of the full two-country A (per target)",
         f"{min(v['seconds'] for k_, v in fullA.items() if k_ != 'grover'):.0f}-"
         f"{max(v['seconds'] for k_, v in fullA.items() if k_ != 'grover'):.0f} s"],
        [f"All {len(runs)} shipped portfolio configurations (`adbq run`)", f"{sum(x['seconds'] for x in runs):.0f} s"],
    ]
    c["table_classical_compute"] = table(["step", "wall-clock"], rows, ["---", "---:"])
    for k_, v in env["packages"].items():
        c[f"env_{k_.lower().replace('-', '_')}"] = v or "not installed"
    c["env_python"] = env["python"]
    c["env_os"] = env["os"]

    worst = max(ft["rows"], key=lambda x: x["whole_run"]["seconds"])
    c["ft_worst_days"] = f"{worst['whole_run']['seconds'] / 86400:.0f} days"
    c["ft_worst_qubits"] = f"{worst['whole_run']['physical_qubits']:.1e}"
    c["ft_worst_kmax"] = f"{worst['k_max']:,}"

    # --- digital twin ---
    tw = load("twin.json")
    sw = tw["sweeps"]
    mq = [x for x in sw if x["param"] == "quake_mult" and x["tol"] == 0.1]
    rs = [x for x in sw if x["param"] == "rain_shared_share"]
    c["twin_reps"] = str(tw["reps"])
    c["twin_m_points"] = str(mq[0]["points"])
    c["twin_save_lo"] = f"{min(x['saving'] for x in mq):.0%}"
    c["twin_save_hi"] = f"{max(x['saving'] for x in mq):.0%}"
    m5 = [x for x in sw if x["param"] == "quake_mult" and x["tol"] == 0.05]
    c["twin_save5_lo"] = f"{min(x['saving'] for x in m5):.0%}"
    c["twin_save5_hi"] = f"{max(x['saving'] for x in m5):.0%}"
    c["twin_cov_min"] = f"{min(x['coverage'] for x in sw):.2%}"
    c["twin_rain_save_absmax"] = f"{max(abs(x['saving']) for x in rs):.0%}"
    rows = [[x["param"].replace("quake_mult", "shared quakes m").replace("rain_shared_share", "shared-rain share"),
             x["target"], f"{x['tol']:.0%}", x["points"], q(x["twin_queries"]), q(x["baseline_queries"]),
             f"{x['saving']:+.0%}", f"{x['recalibrations']:.1f}", f"{x['coverage']:.2%}"] for x in sw]
    c["table_twin_sweep"] = table(["sweep", "target", "half-width", "versions", "twin queries", "recompute queries",
                                   "saving", "fresh estimates", "coverage"], rows)
    lib = tw["library"]
    c["lib_n"] = str(lib["n"])
    c["lib_saving"] = f"{lib['saving']:.0%}"
    c["lib_kept"] = f"{lib['mean_kept_fraction']:.0%}"
    c["lib_q_per_sample"] = f"{lib['queries_per_sample_m1']:.0f}"
    c["lib_classical_per_sample"] = f"{lib['classical_rejection_per_sample_m1']:,.0f}"
    c["lib_max_err"] = f"{lib['max_driver_share_error']:.2f}"
    d1, d2 = lib["drivers_exact"]["1.0"], lib["drivers_exact"]["2.0"]
    k_sh, k_own, k_rain = list(d1)
    c["drv_shared_m1"], c["drv_shared_m2"] = f"{d1[k_sh]:.0%}", f"{d2[k_sh]:.0%}"
    c["drv_own_m1"], c["drv_own_m2"] = f"{d1[k_own]:.0%}", f"{d2[k_own]:.0%}"
    c["drv_rain_m1"] = f"{d1[k_rain]:.0%}"
    st = tw["structures"]
    rows = [[p["split_1"], p["split_2"], usd(p["stop_loss_1"], 3), usd(p["stop_loss_2"], 3),
             f"{q(p['paired_queries'])} ({p['paired_correct']:.0%})", f"{q(p['separate_queries'])} ({p['separate_correct']:.0%})",
             q(p["crn_classical_samples"])] for p in st["pairs"]]
    c["table_structures"] = table(["split A", "split B", "stop-loss A", "stop-loss B", "paired circuit (correct)",
                                   "two estimates (correct)", "classical paired MC"], rows)
    gain = [p["separate_queries"] / p["paired_queries"] for p in st["pairs"]]
    crn = [p["paired_queries"] / p["crn_classical_samples"] for p in st["pairs"]]
    c["paired_gain_lo"], c["paired_gain_hi"] = r(min(gain)), r(max(gain))
    c["crn_win_lo"], c["crn_win_hi"] = r(min(crn)), r(max(crn))
    al = tw["allocation"]
    c["alloc_vs_equal"] = f"{al['reduction']:.0%}"
    c["alloc_vs_prop"] = f"{al['reduction_vs_proportional']:.0%}"
    from adbq.model import loss_distribution, var_cvar
    rows = []
    s_base = SCENARIOS["adb-base"]
    from dataclasses import replace as _rep
    from adbq.model import targets as _tg, term_distribution_sources as _td
    for eq in (50, 55, 60, 65, 70, 75):
        sx = _rep(s_base, quake_limit=eq * 1e6, rain_limit=(80 - eq) * 1e6)
        lv, pr = loss_distribution(sx)
        v99, c99 = var_cvar(lv, pr, 0.99)
        tt = _tg(_td(sx), sx)
        rows.append([f"{eq}/{80 - eq}", usd(tt[TARGETS[0]], 2), usd(v99, 2), usd(c99, 1), usd(tt[TARGETS[3]], 3),
                     pct(tt[TARGETS[2]], 2)])
    c["table_splits"] = table(["earthquake / rain (USD m)", "E[L]", "VaR99", "CVaR99", "stop-loss", "P(L > T)"], rows)
    tr = json.loads((RES / "twin_runs" / "twin-adb-base" / "summary.md").exists() and
                    (RES / "twin_runs" / "twin-adb-base" / "report.json").read_text() or "{}")
    c["twin_run_seconds"] = f"{tr.get('seconds', 0):.0f} s"

    # --- prior work (cited from the team's manuscripts) ---
    pw = json.loads((ROOT / "docs" / "prior_work.json").read_text())   # cited, not generated
    nv, ar = pw["newsvendor"], pw["arithmetic"]
    c["ar_auto"], c["ar_tarf"] = r(ar["autocallable_saving"]), r(ar["tarf_saving"])
    c["ar_tarf_tof"], c["ar_tarf_q"] = f"{ar['tarf_toffoli']:,}", str(ar["tarf_qubits"])
    c["ar_ref_tof"], c["ar_ref_q"] = f"{ar['tarf_ref_toffoli']:,}", str(ar["tarf_ref_qubits"])
    c["ar_holdout"] = f"{ar['holdout_paths']:,}"
    rc2 = load("real_circuits.json")["resources"][0]
    c["tof_t_share"] = pct(4 * rc2["ccx"] / rc2["t_estimate_gidney"], 0)
    c["tof_n2"], c["rot_n2"] = str(rc2["ccx"]), str(rc2["arbitrary_rotations"])
    c["compiled_over_model"] = r(rc2["t_estimate_gidney"] / rc2["model_t"])

    # --- benchmark interface (joint-loss-baseline submission format) ---
    from adbq.benchmark import case_from_config, exact_targets, tolerance
    worst, secs, shots, evals = 0.0, [], [], []
    for name in SCENARIOS:
        sub = json.loads((RES / "benchmark_submission" / f"{name}.json").read_text())
        case = case_from_config(ROOT / "benchmark_inputs" / "scenarios" / f"{name}.json")
        ref = exact_targets(case)
        for k in TARGETS:
            worst = max(worst, abs(sub["estimates"][k] - ref[k]) / tolerance(case, k, ref[k]))
        secs.append(sub["resources"]["elapsed_seconds"])
        shots.append(sub["resources"]["shots"])
        evals.append(sub["resources"]["model_evaluations"])
    c["bm_worst"] = pct(worst, 0)
    c["bm_sec_lo"], c["bm_sec_hi"] = f"{min(secs):.2f}", f"{max(secs):.2f} s"
    c["bm_shots"] = f"{max(shots):,}"
    c["bm_evals_lo"], c["bm_evals_hi"] = q(min(evals)), q(max(evals))
    c["pw_cos"] = f"{nv['certify_or_stop_classical_saving_median']:.0%}"
    c["pw_probe"] = f"{nv['quantum_probe_share_max']:.1%}"
    c["pw_runs"] = str(nv["hardware_correct_order_runs"])
    c["pw_device"] = nv["hardware_device"]
    c["pw_c_short"], c["pw_g_short"] = f"{nv['contrast_short']:.2f}", str(nv["gates_short"])
    c["pw_c_long"], c["pw_g_long"] = f"{nv['contrast_long']:.2f}", str(nv["gates_long"])
    c["exh_p"] = sci(p_exh, 2)
    # --- baseline sensitivity (settings the brief leaves open) ---
    bs = load("baseline_sensitivity.json")
    shifts = [abs(r_["ratio_with_variants"] / r_["ratio_shipped"] - 1)
              for rows_ in bs["rows"].values() for r_ in rows_.values()]
    c["bs_max_shift"] = pct(max(shifts), 1)
    c["bs_changed"] = str(len(bs["best_classical_changed"]))
    c["bs_mix_lo"], c["bs_mix_hi"] = f"{min(bs['is_mix']):g}", f"{max(bs['is_mix']):g}"
    return c


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(template: str, ctx: dict) -> str:
    missing = sorted({m.group(1) for m in PLACEHOLDER.finditer(template)} - set(ctx))
    if missing:
        raise KeyError(f"unknown placeholders: {missing}")
    return PLACEHOLDER.sub(lambda m: ctx[m.group(1)], template)


TARGETS_OUT = {"docs/proposal.template.md": "docs/proposal.md", "docs/README.template.md": "README.md"}
BANNER = "<!-- Generated by docs/build_docs.py from {src} and results/*.json. Edit the template, not this file. -->\n"


def build(write: bool = True) -> dict:
    ctx = context()
    out = {}
    for src, dst in TARGETS_OUT.items():
        text = BANNER.format(src=src) + render((ROOT / src).read_text(encoding="utf-8"), ctx)
        out[dst] = text
        if write:
            (ROOT / dst).write_text(text, encoding="utf-8")
    return out


FORM_TEMPLATE, FORM_OUT = "docs/application_form.template.md", "submission/application_form.md"
WORD_LIMITS = {"Team Name": 10, "Team Experience and Qualifications": 300, "Solution Title": 15,
               "Executive Summary": 300, "Expected Advantage Over Classical Approaches": 200, "Additional Notes": 150}


def form_fields(text: str) -> dict:
    """{field name: body} for every '## Field (...)' section of the rendered form."""
    out = {}
    for block in re.split(r"(?m)^## ", text)[1:]:
        head, _, body = block.partition("\n")
        out[re.sub(r"\s*\(.*\)\s*$", "", head).strip()] = body.strip()
    return out


def build_form(ctx: dict | None = None, write: bool = True) -> str:
    """Draft answers for the application form (written to the git-ignored submission/ folder)."""
    text = render((ROOT / FORM_TEMPLATE).read_text(encoding="utf-8"), ctx or context())
    if write:
        (ROOT / "submission").mkdir(exist_ok=True)
        (ROOT / FORM_OUT).write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    for dst in build():
        print("wrote", dst)
    build_form()
    print("wrote", FORM_OUT)
