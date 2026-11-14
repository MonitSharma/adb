"""Figures for the README and proposal from results/*.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHORT = ("E[L]", "P(both pay)", "P(L > 80m)", "Stop-loss", "P(both exhausted)")
KEYS = ("term_expected_principal_loss_usd", "term_joint_attachment_probability",
        "term_portfolio_loss_exceedance_probability", "term_portfolio_stop_loss_usd",
        "term_both_bonds_exhausted_probability")
ACCENT, INK, GRID = "#2a6fdb", "#1f2328", "#e6e8eb"
GREYS = {"mc": ("#9aa0a6", "-"), "sobol": ("#6b7178", "--"), "strat": ("#9aa0a6", ":"), "is": ("#4a4f55", "-.")}
NAMES = {"mc": "Monte Carlo", "sobol": "Sobol", "strat": "Stratified", "is": "Importance", "qae": "IQAE", "qae_cond": "IQAE, conditioned"}


def main():
    d = json.loads((ROOT / "results" / "accuracy_per_query.json").read_text())
    base = d["scenarios"]["adb-base"]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9), sharey=True)
    for ax, k, title in zip(axes, KEYS, SHORT):
        r = base[k]
        for m, (c, ls) in GREYS.items():
            x = d["sobol_budgets"] if m == "sobol" else d["budgets"]
            ax.loglog(x, r[m], color=c, ls=ls, lw=1.6, label=NAMES[m])
        ax.loglog(r["qae"]["queries"], r["qae"]["rel_rmse"], color=ACCENT, lw=2.2, marker="o", ms=3, label=NAMES["qae"])
        if "qae_cond" in r:
            ax.loglog(r["qae_cond"]["queries"], r["qae_cond"]["rel_rmse"], color=ACCENT, lw=2.2, ls="--", marker="o", ms=3,
                      label=NAMES["qae_cond"])
        ax.axhline(0.01, color="#b0b4b9", lw=0.8)
        ax.set_title(f"{title}  (exact {r['exact']:.3g})", fontsize=10, color=INK)
        ax.set_xlabel("queries")
        ax.set_xlim(1e2, 2e7); ax.set_ylim(1e-4, 10)
        ax.grid(True, which="major", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("relative RMSE")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Accuracy per query, adb-base: IQAE falls as 1/M, samplers as 1/sqrt(N); grey line = 1% error",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    (ROOT / "results" / "figures").mkdir(exist_ok=True, parents=True)
    fig.savefig(ROOT / "results" / "figures" / "accuracy_per_query.png", dpi=160)


def _clean(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def risk_figures(fn="risk_accuracy.json", scenario="adb-base"):
    """CVaR accuracy per query (one panel per alpha) and VaR vs CVaR across scenarios."""
    path = ROOT / "results" / fn
    if not path.exists():
        return
    d = json.loads(path.read_text())
    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(exist_ok=True, parents=True)

    acc = d["accuracy"][scenario]
    fig, axes = plt.subplots(1, len(acc), figsize=(5.2 * len(acc), 3.9), sharey=True)
    styles = {**GREYS, "tilt": ("#1f2328", "-")}
    names = {**NAMES, "tilt": "Loss-tilted IS (tuned)"}
    for ax, (a, r) in zip(np.atleast_1d(axes), acc.items()):
        for m, (c, ls) in styles.items():
            cm = r["classical"][m]
            ax.loglog(cm["budgets"], cm["cvar_rel_rmse"], color=c, ls=ls, lw=1.6, label=names[m])
        q = r["quantum"]
        ax.loglog(q["q_certify"]["queries"], q["q_certify"]["cvar_rel_rmse"], color=ACCENT, lw=2.2,
                  marker="o", ms=3, label="IQAE: certified VaR bisection + CVaR")
        ax.loglog(q["q_matched"]["queries"], q["q_matched"]["cvar_rel_rmse"], color=ACCENT, lw=2.0,
                  ls="--", marker="o", ms=3, label="IQAE: accuracy-matched bisection")
        ax.loglog(q["q_varknown"]["queries"], q["q_varknown"]["cvar_rel_rmse"], color=ACCENT, lw=1.4,
                  ls=":", label="IQAE: CVaR stage only (VaR given)")
        ax.axhline(0.01, color="#b0b4b9", lw=0.8)
        ex = r["exact"]
        ax.set_title(f"alpha = {a}%: VaR {ex['VaR'] / 1e6:.1f}m, CVaR {ex['CVaR'] / 1e6:.1f}m",
                     fontsize=10, color=INK)
        ax.set_xlabel("queries")
        ax.set_xlim(1e2, 3e6); ax.set_ylim(1e-4, 1)
        _clean(ax)
    np.atleast_1d(axes)[0].set_ylabel("relative RMSE of CVaR")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle(f"CVaR accuracy per query, {scenario} (quantum queries include the VaR bisection)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "cvar_accuracy_per_query.png", dpi=160)
    plt.close(fig)

    ex = d["exact"]["scenarios"]
    names_s = list(ex)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(names_s))
    for ax, a in zip(axes, ("95", "99")):
        ax.bar(x - 0.2, [ex[n][f"VaR{a}"] / 1e6 for n in names_s], 0.4, color="#9aa0a6", label=f"VaR{a}")
        ax.bar(x + 0.2, [ex[n][f"CVaR{a}"] / 1e6 for n in names_s], 0.4, color=ACCENT, label=f"CVaR{a}")
        ax.set_xticks(x, [n.replace("adb-", "") for n in names_s], fontsize=8)
        ax.set_ylabel("USD m")
        ax.set_title(f"alpha = {a}%", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=8)
        _clean(ax)
    fig.suptitle("Exact three-year VaR and CVaR by scenario", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "var_cvar_by_scenario.png", dpi=160)
    plt.close(fig)


def grid_figure(fn="grid_qae.json"):
    """Best classical / IQAE query ratio against grid size, one line per target."""
    path = ROOT / "results" / fn
    if not path.exists():
        return
    d = json.loads(path.read_text())
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    styles = {"E[L]": ("#9aa0a6", ":"), "P(L>T3)": (ACCENT, "-"), "P(L>T5)": ("#1f2328", "--"), "stop-loss": ("#6b7178", "-.")}
    n = [g["n"] for g in d["grids"]]
    for k, (c, ls) in styles.items():
        ax.loglog(n, [g["targets"][k]["quantum_over_best"] for g in d["grids"]], color=c, ls=ls, marker="o", ms=3.5, lw=1.8, label=k)
    ax.axhline(1, color="#b0b4b9", lw=0.9)
    exact_n = max(g["n"] for g in d["grids"] if g["targets"]["E[L]"]["mu_source"] == "exact")
    ax.axvspan(exact_n * 1.15, max(n) * 1.3, color="#eef0f2", zorder=0)
    ax.text(exact_n * 1.25, ax.get_ylim()[1] * 0.5, "no exact law: references estimated", fontsize=8, color="#6b7178")
    ax.set_xlabel("zones in the grid (n)"); ax.set_ylabel("best classical / IQAE queries")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _clean(ax)
    ax.set_title("Above 1: IQAE needs fewer queries (1% relative RMSE, grid source model)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "grid_qae_ratio.png", dpi=160)
    plt.close(fig)


def qmcmc_figure(fn="qmcmc_chains.json"):
    """Relative RMSE of CVaR95 at the largest budget, every sampler, both worlds."""
    path = ROOT / "results" / fn
    if not path.exists():
        return
    d = json.loads(path.read_text())
    B = str(max(d["budgets"]))
    fig, axes = plt.subplots(1, len(d["worlds"]), figsize=(11, 3.8), sharex=False)
    for ax, (w, r) in zip(np.atleast_1d(axes), d["worlds"].items()):
        items = sorted(r["samplers"].items(), key=lambda kv: kv[1][B]["rel_rmse"])
        names = [k for k, _ in items]
        vals = [v[B]["rel_rmse"] * 100 for _, v in items]
        cols = [ACCENT if ("quantum" in k and "mismatched" not in k) else "#9aa0a6" for k in names]
        ax.barh(range(len(names)), vals, color=cols)
        ax.set_yticks(range(len(names)), names, fontsize=8); ax.invert_yaxis()
        ax.set_xlabel(f"relative RMSE of CVaR95 at {int(B):,} evaluations (%)")
        ax.set_title(w, fontsize=10, color=INK, loc="left")
        _clean(ax)
    fig.suptitle("QeMCMC on the 4 x 4 trigger grid (blue: quantum proposals; noiseless emulation)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "qmcmc_chains.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
    risk_figures()
    grid_figure()
    qmcmc_figure()
