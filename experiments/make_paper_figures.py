"""Publication-quality figures for the proposal, from results/*.json.

Writes docs/figures/<name>.pdf (vector, for papers) and docs/figures/<name>.png (600 dpi, for the
Word-built proposal PDF). Figures are drawn at their printed size (the A4 text block is 6.85 in wide),
so the type is 7-9 pt on the page, matching the document's Calibri body text.

Colour: IQAE is always blue; the classical samplers use orange, aqua, and greys/black separated by
lightness, dash and marker, so no series is told apart by colour alone and the figures survive
greyscale printing. The three hues pass the all-pairs colour-vision check (dataviz validator).

    python experiments/make_paper_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "figures"

TEXT_W = 6.85                       # inches: A4 text block at the proposal's margins
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, RULE, GRID = "#1b1b1b", "#52514e", "#8a8984", "#b5b4ae", "#e9e8e4"

# Classical samplers: (label, colour, dash, marker). Same entity, same style in every figure.
CLASSICAL = {
    "mc":        ("Monte Carlo",                     MUTED,  "-",            None),
    "sobol":     ("Scrambled Sobol",                 ORANGE, (0, (5, 1.6)),  "s"),
    "strat":     ("Stratified",                      AQUA,   (0, (1, 1.4)),  "^"),
    "is":        ("Importance sampling (brief)",     INK2,   (0, (5, 1.5, 1, 1.5)), "D"),
    "tilt":      ("Importance sampling, tuned tilt", INK,    "-",            "v"),
    "cond_tilt": ("Tuned tilt, rain-conditioned",    INK,    (0, (3, 1.5)),  "x"),
}


def setup():
    family = "Calibri" if any(f.name == "Calibri" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 8.5,
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "axes.labelsize": 8.5, "axes.titlesize": 8.5, "axes.labelcolor": INK, "axes.edgecolor": INK2,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
        "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 2.6,
        "lines.linewidth": 1.1, "lines.markersize": 3.2,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    plt.close(fig)
    print(f"wrote docs/figures/{name}.pdf and .png")


def log_axes(ax, x=True, y=True):
    for on, axis in ((x, ax.xaxis), (y, ax.yaxis)):
        if on:
            axis.set_major_locator(LogLocator(base=10, numticks=12))
            axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
            axis.set_minor_formatter(NullFormatter())
    ax.grid(False, which="minor")


def classical_line(ax, x, y, m, n_markers=5):
    label, c, ls, mk = CLASSICAL[m]
    x, y = np.asarray(x, float), np.asarray(y, float)
    every = max(1, len(x) // n_markers)
    ax.plot(x, y, color=c, ls=ls, lw=1.0 if m != "tilt" else 1.2, marker=mk, markevery=(every // 2, every),
            ms=3.2, mfc="white" if mk not in (None, "x") else c, mec=c, mew=0.8, label=label)


def target_line(ax, y, text=None):
    ax.axhline(y, color=RULE, lw=0.7, ls="-", zorder=1)
    if text:
        ax.annotate(text, (1, y), xycoords=("axes fraction", "data"), xytext=(-2, 2), textcoords="offset points",
                    ha="right", va="bottom", fontsize=7, color=MUTED)


def panel_title(ax, letter, text):
    ax.set_title(rf"$\mathbf{{({letter})}}$  {text}", loc="left", color=INK, pad=4)


def pct(v):
    return f"{v * 100:.3g}%"


def usd(v):
    return f"USD {v / 1e6:.3g}m"


def sci(v):
    m, e = f"{v:.2e}".split("e")
    return rf"${m} \times 10^{{{int(e)}}}$"


# ---------------------------------------------------------------------------
# Figure: accuracy per query, two-country model
# ---------------------------------------------------------------------------
TARGETS = (
    ("term_expected_principal_loss_usd", "Expected loss E[L]", usd),
    ("term_joint_attachment_probability", "P(both bonds pay)", pct),
    ("term_portfolio_loss_exceedance_probability", "P(L > USD 80m)", pct),
    ("term_portfolio_stop_loss_usd", "Stop-loss above USD 80m", usd),
    ("term_both_bonds_exhausted_probability", "P(both bonds exhausted)", sci),
)


def accuracy_per_query():
    d = json.loads((RES / "accuracy_per_query.json").read_text())
    base = d["scenarios"]["adb-base"]
    fig, grid = plt.subplots(2, 3, figsize=(TEXT_W, 4.9), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.12})
    axes, spare = list(grid.flat[:5]), grid.flat[5]
    for i, (ax, (k, name, fmt)) in enumerate(zip(axes, TARGETS)):
        r = base[k]
        for m in CLASSICAL:
            if m in r:
                classical_line(ax, d["sobol_budgets"] if m == "sobol" else d["budgets"], r[m], m)
        ax.plot(r["qae"]["queries"], r["qae"]["rel_rmse"], color=BLUE, lw=1.6, marker="o", ms=2.8, label="IQAE", zorder=4)
        if "qae_cond" in r:
            ax.plot(r["qae_cond"]["queries"], r["qae_cond"]["rel_rmse"], color=BLUE, lw=1.6, ls=(0, (3, 1.5)),
                    marker="o", ms=2.8, mfc="white", label="IQAE, rain-conditioned", zorder=4)
        target_line(ax, 0.01, "1% error" if i == 2 else None)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e2, 2e7); ax.set_ylim(1e-4, 10)
        log_axes(ax)
        panel_title(ax, "abcde"[i], f"{name}\nexact: {fmt(r['exact'])}")
    for ax in grid[:, 0]:
        ax.set_ylabel("Relative RMSE")
    for ax in grid[1, :2]:
        ax.set_xlabel("Queries")
    grid[0, 2].set_xlabel("Queries")
    grid[0, 2].xaxis.set_tick_params(labelbottom=True)
    spare.axis("off")
    h, l = axes[-1].get_legend_handles_labels()
    order = [l.index(x) for x in ("IQAE", "IQAE, rain-conditioned")] + [i for i, x in enumerate(l) if not x.startswith("IQAE")]
    spare.legend([h[i] for i in order], [l[i] for i in order], loc="center", fontsize=7.5, handlelength=3.0,
                 labelspacing=0.75, title="Estimator", title_fontsize=8)
    save(fig, "accuracy_per_query")


# ---------------------------------------------------------------------------
# Figure: CVaR accuracy per query
# ---------------------------------------------------------------------------
def cvar_accuracy(scenario="adb-base"):
    d = json.loads((RES / "risk_accuracy.json").read_text())
    acc = d["accuracy"][scenario]
    fig, axes = plt.subplots(1, len(acc), figsize=(TEXT_W, 2.95), sharey=True, gridspec_kw={"wspace": 0.1})
    for i, (ax, (a, r)) in enumerate(zip(np.atleast_1d(axes), acc.items())):
        for m in CLASSICAL:
            if m in r["classical"]:
                cm = r["classical"][m]
                classical_line(ax, cm["budgets"], cm["cvar_rel_rmse"], m)
        q = r["quantum"]
        ax.plot(q["q_certify"]["queries"], q["q_certify"]["cvar_rel_rmse"], color=BLUE, lw=1.6, marker="o", ms=2.8,
                label="IQAE: certified VaR bisection + CVaR", zorder=4)
        ax.plot(q["q_matched"]["queries"], q["q_matched"]["cvar_rel_rmse"], color=BLUE, lw=1.4, ls=(0, (3, 1.5)),
                marker="o", ms=2.8, mfc="white", label="IQAE: accuracy-matched bisection", zorder=4)
        ax.plot(q["q_varknown"]["queries"], q["q_varknown"]["cvar_rel_rmse"], color=BLUE, lw=1.0, ls=(0, (1, 1.2)),
                label="IQAE: CVaR stage only (VaR given)", zorder=4)
        target_line(ax, 0.01, "1% error" if i == len(acc) - 1 else None)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e2, 3e6); ax.set_ylim(1e-4, 1)
        log_axes(ax)
        ex = r["exact"]
        panel_title(ax, "abc"[i], rf"$\alpha$ = {a}%" + f"\nVaR {usd(ex['VaR'])}, CVaR {usd(ex['CVaR'])}")
        ax.set_xlabel("Queries")
    np.atleast_1d(axes)[0].set_ylabel("Relative RMSE of CVaR")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    order = [i for i, x in enumerate(l) if x.startswith("IQAE")] + [i for i, x in enumerate(l) if not x.startswith("IQAE")]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, fontsize=7.5, columnspacing=1.6, handlelength=3.0)
    save(fig, "cvar_accuracy_per_query")


# ---------------------------------------------------------------------------
# Figure: dependence sweep with interval transport
# ---------------------------------------------------------------------------
def dependence_sweep():
    d = json.loads((RES / "twin.json").read_text())
    ex = [s for s in d["sweeps"] if s["example"]][0]["example"]
    m = np.array([r["value"] for r in ex])
    lo, hi, exact = (np.array([r[k] for r in ex]) * 1e5 for k in ("lo", "hi", "exact"))
    fresh = np.array([r["mode"] == "recalibrate" for r in ex])
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.45))
    ax.fill_between(m, lo, hi, color=BLUE, alpha=0.16, lw=0, label="Certified 99% interval (digital model)")
    ax.plot(m, lo, color=BLUE, lw=0.5, alpha=0.6); ax.plot(m, hi, color=BLUE, lw=0.5, alpha=0.6)
    ax.plot(m, exact, color=INK, lw=1.2, label="Exact value")
    ax.plot(m[fresh], exact[fresh], ls="none", marker="o", ms=4, color=BLUE, mec="white", mew=0.6, zorder=4,
            label="New quantum estimate (recalibration)")
    ax.plot(m[~fresh], exact[~fresh], ls="none", marker="o", ms=3.4, mfc="white", mec=BLUE, mew=0.8, zorder=4,
            label="Interval transported, no new queries")
    ax.axvline(1.0, color=RULE, lw=0.7, zorder=1)
    ax.annotate("brief's model (m = 1)", (1.0, ax.get_ylim()[1]), xytext=(3, -9), textcoords="offset points",
                fontsize=7, color=MUTED)
    ax.set_xlabel("Shared-earthquake multiplier m")
    ax.set_ylabel("P(both exhausted)\nper 100,000 terms")
    ax.set_xlim(m.min() - 0.03, m.max() + 0.03)
    ax.legend(loc="upper left", fontsize=7.5, handlelength=2.0)
    save(fig, "dependence_sweep")


# ---------------------------------------------------------------------------
# Figure: grid scaling, best classical / IQAE
# ---------------------------------------------------------------------------
def grid_ratio():
    d = json.loads((RES / "grid_qae.json").read_text())
    n = np.array([g["n"] for g in d["grids"]])
    series = {"P(L>T3)": (r"P(L > $\mu + 3\sigma$)", BLUE, "-", "o"),
              "stop-loss": (r"Stop-loss above $\mu + 3\sigma$", AQUA, (0, (5, 1.6)), "^"),
              "P(L>T5)": (r"P(L > $\mu + 5\sigma$)", ORANGE, (0, (1.2, 1.2)), "s"),
              "E[L]": ("Expected loss E[L]", MUTED, (0, (5, 1.5, 1, 1.5)), "D")}
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.75))
    exact_n = max(g["n"] for g in d["grids"] if g["targets"]["E[L]"]["mu_source"] == "exact")
    ax.axvspan(np.sqrt(exact_n * n[n > exact_n].min()), n.max() * 1.6, color="#f1f0ec", lw=0, zorder=0)
    for k, (label, c, ls, mk) in series.items():
        y = np.array([g["targets"][k]["quantum_over_best"] for g in d["grids"]])
        ax.plot(n, y, color=c, ls=ls, lw=1.5 if k == "P(L>T3)" else 1.1, marker=mk, ms=3.6,
                mfc="white" if c != BLUE else c, mec=c, mew=0.8, label=label, zorder=3)
        ax.annotate(label, (n[-1], y[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7.5,
                    color=INK)
    ax.axhline(1, color=INK2, lw=0.7, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    log_axes(ax)
    ax.set_xlim(n.min() / 1.25, n.max() * 1.6)
    yl = ax.get_ylim()
    ax.annotate("above 1: IQAE needs fewer queries", (45, 1), xytext=(0, 3), textcoords="offset points",
                fontsize=7, color=MUTED, va="bottom")
    ax.annotate("below 1: classical sampler needs fewer", (45, 1), xytext=(0, -3), textcoords="offset points",
                fontsize=7, color=MUTED, va="top")
    ax.set_xticks(n, [f"{int(v)}" for v in n])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.annotate("no exact law: reference values estimated", (np.sqrt(exact_n * n[n > exact_n].min()) * 1.08, yl[1]),
                xytext=(0, -9), textcoords="offset points", fontsize=7, color=MUTED)
    ax.set_xlabel("Zones in the grid, n")
    ax.set_ylabel("Best classical queries / IQAE queries\n(1% relative RMSE)")
    fig.subplots_adjust(right=0.78)
    save(fig, "grid_qae_ratio")


if __name__ == "__main__":
    setup()
    accuracy_per_query()
    cvar_accuracy()
    dependence_sweep()
    grid_ratio()
