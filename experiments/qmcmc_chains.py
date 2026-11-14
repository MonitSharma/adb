"""QeMCMC on the 4 x 4 grid (16 bits): sampled chains at equal loss-evaluation budget.

Worlds: calibrated (kappa = 1) and high contagion (kappa = 4). Ground truth: exact CVaR95
by enumeration of all 2^16 states. Every sampler targets the same loss-tilted law (lam =
the tilt that is best for i.i.d. sampling, shared by all samplers) and estimates CVaR95
under the true law with exact importance weights. Budget = loss evaluations = proposed
states (parallel tempering: one per replica per sweep). Chains start in the all-zero
state; BURN steps are discarded and not counted.

Samplers: Local MH; Local + random jump (classical non-local control); Local + pair
flips; Local + plaquette flips; Local + Wolff clusters; parallel tempering (local + pair
moves, 4 replicas); Local + quantum (short); Pair flips + quantum (short); Local +
quantum (default) [optional]; Local + mismatched quantum (control).

Quantum proposals are exact draws from the emulated Layden proposal (no hardware, no
noise); the claim level is noiseless emulation.

Writes results/qmcmc_chains.json. Run: python experiments/qmcmc_chains.py [--quick] [--default]   (--quick: 3 x 4 grid, few chains)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adbq.grid import calibrated_grid                                       # noqa: E402
from adbq.qmcmc import (QuantumProposal, Target, best_iid_lambda,          # noqa: E402
                        mismatched_landscape, quantum_params, run_chains, run_tempering)

WORLDS = {"calibrated (kappa = 1)": 1.0, "high contagion (kappa = 4)": 4.0}
SEED = 20261115
PT_BETAS = (1.0, 0.7, 0.45, 0.3)


def log(msg):
    print(msg, flush=True)


def prefilled(E_h, params, tgt, R, total_steps, rng, mass=0.999, steps=4):
    """Quantum proposal with pools precomputed for the states holding ``mass`` of pi_lam;
    pool length ~ 1.5 x the expected number of quantum proposals from each state."""
    qp = QuantumProposal(E_h, params, steps=steps, dtype=np.complex64)
    order = np.argsort(tgt.pi_lam)[::-1]
    k = int(np.searchsorted(np.cumsum(tgt.pi_lam[order]), mass)) + 1
    st = order[:k]
    sizes = np.clip(np.ceil(1.5 * R * total_steps * tgt.pi_lam[st]) + 32, 64, 200_000)
    t0 = time.time()
    qp.prefill(st, sizes, rng, log=log)
    log(f"    prefilled {k} states in {time.time() - t0:.0f}s")
    return qp


def errors(tgt, chains, budgets):
    return {B: np.array([tgt.cvar_from_states(c[:B]) / tgt.cvar - 1 for c in chains]) for B in budgets}


def errors_pt(tgt, cold, budgets, K):
    return {B: np.array([tgt.cvar_from_states(c[:B // K]) / tgt.cvar - 1 for c in cold]) for B in budgets}


def summarise(errs, rng):
    out = {}
    for B, e in errs.items():
        boot = [np.sqrt(np.mean(rng.choice(e, len(e)) ** 2)) for _ in range(1000)]
        out[str(B)] = {"rel_rmse": float(np.sqrt(np.mean(e**2))), "bias": float(e.mean()),
                       "rmse_90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))],
                       "errors": e.tolist()}
    return out


def run_world(kappa, budgets, R, burn, with_default, rng, mass=0.999, shape=(4, 4)):
    m = calibrated_grid(*shape, kappa=kappa)
    lam = best_iid_lambda(m)
    tgt = Target(m, lam)
    total = burn + max(budgets)
    starts = np.zeros(R, np.int64)
    log(f"  lam = {lam:g}, CVaR95 = {tgt.cvar / 1e6:.1f}m")
    res = {"lambda": lam, "VaR95": tgt.var, "CVaR95": tgt.cvar,
           "iid_steps_10pct": tgt.iid_variance() / (0.1 * tgt.cvar) ** 2, "samplers": {}}
    q_short = prefilled(tgt.E_lam, quantum_params("short", rng, 4), tgt, R, total, rng, mass)
    q_mis = prefilled(mismatched_landscape(m, lam, rng), quantum_params("short", rng, 4), tgt, R, total, rng, mass)
    runs = {
        "Local MH": [(1.0, "local")],
        "Local + random jump": [(0.5, "local"), (0.5, "uniform")],
        "Local + pair flips": [(0.5, "local"), (0.5, "pair")],
        "Local + plaquette flips": [(0.5, "local"), (0.5, "plaquette")],
        "Local + Wolff clusters": [(0.5, "local"), (0.5, "wolff")],
        "Local + quantum (short)": [(0.5, "local"), (0.5, q_short)],
        "Pair flips + quantum (short)": [(0.5, "pair"), (0.5, q_short)],
        "Local + mismatched quantum": [(0.5, "local"), (0.5, q_mis)],
    }
    if with_default:
        runs["Local + quantum (default)"] = [(0.5, "local"),
                                             (0.5, prefilled(tgt.E_lam, quantum_params("default", rng, 4),
                                                             tgt, R, total, rng, mass))]
    for name, moves in runs.items():
        t0 = time.time()
        S, acc = run_chains(tgt, moves, starts, total, rng)
        res["samplers"][name] = {"acceptance": float(acc.mean()),
                                 **summarise(errors(tgt, S[:, burn:], budgets), rng)}
        log(f"  {name}: {time.time() - t0:.0f}s, rmse@{max(budgets)} = "
            f"{res['samplers'][name][str(max(budgets))]['rel_rmse']:.3%}")
    K = len(PT_BETAS)
    t0 = time.time()
    cold = run_tempering(tgt, [(0.5, "local"), (0.5, "pair")], PT_BETAS, starts, (total + K - 1) // K, rng)
    res["samplers"]["Parallel tempering (local + pair)"] = summarise(
        errors_pt(tgt, cold[:, burn // K:], budgets, K), rng)
    log(f"  PT: {time.time() - t0:.0f}s")
    res["quantum_rows_computed"] = {"short": q_short.rows_computed, "mismatch": q_mis.rows_computed}
    return res


def verdict(world, budgets, rng):
    """MSE ratio (= evaluation ratio) of the best classical vs the best quantum sampler at the
    largest budget, with a 90% bootstrap interval. Above 1: quantum needs fewer evaluations."""
    B = str(max(budgets))
    s = world["samplers"]
    cl = [k for k in s if "quantum" not in k]
    qu = [k for k in s if "quantum" in k and "mismatched" not in k]
    best_c = min(cl, key=lambda k: s[k][B]["rel_rmse"])
    best_q = min(qu, key=lambda k: s[k][B]["rel_rmse"])
    ec, eq = np.array(s[best_c][B]["errors"]), np.array(s[best_q][B]["errors"])
    boot = [np.mean(rng.choice(ec, len(ec)) ** 2) / np.mean(rng.choice(eq, len(eq)) ** 2) for _ in range(2000)]
    return {"budget": int(B), "best_classical": best_c, "best_quantum": best_q,
            "evaluation_ratio": float(np.mean(ec**2) / np.mean(eq**2)),
            "ratio_90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))]}


def render(d):
    L = [f"# QeMCMC on the {d['shape'][0]} x {d['shape'][1]} grid (sampled chains, noiseless emulation)", "",
         f"Generated by `experiments/qmcmc_chains.py`. {d['chains']} chains per sampler, burn-in "
         f"{d['burn']} (not counted). Relative RMSE of CVaR95 against the exact value; budget = loss "
         "evaluations.", ""]
    for w, r in d["worlds"].items():
        bs = d["budgets"]
        L += [f"## {w}: CVaR95 = USD {r['CVaR95'] / 1e6:.1f}m, tilt lambda = {r['lambda']:g}", "",
              "| sampler | " + " | ".join(f"RMSE @ {b:,}" for b in bs) + " | 90% interval @ max |",
              "|---|" + "---:|" * (len(bs) + 1)]
        for k, v in sorted(r["samplers"].items(), key=lambda kv: kv[1][str(bs[-1])]["rel_rmse"]):
            L.append(f"| {k} | " + " | ".join(f"{v[str(b)]['rel_rmse']:.2%}" for b in bs)
                     + f" | {v[str(bs[-1])]['rmse_90'][0]:.2%}-{v[str(bs[-1])]['rmse_90'][1]:.2%} |")
        vd = r["verdict"]
        L += ["", f"Best classical: {vd['best_classical']}; best quantum: {vd['best_quantum']}. "
              f"Evaluation ratio (classical MSE / quantum MSE) at {vd['budget']:,}: "
              f"**{vd['evaluation_ratio']:.3g}x**, i.e. quantum needs {1 / vd['evaluation_ratio']:.3g}x the evaluations "
              f"(90% interval {vd['ratio_90'][0]:.3g}-{vd['ratio_90'][1]:.3g}; "
              "above 1 = quantum needs fewer evaluations; an interval containing 1 is a tie).", ""]
    return "\n".join(L)


def main(quick, with_default):
    rng = np.random.default_rng(SEED)
    budgets = [250, 1000, 4000] if quick else [1000, 4000, 16000]
    R, burn = (40, 200) if quick else (200, 1000)
    out = {"budgets": budgets, "chains": R, "burn": burn, "seed": SEED, "pt_betas": PT_BETAS,
           "shape": (3, 4) if quick else (4, 4), "worlds": {}}
    for w, kappa in WORLDS.items():
        log(w)
        out["worlds"][w] = run_world(kappa, budgets, R, burn, with_default, rng, 0.999, (3, 4) if quick else (4, 4))
        out["worlds"][w]["verdict"] = verdict(out["worlds"][w], budgets, rng)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--default", action="store_true", help="also run the default-time quantum ensemble")
    a = ap.parse_args()
    d = main(a.quick, a.default)
    sfx = "_quick" if a.quick else ""
    (ROOT / "results" / f"qmcmc_chains{sfx}.json").write_text(json.dumps(d))
    (ROOT / "results" / f"qmcmc_chains{sfx}.md").write_text(render(d), encoding="utf-8")
    print(render(d))
