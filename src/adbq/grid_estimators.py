"""Classical estimators on the grid source model, and the IQAE comparison.

Targets (losses in lattice units, thresholds T_k = E[L] + k sd(L), exact moments):
  "E[L]"        f = L,                 f_max = L_max
  "P(L>T3)"     f = 1{L > T_3}         f_max = 1
  "P(L>T5)"     f = 1{L > T_5}         f_max = 1   (rare)
  "stop-loss"   f = (L - T_3)+         f_max = L_max - T_3

Classical samplers (one query = one draw of every source):
  mc     plain Monte Carlo;
  rqmc   randomized QMC: scrambled Sobol over all source uniforms, 8 independent scrambles;
  tilt   exponential tilting: each source's law multiplied by exp(theta x its loss units),
         theta tuned on a grid by pilot variance (pilot cost counted);
  ce     cross-entropy IS (Rubinstein and Kroese) in the exponential family with one tilt
         per source kind, multilevel, adaptation cost counted. (A free categorical law per
         source, family='per_source', degenerates on large grids; reported as a diagnostic.)
IS variance per query is estimated from a fresh pilot sample of the final proposal.

Quantum: IQAE (alpha = 0.01) emulated at the amplitude a = mu / f_max, as in
experiments/accuracy_per_query.py; when mu is not exact (grids beyond the exact sweep)
the amplitude comes from the reference estimate and is labelled as estimated.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from .gridsources import (KEYS, GridPortfolio, draw, exact_mean_var, log_weight, losses,
                          source_laws, threshold)
from .model import STEPS

TARGETS = ("E[L]", "P(L>T3)", "P(L>T5)", "stop-loss")


def target_functions(gp: GridPortfolio) -> dict:
    """{name: (f(L_units) -> values in USD or indicator, f_max)}."""
    u = gp.units[2]
    Lmax = gp.n * gp.full_units
    T3 = int(round(threshold(gp, 3) / u)); T5 = int(round(threshold(gp, 5) / u))
    return {"E[L]": (lambda L: L * u, Lmax * u),
            "P(L>T3)": (lambda L: (L > T3).astype(float), 1.0),
            "P(L>T5)": (lambda L: (L > T5).astype(float), 1.0),
            "stop-loss": (lambda L: np.maximum(L - T3, 0) * u, (Lmax - T3) * u),
            "_T": (T3 * u, T5 * u)}


def exact_targets(gp, values, pmf) -> dict:
    tf = target_functions(gp)
    Lu = np.rint(values / gp.units[2]).astype(int)
    return {k: float(pmf @ tf[k][0](Lu)) for k in TARGETS}


# ---------------------------------------------------------------------------
# Plain MC and randomized QMC
# ---------------------------------------------------------------------------
def mc_sample(gp, N, rng):
    return losses(gp, draw(gp, rng.random((N, gp.n_sources))))


def rqmc_estimate(gp, f, N, rng, scrambles: int = 8, chunk: int = 4096) -> float:
    """Mean of ``scrambles`` independent scrambled-Sobol estimates of 2^m points each
    (N = scrambles x 2^m); points are generated in chunks to bound memory."""
    m = int(np.log2(max(2, N // scrambles)))
    ests = []
    for _ in range(scrambles):
        eng = qmc.Sobol(d=gp.n_sources, scramble=True, seed=rng)
        s, left = 0.0, 2**m
        while left:
            k = min(chunk, left)
            s += f(losses(gp, draw(gp, eng.random(k)))).sum(); left -= k
        ests.append(s / 2**m)
    return float(np.mean(ests))


# ---------------------------------------------------------------------------
# Importance sampling: exponential tilt and cross-entropy
# ---------------------------------------------------------------------------
def _source_loss_units(gp) -> dict:
    """Loss units each source outcome would add if uncapped (used to shape the tilt)."""
    qu, ru, _ = gp.units
    k = np.arange((STEPS + 1) ** 2)
    xi, xj = np.divmod(k, STEPS + 1)
    return {"own_quake": np.arange(STEPS + 1) * qu, "own_rain": np.array([0, ru]),
            "shared_quake": (xi + xj) * qu, "shared_rain": np.array([0, 2 * ru])}


def tilted_laws(gp, theta: float) -> dict:
    p, lu = source_laws(gp), _source_loss_units(gp)
    out = {}
    for k in KEYS:
        q = p[k] * np.exp(theta * lu[k])[None, :]
        out[k] = q / q.sum(1, keepdims=True)
    return out


def is_sample(gp, laws, N, rng):
    X = draw(gp, rng.random((N, gp.n_sources)), laws)
    return losses(gp, X), np.exp(log_weight(gp, X, laws))


def per_query_variance(gp, f, laws, N, rng) -> tuple[float, float]:
    """(mean, variance) of f(L) w(X) under ``laws`` from N fresh draws."""
    L, w = is_sample(gp, laws, N, rng)
    y = f(L) * w
    return float(y.mean()), float(y.var())


def tune_tilt(gp, f, rng, thetas=None, pilot: int = 20_000):
    """Pick theta by pilot relative variance. The grid scales as 1/sqrt(n): a k-sigma event
    needs a mean shift ~ sqrt(n) while the tilt moves the mean ~ n theta.
    Returns (laws, theta, pilot cost in queries)."""
    if thetas is None:
        thetas = np.linspace(0.0, 0.3, 13) * np.sqrt(9 / gp.n)
    best, cost = None, 0
    for th in thetas:
        laws = tilted_laws(gp, th)
        m, v = chunked_is(gp, f, laws, pilot, rng)
        cost += pilot
        if m > 0 and m**2 > 0 and (best is None or v / m**2 < best[0]):
            best = (v / m**2, laws, float(th))
    if best is None:
        raise RuntimeError("no tilt on the grid produced a nonzero pilot estimate")
    return best[1], best[2], cost


def kind_tilted_laws(gp, thetas: dict) -> dict:
    """Exponential tilt with one parameter per source kind (own quake, own rain, shared
    quake, shared rain): q_s(x) ∝ p_s(x) exp(theta_kind * loss units of x)."""
    p, lu = source_laws(gp), _source_loss_units(gp)
    out = {}
    for k in KEYS:
        q = p[k] * np.exp(thetas[k] * lu[k])[None, :]
        out[k] = q / q.sum(1, keepdims=True)
    return out


def _kind_mean(gp, k, th) -> float:
    """E_q[total loss units from sources of kind k] under the kind tilt th."""
    p, lu = source_laws(gp)[k], _source_loss_units(gp)[k]
    q = p * np.exp(th * lu)[None, :]
    q /= q.sum(1, keepdims=True)
    return float((q @ lu).sum())


def cross_entropy(gp, f, rng, kind: str, T_units: int | None = None, N: int = 20_000,
                  rho: float = 0.1, iters: int = 12, family: str = "kind_tilt"):
    """Multilevel cross-entropy importance sampling (Rubinstein and Kroese).

    family 'kind_tilt' (default): the natural exponential family with one tilt parameter
    per source kind; the CE update is moment matching, E_theta[T_k] = weighted elite mean
    of T_k (T_k = loss units from kind-k sources), solved by bisection.
    family 'per_source': a free categorical law for every source (thousands of parameters
    on large grids). Kept as a diagnostic: it degenerates in high dimension.
    kind 'indicator': f = 1{L > T}, levels gamma_t = min(T, (1 - rho) quantile of L);
    kind 'value': weights f(L) w. Returns (laws, adaptation cost in queries)."""
    p = source_laws(gp)
    lu = _source_loss_units(gp)
    th = {k: 0.0 for k in KEYS}
    q = {k: v.copy() for k, v in p.items()}
    cost = 0
    for _ in range(iters):
        X = draw(gp, rng.random((N, gp.n_sources)), q)
        cost += N
        L = losses(gp, X)
        w = np.exp(log_weight(gp, X, q))
        gamma = None
        if kind == "indicator":
            gamma = min(T_units, np.quantile(L, 1 - rho))
            v = w * (L > T_units) if gamma >= T_units else w * (L >= gamma)
        else:
            v = w * f(L)
        if v.sum() <= 0:
            break
        if family == "kind_tilt":
            old_th = dict(th)
            for k in KEYS:
                target = float((v * lu[k][X[k]].sum(1)).sum() / v.sum())
                lo, hi = -2.0, 2.0
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if _kind_mean(gp, k, mid) < target else (lo, mid)
                th[k] = 0.5 * (lo + hi)
            q = kind_tilted_laws(gp, th)
            if kind == "value" and max(abs(th[k] - old_th[k]) for k in KEYS) < 2e-3:
                break                                    # converged
        else:
            new = {}
            for k in KEYS:
                idx = X[k]
                K = p[k].shape[1]
                cnt = np.zeros((idx.shape[1], K))
                for j in range(idx.shape[1]):
                    cnt[j] = np.bincount(idx[:, j], weights=v, minlength=K)
                cnt /= cnt.sum(1, keepdims=True)
                new[k] = 0.7 * cnt + 0.3 * q[k]
                new[k] = np.where(p[k] > 0, np.maximum(new[k], 1e-12), 0.0)
                new[k] /= new[k].sum(1, keepdims=True)
            q = new
        if kind == "indicator" and gamma >= T_units:
            break
    return q, cost


# ---------------------------------------------------------------------------
# Queries to a target relative RMSE
# ---------------------------------------------------------------------------
def queries_iid(var_per_query: float, mu: float, rel: float, overhead: int = 0) -> float:
    return overhead + var_per_query / (rel * mu) ** 2


def rqmc_queries(gp, f, mu, rel, rng, budgets=(2**10, 2**12, 2**14), reps: int = 20) -> dict:
    """Empirical RMSE at a few budgets and a power-law extrapolation to ``rel``."""
    rm = []
    for N in budgets:
        e = [rqmc_estimate(gp, f, N, rng) - mu for _ in range(reps)]
        rm.append(float(np.sqrt(np.mean(np.square(e))) / mu))
    x, y = np.log(budgets), np.log(np.maximum(rm, 1e-300))
    slope, icpt = np.polyfit(x, y, 1)
    # clamp: never extrapolate worse than MC (-1/2) or better than O(1/N) (-1)
    slope = float(np.clip(slope, -1.0, -0.5)) if np.isfinite(slope) else -0.5
    icpt = float(np.mean(y - slope * x))                         # refit the level at that slope
    return {"budgets": list(budgets), "rel_rmse": rm, "slope": slope,
            "queries": float(np.exp((np.log(rel) - icpt) / slope))}


# ---------------------------------------------------------------------------
# Chunked helpers for large grids, and emulated IQAE
# ---------------------------------------------------------------------------
def chunked_is(gp, f, laws, N, rng, chunk: int = 10_000) -> tuple[float, float]:
    """(mean, per-query variance) of f(L) w over N draws under ``laws``, in chunks."""
    s = s2 = 0.0
    done = 0
    while done < N:
        k = min(chunk, N - done)
        L, w = is_sample(gp, laws, k, rng)
        y = f(L) * w
        s += y.sum(); s2 += (y**2).sum(); done += k
    m = s / N
    return float(m), float(s2 / N - m**2)


def iqae_queries(a: float, rel: float, rng, reps: int = 200, alpha: float = 0.01) -> dict:
    """Queries for IQAE to reach relative RMSE ``rel`` on a, by emulation at amplitude a
    (Bernoulli outcomes of Q^k A|0>; valid for any A, Brassard et al. 2002)."""
    from .qae import iqae
    eps = np.geomspace(0.3, rel / 4, 8)
    qs, es = [], []
    for e in eps:
        runs = [iqae(a, e * a, rng, alpha=alpha) for _ in range(reps)]
        qs.append(np.mean([r[1] for r in runs]))
        es.append(np.sqrt(np.mean([(r[0] - a) ** 2 for r in runs])) / a)
    q, e = np.log(qs), np.log(np.minimum.accumulate(es))
    i = int(np.argmax(e <= np.log(rel)))
    if e[i] > np.log(rel):
        raise RuntimeError("epsilon grid too coarse")
    t = (np.log(rel) - e[i - 1]) / (e[i] - e[i - 1]) if i > 0 else 0.0
    return {"queries": float(np.exp(q[i - 1] + t * (q[i] - q[i - 1]))) if i > 0 else float(qs[0]),
            "eps_rel": eps.tolist(), "curve_queries": list(map(float, qs)), "curve_rel_rmse": list(map(float, es))}
