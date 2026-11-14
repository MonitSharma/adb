"""Exact loss distributions and the risk measures VaR / CVaR.

Conventions (identical to logistics_under_disruption/src/supplychain_qmcmc/risk.py):

* VaR_alpha  = inf{ l : F(l) >= alpha }   (lower alpha-quantile of the loss L);
* CVaR_alpha = min_v { v + E[(L - v)+] / (1 - alpha) }   (Rockafellar and Uryasev 2002).

For a discrete law the Rockafellar-Uryasev minimum is attained at v = VaR_alpha, so
CVaR_alpha = VaR_alpha + E[(L - VaR_alpha)+] / (1 - alpha). This is the identity the
quantum estimator uses; ``tests/test_risk.py`` checks it against the brute-force minimum.

Losses live on an integer lattice: every payout is a multiple of the unit
u = gcd(quake step, rain limit) (USD 2.5m in the brief), so the loss distribution is an
exact probability vector over 0, u, 2u, ..., and quantiles are exact atoms.
"""
from __future__ import annotations

import numpy as np

from .model import STEPS, Scenario, term_distribution_sources
from .portfolio import ChainPortfolio, _units

ALPHAS = (0.95, 0.99, 0.995)


# ---------------------------------------------------------------------------
# Risk measures of a discrete law
# ---------------------------------------------------------------------------
def var_cvar(losses, probs, alpha: float = 0.95, tol: float = 1e-12) -> tuple[float, float]:
    """(VaR_alpha, CVaR_alpha) of a discrete law (losses need not be sorted or unique).

    ``probs`` may be an exact PMF or nonnegative sample weights; they are normalised.
    ``tol`` guards the comparison F(l) >= alpha against round-off in a computed CDF.
    """
    losses = np.asarray(losses, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if not 0 < alpha < 1 or (probs < 0).any() or probs.sum() <= 0:
        raise ValueError("need 0 < alpha < 1 and nonnegative weights with positive sum")
    probs = probs / probs.sum()
    order = np.argsort(losses, kind="stable")
    l, p = losses[order], probs[order]
    cdf = np.cumsum(p)
    var = float(l[np.flatnonzero(cdf >= alpha - tol)[0]])
    cvar = var + float(np.dot(p, np.maximum(l - var, 0.0))) / (1 - alpha)
    return var, cvar


def ru_cvar_bruteforce(losses, probs, alpha: float) -> float:
    """Rockafellar-Uryasev minimum over all support points (reference implementation,
    as in the logistics project's risk.py)."""
    losses = np.asarray(losses, float)
    probs = np.asarray(probs, float) / np.sum(probs)
    support = np.unique(losses[probs > 0])
    ru = support + (np.maximum(losses[None, :] - support[:, None], 0) @ probs) / (1 - alpha)
    return float(ru.min())


def risk_summary(values, pmf, alphas=ALPHAS) -> dict[str, float]:
    out = {"E[L]": float(np.dot(values, pmf))}
    for a in alphas:
        v, c = var_cvar(values, pmf, a)
        out[f"VaR{_tag(a)}"] = v
        out[f"CVaR{_tag(a)}"] = c
    return out


def _tag(alpha: float) -> str:
    """0.95 -> '95', 0.99 -> '99', 0.995 -> '99.5'."""
    return f"{100 * alpha:g}"


def cdf_gap(values, pmf, alpha: float) -> float:
    """min_l |F(l) - alpha| over the support: how sharply alpha separates two atoms.
    A small gap makes VaR hard to identify by any sampling or estimation method."""
    order = np.argsort(values)
    cdf = np.cumsum(np.asarray(pmf)[order])
    return float(np.min(np.abs(cdf - alpha)))


# ---------------------------------------------------------------------------
# Two-country model (the brief): loss PMF from the exact term law
# ---------------------------------------------------------------------------
def loss_units_two_country(s: Scenario) -> tuple[np.ndarray, float]:
    """Integer loss (in units u) on the (5, 5, 2, 2) term grid, and u in USD."""
    qu, ru, _, u = _units(s)
    a, b, r1, r2 = np.indices((STEPS + 1, STEPS + 1, 2, 2))
    return qu * (a + b) + ru * (r1 + r2), float(u)


def loss_pmf_two_country(s: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """(loss values in USD, probabilities) on the full lattice 0, u, ..., L_max."""
    term = term_distribution_sources(s)
    Lu, u = loss_units_two_country(s)
    pmf = np.bincount(Lu.reshape(-1), weights=term.reshape(-1))
    return u * np.arange(len(pmf)), pmf


# ---------------------------------------------------------------------------
# n-country chain: transfer-matrix contraction of the full loss distribution
# ---------------------------------------------------------------------------
def loss_pmf_chain(pf: ChainPortfolio) -> tuple[np.ndarray, np.ndarray]:
    """Exact law of the portfolio loss for the chain family, in O(n^2) time.

    Same left-to-right contraction as ``portfolio.exact_chain`` but the carried statistic
    is the whole loss distribution (n * full_u + 1 bins) instead of a capped summary.
    Bond state = (quake steps the edge sends to the next country, edge rain bit).
    Each country's term loss li depends only on (a_i, rain_i), so outcomes are first
    aggregated by (next bond state, li) and each aggregate costs one shifted add.
    """
    from .model import capped_compound_poisson, capped_shared_pair

    s, H = pf.scen, pf.scen.years
    qu, ru, _, u = _units(s)
    full_u = STEPS * qu + ru
    nb = pf.n * full_u + 1
    P = np.zeros((STEPS + 1, 2, nb))
    P[0, 0, 0] = 1.0
    own_q, own_r = pf.own_quake_rate, pf.own_rain_rate
    for i in range(pf.n):
        po = capped_compound_poisson(own_q[i] * pf.w[i] * H)
        pr = 1 - np.exp(-own_r[i] * H)
        if i < pf.n - 1:
            pe = capped_shared_pair(pf.c[i] * H, pf.w[i], pf.w[i + 1])
            pz = 1 - np.exp(-pf.cr[i] * H)
        else:
            pe = np.zeros((STEPS + 1, STEPS + 1)); pe[0, 0] = 1.0
            pz = 0.0
        new = np.zeros_like(P)
        for xin, zin in np.ndindex(STEPS + 1, 2):
            block = P[xin, zin]
            if not block.any():
                continue
            # W[xr, z, li]: weight of sending (xr, z) right while country i loses li
            W = np.zeros((STEPS + 1, 2, full_u + 1))
            for o, p_o in enumerate(po):
                for (xl, xr), p_e in np.ndenumerate(pe):
                    w_q = p_o * p_e
                    if w_q == 0:
                        continue
                    a = min(STEPS, xin + o + xl)
                    for sb, p_s in ((0, 1 - pr), (1, pr)):
                        for z, p_z in ((0, 1 - pz), (1, pz)):
                            W[xr, z, a * qu + (sb | zin | z) * ru] += w_q * p_s * p_z
            for xr, z, li in zip(*np.nonzero(W)):
                new[xr, z, li:] += W[xr, z, li] * block[: nb - li]
        P = new
    pmf = P.sum(axis=(0, 1))
    return u * np.arange(nb), pmf
