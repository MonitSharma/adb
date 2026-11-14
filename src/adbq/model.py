"""ADB two-country catastrophe-bond model, exactly as specified in the challenge brief.

Two equivalent routes to the exact term distribution are implemented:

* ``term_distribution_brief``: the brief's route -- 25-state CTMC per year, precipitation
  per year, then three capped convolutions over 100 x 100 annual state pairs.
* ``term_distribution_sources``: the same law written as a product of independent
  Poisson *sources* over the whole three-year horizon (own earthquakes per country,
  shared earthquakes, own and shared precipitation), followed by a capped sum.

The two agree to machine precision because every payout counter is a non-negative sum
capped at a fixed level, and min(c, x + y) = min(c, min(c, x) + min(c, y)). The source
form is what the quantum state-preparation circuit loads.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
from scipy.linalg import expm

# ---------------------------------------------------------------------------
# Inputs from the brief (Table 1, Eq. 4, step 5)
# ---------------------------------------------------------------------------
EXCEEDANCE = {
    "KGZ": (0.03883, 0.02331, 0.01790, 0.01184),
    "TJK": (0.03925, 0.01743, 0.01332, 0.01043),
}
COND_TRIGGER = {("KGZ", "TJK"): 0.075, ("TJK", "KGZ"): 0.074}  # P(other | one)
STEPS = 4            # earthquake payout in quarter steps, capped at 4
TERM_YEARS = 3
THRESHOLD_USD = 80e6


@dataclass(frozen=True)
class Scenario:
    name: str = "adb-base"
    quake_mult: float = 1.0          # m in Eq. (4)
    rain_shared_share: float = 0.25  # c_r = share * min(mu_1, mu_2)
    rain_mult: float = 1.0           # multiplies both precipitation rates
    rain_annual_prob: float = 0.02
    quake_limit: float = 70e6
    rain_limit: float = 10e6
    threshold: float = THRESHOLD_USD
    years: int = TERM_YEARS
    # Country calibration. None = the brief's (Table 1 and the conditional trigger rates).
    # country_types names the two calibrations that alternate along chains and grids;
    # exceedance gives each type's four annual exceedance probabilities; cond_trigger is
    # (P(B triggers | A triggers), P(A | B)) for types (A, B).
    country_types: tuple = ("KGZ", "TJK")
    exceedance: tuple | None = None
    cond_trigger: tuple | None = None


SCENARIOS = {
    "adb-base": Scenario(),
    "adb-independent": Scenario("adb-independent", quake_mult=0.0, rain_shared_share=0.0),
    "adb-common-shock": Scenario("adb-common-shock", quake_mult=2.0),
    "adb-rain-stress": Scenario("adb-rain-stress", rain_mult=2.0),
    "adb-rain-cover": Scenario("adb-rain-cover", quake_limit=60e6, rain_limit=20e6),
}

# Table 5 of the brief (three-year term); used by tests and verification scripts.
TABLE5 = {
    "adb-base": (10.06e6, 0.04414, 0.01205, 282_355, 0.000032),
    "adb-independent": (10.06e6, 0.02716, 0.00719, 171_266, 0.000004),
    "adb-common-shock": (10.06e6, 0.05053, 0.01607, 385_043, 0.000044),
    "adb-rain-stress": (11.17e6, 0.07046, 0.01322, 306_286, 0.000069),
    "adb-rain-cover": (9.97e6, 0.04414, 0.00700, 170_927, 0.000032),
}
TARGETS = (
    "term_expected_principal_loss_usd",
    "term_joint_attachment_probability",
    "term_portfolio_loss_exceedance_probability",
    "term_portfolio_stop_loss_usd",
    "term_both_bonds_exhausted_probability",
)
SHORT = ("E[L]", "P(both pay)", "P(L>80m)", "stop-loss", "P(both exhausted)")


def country_tables(s: "Scenario"):
    """((name_A, table_A), (name_B, table_B), (P(B | A), P(A | B))) for a scenario."""
    a, b = s.country_types
    ex = s.exceedance if s.exceedance is not None else (EXCEEDANCE[a], EXCEEDANCE[b])
    ct = s.cond_trigger if s.cond_trigger is not None else (COND_TRIGGER[(a, b)], COND_TRIGGER[(b, a)])
    return (a, tuple(ex[0])), (b, tuple(ex[1])), tuple(ct)


# ---------------------------------------------------------------------------
# Compound-Poisson calibration (Eqs. 2-3)
# ---------------------------------------------------------------------------
def poisson_rates(t: tuple[float, float, float, float]) -> tuple[float, np.ndarray]:
    """Total rate Lambda and step rates lambda_1..4 from the four exceedance probabilities."""
    p0, p1, p2, p3 = 1 - t[0], t[0] - t[1], t[1] - t[2], t[2] - t[3]
    lam_tot = -np.log(p0)
    l1 = p1 / p0
    l2 = p2 / p0 - l1**2 / 2
    l3 = p3 / p0 - l1 * l2 - l1**3 / 6
    l4 = lam_tot - l1 - l2 - l3
    return lam_tot, np.array([l1, l2, l3, l4])


@dataclass(frozen=True)
class TwoCountryParams:
    lam: tuple[float, float]             # total quake rates Lambda_1, Lambda_2
    w: tuple[np.ndarray, np.ndarray]     # size distributions w_{i,k}
    c: float                             # shared quake rate
    mu: tuple[float, float]              # precipitation rates
    cr: float                            # shared precipitation rate


def two_country_params(s: Scenario) -> TwoCountryParams:
    (_, ta), (_, tb), (p_ba, p_ab) = country_tables(s)
    L1, l1 = poisson_rates(ta)
    L2, l2 = poisson_rates(tb)
    c = s.quake_mult * 0.5 * (p_ba * L1 + p_ab * L2)
    mu = -np.log(1 - s.rain_annual_prob) * s.rain_mult
    cr = s.rain_shared_share * mu
    return TwoCountryParams((L1, L2), (l1 / L1, l2 / L2), c, (mu, mu), cr)


# ---------------------------------------------------------------------------
# Route 1: the brief's annual CTMC + capped convolution over years
# ---------------------------------------------------------------------------
def quake_generator(p: TwoCountryParams) -> np.ndarray:
    (L1, L2), (w1, w2), c = p.lam, p.w, p.c
    n = STEPS + 1
    G = np.zeros((n * n, n * n))
    idx = lambda a, b: a * n + b
    for a in range(n):
        for b in range(n):
            s = idx(a, b)
            for k in range(1, STEPS + 1):
                G[s, idx(min(a + k, STEPS), b)] += (L1 - c) * w1[k - 1]
                G[s, idx(a, min(b + k, STEPS))] += (L2 - c) * w2[k - 1]
                for j in range(1, STEPS + 1):
                    G[s, idx(min(a + k, STEPS), min(b + j, STEPS))] += c * w1[k - 1] * w2[j - 1]
            G[s, s] -= G[s].sum()
    return G


def rain_joint(mu1: float, mu2: float, cr: float, t: float = 1.0) -> np.ndarray:
    """2x2 law of (r1, r2) for Poisson streams over horizon t (Eqs. 5-6)."""
    P00 = np.exp(-(mu1 + mu2 - cr) * t)
    P10 = np.exp(-mu2 * t) - P00
    P01 = np.exp(-mu1 * t) - P00
    return np.array([[P00, P01], [P10, 1 - P00 - P10 - P01]])


def annual_distribution(s: Scenario) -> np.ndarray:
    """Annual law over (a, b, r1, r2), shape (5, 5, 2, 2)."""
    p = two_country_params(s)
    peq = expm(quake_generator(p))[0].reshape(STEPS + 1, STEPS + 1)
    pr = rain_joint(*p.mu, p.cr)
    return np.einsum("ab,rs->abrs", peq, pr)


CAP = np.array([STEPS, STEPS, 1, 1])


def term_distribution_brief(s: Scenario) -> np.ndarray:
    ann = annual_distribution(s)
    term = np.zeros_like(ann)
    term[0, 0, 0, 0] = 1.0
    states = np.argwhere(np.ones_like(ann, dtype=bool))
    for _ in range(s.years):
        new = np.zeros_like(term)
        for st in states:
            pt = term[tuple(st)]
            if pt == 0:
                continue
            for u in states:
                new[tuple(np.minimum(st + u, CAP))] += pt * ann[tuple(u)]
        term = new
    return term


# ---------------------------------------------------------------------------
# Route 2: independent Poisson sources over the whole horizon
# ---------------------------------------------------------------------------
def capped_compound_poisson(rates: np.ndarray, cap: int = STEPS) -> np.ndarray:
    """Law of min(cap, S) for S compound Poisson with jump-size rates rates[k-1] (Panjer)."""
    p = np.zeros(cap + 1)
    p[0] = np.exp(-rates.sum())
    for s_ in range(1, cap):
        p[s_] = sum(k * rates[k - 1] * p[s_ - k] for k in range(1, min(s_, len(rates)) + 1)) / s_
    p[cap] = 1.0 - p[:cap].sum()
    return p


def capped_shared_pair(rate: float, w1: np.ndarray, w2: np.ndarray, cap: int = STEPS,
                       nmax: int = 40) -> np.ndarray:
    """Joint law of capped contributions (x, y) of one shared Poisson stream with
    independent sizes w1 and w2 per event."""
    jump = np.zeros((cap + 1, cap + 1))
    for k in range(1, len(w1) + 1):
        for j in range(1, len(w2) + 1):
            jump[min(k, cap), min(j, cap)] += w1[k - 1] * w2[j - 1]
    cur = np.zeros_like(jump)
    cur[0, 0] = 1.0
    out = np.exp(-rate) * cur
    weight = np.exp(-rate)
    for n in range(1, nmax + 1):
        nxt = np.zeros_like(cur)
        for (a, b), pa in np.ndenumerate(cur):
            if pa == 0:
                continue
            for (k, j), pj in np.ndenumerate(jump):
                if pj:
                    nxt[min(a + k, cap), min(b + j, cap)] += pa * pj
        cur = nxt
        weight *= rate / n
        out += weight * cur
    return out / out.sum()


@dataclass(frozen=True)
class Sources:
    """Independent source laws over the term (all horizon-integrated)."""
    own_quake: tuple[np.ndarray, np.ndarray]   # each length 5
    shared_quake: np.ndarray                   # 5 x 5
    own_rain: tuple[float, float]              # P(trigger)
    shared_rain: float


def two_country_sources(s: Scenario, rate_scale: tuple[float, ...] | None = None) -> Sources:
    """Source laws for the two-country model. ``rate_scale`` = optional multipliers on the
    five source rates (own quake 1, own quake 2, shared quake, own rain (both), shared rain)
    used for exponential tilting of the loading step."""
    p = two_country_params(s)
    H = s.years
    sc = rate_scale or (1, 1, 1, 1, 1)
    o1 = capped_compound_poisson((p.lam[0] - p.c) * p.w[0] * H * sc[0])
    o2 = capped_compound_poisson((p.lam[1] - p.c) * p.w[1] * H * sc[1])
    sh = capped_shared_pair(p.c * H * sc[2], p.w[0], p.w[1])
    r1 = 1 - np.exp(-(p.mu[0] - p.cr) * H * sc[3])
    r2 = 1 - np.exp(-(p.mu[1] - p.cr) * H * sc[3])
    rs = 1 - np.exp(-p.cr * H * sc[4])
    return Sources((o1, o2), sh, (r1, r2), rs)


def combine_sources(src: Sources) -> np.ndarray:
    term = np.zeros((STEPS + 1, STEPS + 1, 2, 2))
    o1, o2 = src.own_quake
    for x1, p1 in enumerate(o1):
        for x2, p2 in enumerate(o2):
            for (y1, y2), ps in np.ndenumerate(src.shared_quake):
                pq = p1 * p2 * ps
                if pq == 0:
                    continue
                a, b = min(STEPS, x1 + y1), min(STEPS, x2 + y2)
                for u1 in (0, 1):
                    for u2 in (0, 1):
                        for z in (0, 1):
                            pr = ((src.own_rain[0] if u1 else 1 - src.own_rain[0])
                                  * (src.own_rain[1] if u2 else 1 - src.own_rain[1])
                                  * (src.shared_rain if z else 1 - src.shared_rain))
                            term[a, b, u1 | z, u2 | z] += pq * pr
    return term


def term_distribution_sources(s: Scenario) -> np.ndarray:
    return combine_sources(two_country_sources(s))


# ---------------------------------------------------------------------------
# Losses and targets
# ---------------------------------------------------------------------------
def loss_arrays(s: Scenario):
    a, b, r1, r2 = np.indices((STEPS + 1, STEPS + 1, 2, 2))
    L1 = s.quake_limit * a / STEPS + s.rain_limit * r1
    L2 = s.quake_limit * b / STEPS + s.rain_limit * r2
    return L1, L2


def payoff_arrays(s: Scenario) -> dict[str, np.ndarray]:
    """f(x) for each target on the (5,5,2,2) term grid."""
    L1, L2 = loss_arrays(s)
    L = L1 + L2
    full = s.quake_limit + s.rain_limit
    return {
        TARGETS[0]: L,
        TARGETS[1]: ((L1 > 0) & (L2 > 0)).astype(float),
        TARGETS[2]: (L > s.threshold).astype(float),
        TARGETS[3]: np.maximum(L - s.threshold, 0.0),
        TARGETS[4]: ((L1 == full) & (L2 == full)).astype(float),
    }


def targets(term: np.ndarray, s: Scenario) -> dict[str, float]:
    return {k: float((term * f).sum()) for k, f in payoff_arrays(s).items()}


@lru_cache(maxsize=None)
def exact(name: str) -> dict[str, float]:
    s = SCENARIOS[name]
    return targets(term_distribution_sources(s), s)


# ---------------------------------------------------------------------------
# Tail risk measures on the term loss distribution
# ---------------------------------------------------------------------------
def loss_distribution(s: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Distinct portfolio loss levels (ascending) and their term probabilities."""
    term = term_distribution_sources(s)
    L1, L2 = loss_arrays(s)
    L, p = (L1 + L2).ravel(), term.ravel()
    levels = np.unique(L)
    probs = np.array([p[L == v].sum() for v in levels])
    return levels, probs


def var_cvar(levels: np.ndarray, probs: np.ndarray, alpha: float) -> tuple[float, float]:
    """VaR_alpha = smallest level with CDF >= alpha; CVaR by Rockafellar-Uryasev:
    CVaR = VaR + E[(L - VaR)+] / (1 - alpha). Exact for distributions with atoms."""
    cdf = np.cumsum(probs)
    var = float(levels[np.searchsorted(cdf, alpha - 1e-15)])
    cvar = var + float((probs * np.maximum(levels - var, 0.0)).sum()) / (1 - alpha)
    return var, cvar
