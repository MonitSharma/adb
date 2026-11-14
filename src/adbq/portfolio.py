"""n-country generalisation of the ADB model and its exact transfer-matrix contraction.

Scaling family (our modelling choice; the brief defines only n = 2):
countries 1..n sit along a chain of shared fault lines and glacier systems. Each
neighbouring pair (i, i+1) has a shared earthquake stream and a shared precipitation
stream, calibrated exactly as the brief does for the Kyrgyz-Tajik pair. Country i keeps
its own total rates, so its marginal law is unchanged by the coupling. For n = 2 the
family reproduces the brief's model and Table 5 exactly.

Generalised targets (identical to the brief's five when n = 2):
  E[L]; P(at least two countries pay); P(L > T); E[max(L - T, 0)];
  P(at least two bonds exhausted), with T = USD 40m x n (= USD 80m at n = 2,
  i.e. half the portfolio principal).

Because the dependence graph is a chain, the joint law is a matrix product state whose
bond carries one edge's contribution to the next country (5 x 2 values) plus the running
statistics. Contracting it costs O(n) instead of the 10^n of naive enumeration.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import gcd

import numpy as np
from scipy.linalg import expm

from .model import (STEPS, Scenario, capped_compound_poisson, country_tables,
                    capped_shared_pair, poisson_rates)

GEN_TARGETS = ("E[L]", "P(>=2 pay)", "P(L>T)", "stop-loss", "P(>=2 exhausted)")


@dataclass(frozen=True)
class ChainPortfolio:
    n: int
    lam: np.ndarray        # (n,) total quake rates
    w: np.ndarray          # (n, 4) quake size distributions
    c: np.ndarray          # (n-1,) shared quake rates on edges
    mu: np.ndarray         # (n,) total precipitation rates
    cr: np.ndarray         # (n-1,) shared precipitation rates
    scen: Scenario

    @property
    def own_quake_rate(self) -> np.ndarray:
        out = self.lam.copy()
        out[:-1] -= self.c
        out[1:] -= self.c
        return out

    @property
    def own_rain_rate(self) -> np.ndarray:
        out = self.mu.copy()
        out[:-1] -= self.cr
        out[1:] -= self.cr
        return out


def chain_portfolio(n: int, s: Scenario = Scenario()) -> ChainPortfolio:
    (_, ta), (_, tb), (p_fwd, p_bwd) = country_tables(s)
    tables = [(ta, tb)[i % 2] for i in range(n)]
    lam, w = [], []
    for tab in tables:
        L, l = poisson_rates(tab)
        lam.append(L)
        w.append(l / L)
    lam, w = np.array(lam), np.array(w)
    # The brief: c = 0.5 (P(B | A) Lambda_A + P(A | B) Lambda_B), applied by country type.
    # Type A sits at even positions, so on edges (i, i+1) with i odd the type-A end is i+1.
    def edge_rate(i):
        a, b = (i, i + 1) if i % 2 == 0 else (i + 1, i)
        return s.quake_mult * 0.5 * (p_fwd * lam[a] + p_bwd * lam[b])
    c = np.array([edge_rate(i) for i in range(n - 1)])
    mu = np.full(n, -np.log(1 - s.rain_annual_prob) * s.rain_mult)
    cr = np.array([s.rain_shared_share * min(mu[i], mu[i + 1]) for i in range(n - 1)])
    s = replace(s, threshold=s.threshold * n / 2)   # T = USD 40m x n (80m at n = 2)
    pf = ChainPortfolio(n, lam, w, c, mu, cr, s)
    assert (pf.own_quake_rate > 0).all() and (pf.own_rain_rate >= 0).all()
    return pf


def _units(s: Scenario) -> tuple[int, int, int, int]:
    q = int(round(s.quake_limit / STEPS))
    r = int(round(s.rain_limit))
    u = gcd(q, r)
    T = int(round(s.threshold))
    assert T % u == 0
    return q // u, r // u, T // u, u


def _bump(arr: np.ndarray, axis: int) -> np.ndarray:
    """Increment a counter axis capped at 2 (0 -> 1, 1 -> 2, 2 -> 2)."""
    out = np.zeros_like(arr)
    idx = [slice(None)] * arr.ndim
    def at(k):
        j = list(idx); j[axis] = k; return tuple(j)
    out[at(1)] += arr[at(0)]
    out[at(2)] += arr[at(1)] + arr[at(2)]
    return out


def exact_chain(pf: ChainPortfolio) -> dict[str, float]:
    """Exact generalised targets by left-to-right contraction of the chain.

    Bond state = (edge contribution entering the next country: quake steps 0..4, rain 0/1).
    Carried statistics = portfolio loss in units u (capped at 'above T'), number of
    paying countries (capped at 2), number of exhausted bonds (capped at 2), and the
    expected excess over T for mass already above T.
    """
    s, H = pf.scen, pf.scen.years
    qu, ru, Tu, u = _units(s)
    full_u = STEPS * qu + ru
    K = Tu + 1                       # loss bins 0..Tu, and K = "above T"
    P = np.zeros((STEPS + 1, 2, K + 1, 3, 3))
    X = np.zeros((STEPS + 1, 2, 3, 3))  # E[(L - T) 1{L > T, state}] in units u
    P[0, 0, 0, 0, 0] = 1.0
    EL = 0.0
    own_q, own_r = pf.own_quake_rate, pf.own_rain_rate
    bins = np.arange(K + 1)
    for i in range(pf.n):
        po = capped_compound_poisson(own_q[i] * pf.w[i] * H)
        pr = 1 - np.exp(-own_r[i] * H)
        if i < pf.n - 1:
            pe = capped_shared_pair(pf.c[i] * H, pf.w[i], pf.w[i + 1])
            pz = 1 - np.exp(-pf.cr[i] * H)
        else:
            pe = np.zeros((STEPS + 1, STEPS + 1)); pe[0, 0] = 1.0
            pz = 0.0
        newP, newX = np.zeros_like(P), np.zeros_like(X)
        for (xin, zin) in np.ndindex(STEPS + 1, 2):
            block, xblk = P[xin, zin], X[xin, zin]
            if not block.any():
                continue
            mass = block.sum()
            for o, p_o in enumerate(po):
                for (xl, xr), p_e in np.ndenumerate(pe):
                    w_q = p_o * p_e
                    if w_q == 0:
                        continue
                    a = min(STEPS, xin + o + xl)
                    for sb, p_s in ((0, 1 - pr), (1, pr)):
                        for z, p_zz in ((0, 1 - pz), (1, pz)):
                            wt = w_q * p_s * p_zz
                            if wt == 0:
                                continue
                            li = a * qu + (sb | zin | z) * ru
                            EL += mass * wt * li * u
                            if li == 0:
                                sh, xs = block, xblk
                            else:
                                sh = np.zeros_like(block)
                                sh[li:K] = block[: K - li]
                                cross = block[K - li:K]                     # newly above T
                                over = (bins[K - li:K] + li - Tu)[:, None, None]
                                sh[K] = cross.sum(axis=0) + block[K]
                                xs = xblk + li * block[K] + (over * cross).sum(axis=0)
                            if li > 0:
                                sh, xs = _bump(sh, 1), _bump(xs, 0)
                            if li == full_u:
                                sh, xs = _bump(sh, 2), _bump(xs, 1)
                            newP[xr, z] += wt * sh
                            newX[xr, z] += wt * xs
        P, X = newP, newX
    D = P.sum(axis=(0, 1))                     # (K+1, 3, 3)
    return {
        "E[L]": EL,
        "P(>=2 pay)": float(D[:, 2, :].sum()),
        "P(L>T)": float(D[K].sum()),
        "stop-loss": float(X.sum()) * u,
        "P(>=2 exhausted)": float(D[:, :, 2].sum()),
    }


# ---------------------------------------------------------------------------
# Independent check: full joint CTMC over 5^n quake states (small n only)
# ---------------------------------------------------------------------------
def brute_force_chain(pf: ChainPortfolio) -> dict[str, float]:
    prob, Li = brute_force_joint(pf)
    s = pf.scen
    L = Li.sum(-1)
    full = s.quake_limit + s.rain_limit
    T = s.threshold
    return {
        "E[L]": float((prob * L).sum()),
        "P(>=2 pay)": float(prob[(Li > 0).sum(-1) >= 2].sum()),
        "P(L>T)": float(prob[L > T].sum()),
        "stop-loss": float((prob * np.maximum(L - T, 0)).sum()),
        "P(>=2 exhausted)": float(prob[(Li == full).sum(-1) >= 2].sum()),
    }


def brute_force_joint(pf: ChainPortfolio) -> tuple[np.ndarray, np.ndarray]:
    """Joint law over (quake state, rain state) and per-country losses, by full joint-CTMC
    enumeration: prob has shape (5^n, 2^n), Li shape (5^n, 2^n, n) in USD."""
    n, s, H = pf.n, pf.scen, pf.scen.years
    shape = (STEPS + 1,) * n
    N = int(np.prod(shape))
    G = np.zeros((N, N))
    own = pf.own_quake_rate
    for flat in range(N):
        st = np.array(np.unravel_index(flat, shape))
        for i in range(n):
            for k in range(1, STEPS + 1):
                nx = st.copy(); nx[i] = min(STEPS, nx[i] + k)
                G[flat, np.ravel_multi_index(nx, shape)] += own[i] * pf.w[i, k - 1]
        for e in range(n - 1):
            for k in range(1, STEPS + 1):
                for j in range(1, STEPS + 1):
                    nx = st.copy()
                    nx[e] = min(STEPS, nx[e] + k); nx[e + 1] = min(STEPS, nx[e + 1] + j)
                    G[flat, np.ravel_multi_index(nx, shape)] += pf.c[e] * pf.w[e, k - 1] * pf.w[e + 1, j - 1]
        G[flat, flat] -= G[flat].sum()
    peq = expm(G * H)[0].reshape(shape)
    # rain: enumerate own and shared Bernoulli triggers
    own_r = 1 - np.exp(-pf.own_rain_rate * H)
    sh_r = 1 - np.exp(-pf.cr * H)
    prain = np.zeros((2,) * n)
    for bits in np.ndindex(*((2,) * (2 * n - 1))):
        ob, eb = bits[:n], bits[n:]
        p = np.prod([own_r[i] if ob[i] else 1 - own_r[i] for i in range(n)])
        p *= np.prod([sh_r[e] if eb[e] else 1 - sh_r[e] for e in range(n - 1)]) if n > 1 else 1
        r = [ob[i] | (eb[i - 1] if i > 0 else 0) | (eb[i] if i < n - 1 else 0) for i in range(n)]
        prain[tuple(r)] += p
    qa = np.indices(shape).reshape(n, -1).T                 # quake states
    ra = np.indices((2,) * n).reshape(n, -1).T              # rain states
    Li = (s.quake_limit / STEPS) * qa[:, None, :] + s.rain_limit * ra[None, :, :]
    prob = peq.reshape(-1)[:, None] * prain.reshape(-1)[None, :]
    return prob, Li
