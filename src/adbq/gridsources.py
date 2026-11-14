"""The brief's compound-Poisson source model on a 2-D grid of countries / peril zones.

Scaling family (ours; generalises portfolio.py's chain to two dimensions): zones on a
rows x cols grid alternate KGZ / TJK calibrations (checkerboard). Every horizontal and
vertical neighbour pair shares a quake stream and a precipitation stream, calibrated
exactly as the brief calibrates the Kyrgyz-Tajik pair. Each zone keeps its total rates,
so its marginal law is the brief's; its own streams carry what the shared ones do not.
A 1 x n grid is the chain (tests/test_gridsources.py), and a 1 x 2 grid is the brief.

Interior zones have four neighbours, so with the brief's shared-rain share (25%) their
own rain rate is 1 - 4 x 0.25 = 0: all their rain comes from shared streams (clipped at 0).

The term law is a product of independent sources followed by capped sums, exactly the
structure the quantum state preparation loads (oracle.py): one query of a classical
sampler = one draw of all sources; one quantum query = one application of A.

Exact law: a zone-by-zone (column-major) transfer-matrix sweep. The frontier carries,
for each row, the pending (quake 0..4, rain 0/1) contribution of a horizontal edge, plus
one vertical edge: 10^(rows+1) states x loss bins. That is the treewidth wall for this
model; the measured frontier is in experiments/grid_qae.py. Exact mean and variance of L
are available at any size, because zones covary only through a shared edge.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np

from .model import (STEPS, Scenario, capped_compound_poisson, country_tables,
                    capped_shared_pair, poisson_rates)
from .portfolio import _units

EXACT_MAX_ROWS = 4          # 10^5 frontier states x loss bins; 5 rows needs ~10x the memory


@dataclass(frozen=True)
class GridPortfolio:
    rows: int
    cols: int
    lam: np.ndarray        # (n,) total quake rates
    w: np.ndarray          # (n, 4) quake size laws
    edges: tuple           # ((i, j), ...) i < j; horizontal then vertical
    c: np.ndarray          # (E,) shared quake rates
    mu: np.ndarray         # (n,) total rain rates
    cr: np.ndarray         # (E,) shared rain rates
    scen: Scenario

    @property
    def n(self) -> int:
        return self.rows * self.cols

    @cached_property
    def own_quake_rate(self) -> np.ndarray:
        out = self.lam.copy()
        for e, (i, j) in enumerate(self.edges):
            out[i] -= self.c[e]; out[j] -= self.c[e]
        return out

    @cached_property
    def own_rain_rate(self) -> np.ndarray:
        out = self.mu.copy()
        for e, (i, j) in enumerate(self.edges):
            out[i] -= self.cr[e]; out[j] -= self.cr[e]
        return np.where(np.abs(out) < 1e-14, 0.0, out)

    # ---- independent source laws over the term ----
    @cached_property
    def sources(self) -> dict:
        H = self.scen.years
        oq = np.array([capped_compound_poisson(self.own_quake_rate[i] * self.w[i] * H) for i in range(self.n)])
        orr = 1 - np.exp(-self.own_rain_rate * H)
        sq = np.array([capped_shared_pair(self.c[e] * H, self.w[i], self.w[j])
                       for e, (i, j) in enumerate(self.edges)]).reshape(len(self.edges), -1)
        sr = 1 - np.exp(-self.cr * H)
        return {"own_quake": oq, "own_rain": orr, "shared_quake": sq, "shared_rain": sr}

    @cached_property
    def units(self) -> tuple[int, int, int]:
        qu, ru, _, u = _units(replace(self.scen, threshold=0.0))
        return qu, ru, u

    @property
    def full_units(self) -> int:
        qu, ru, _ = self.units
        return STEPS * qu + ru

    @property
    def n_sources(self) -> int:
        return 2 * self.n + 2 * len(self.edges)


def grid_edges(rows: int, cols: int) -> tuple:
    z = lambda r, c: r * cols + c
    e = [(z(r, c), z(r, c + 1)) for r in range(rows) for c in range(cols - 1)]
    e += [(z(r, c), z(r + 1, c)) for r in range(rows - 1) for c in range(cols)]
    return tuple(e)


def grid_portfolio(rows: int, cols: int, s: Scenario = Scenario()) -> GridPortfolio:
    r, c = np.indices((rows, cols))
    (_, ta), (_, tb), (pf_, pb) = country_tables(s)
    types = ((r + c) % 2).reshape(-1)               # 0 = type A (KGZ by default), 1 = type B
    lam, w = [], []
    for ty in types:
        L, l = poisson_rates((ta, tb)[ty])
        lam.append(L); w.append(l / L)
    lam, w = np.array(lam), np.array(w)
    E = grid_edges(rows, cols)
    # the brief's formula with the type-A end of the edge first
    cq = []
    for i, j in E:
        a, b = (i, j) if types[i] == 0 else (j, i)
        cq.append(s.quake_mult * 0.5 * (pf_ * lam[a] + pb * lam[b]))
    mu = np.full(rows * cols, -np.log(1 - s.rain_annual_prob) * s.rain_mult)
    cr = np.array([s.rain_shared_share * min(mu[i], mu[j]) for i, j in E])
    gp = GridPortfolio(rows, cols, lam, w, E, np.array(cq), mu, cr, s)
    assert (gp.own_quake_rate > 0).all() and (gp.own_rain_rate >= 0).all()
    return gp


# ---------------------------------------------------------------------------
# Exact mean and variance of L (any size)
# ---------------------------------------------------------------------------
def _zone_law_given(gp: GridPortfolio, i: int, exclude: int | None = None) -> np.ndarray:
    """Law of (quake steps 0..4 capped, rain bit) of zone i from all its sources except the
    shared edge ``exclude``: returns P[q_raw (0..4 capped), rain] as a (5, 2) array."""
    S = gp.sources
    q = S["own_quake"][i].copy()
    p_dry = 1 - S["own_rain"][i]
    for e, (a, b) in enumerate(gp.edges):
        if e == exclude or i not in (a, b):
            continue
        J = S["shared_quake"][e].reshape(STEPS + 1, STEPS + 1)
        marg = J.sum(1) if i == a else J.sum(0)
        new = np.zeros(STEPS + 1)
        for x, px in enumerate(q):
            for y, py in enumerate(marg):
                new[min(STEPS, x + y)] += px * py
        q = new
        p_dry *= 1 - S["shared_rain"][e]
    return np.outer(q, [p_dry, 1 - p_dry])


def _loss_of(gp, q, rain):
    qu, ru, u = gp.units
    return (np.minimum(STEPS, q) * qu + rain * ru) * u


def exact_mean_var(gp: GridPortfolio) -> tuple[float, float]:
    qa = np.arange(STEPS + 1)
    Lz = _loss_of(gp, qa[:, None], np.array([0, 1])[None, :])      # (5, 2)
    m = np.zeros(gp.n); v2 = np.zeros(gp.n)
    for i in range(gp.n):
        P = _zone_law_given(gp, i)
        m[i] = (P * Lz).sum(); v2[i] = (P * Lz**2).sum()
    var = float((v2 - m**2).sum())
    S = gp.sources
    for e, (i, j) in enumerate(gp.edges):
        Pi, Pj = _zone_law_given(gp, i, e), _zone_law_given(gp, j, e)
        J = S["shared_quake"][e].reshape(STEPS + 1, STEPS + 1)
        pz = S["shared_rain"][e]
        Eij = 0.0
        for z, p_z in ((0, 1 - pz), (1, pz)):
            for (x, y), pxy in np.ndenumerate(J):
                if pxy == 0:
                    continue
                # E[L_i | x, z] with zone i's other sources
                Li = sum(Pi[q, r] * _loss_of(gp, q + x, r | z) for q in range(STEPS + 1) for r in (0, 1))
                Lj = sum(Pj[q, r] * _loss_of(gp, q + y, r | z) for q in range(STEPS + 1) for r in (0, 1))
                Eij += p_z * pxy * Li * Lj
        var += 2 * (Eij - m[i] * m[j])
    return float(m.sum()), var


def threshold(gp: GridPortfolio, k: float) -> float:
    """Lattice-rounded E[L] + k sd(L): a size-independent definition of a 'k-sigma' loss."""
    mu, var = exact_mean_var(gp)
    u = gp.units[2]
    return float(np.ceil((mu + k * np.sqrt(var)) / u) * u)


# ---------------------------------------------------------------------------
# Exact loss law: zone-by-zone transfer-matrix sweep
# ---------------------------------------------------------------------------
def _zone_table(gp: GridPortfolio, r: int, c: int) -> np.ndarray:
    """W[inc, sout, vout, a, rain]: incoming merged (quake 0..4, rain) -> outgoing right-edge
    slot, down-edge slot, and the zone's (capped quake, rain)."""
    R, C = gp.rows, gp.cols
    i = r * C + c
    S = gp.sources
    eidx = {e: k for k, e in enumerate(gp.edges)}
    one = np.zeros((STEPS + 1, STEPS + 1)); one[0, 0] = 1.0
    if c < C - 1:
        k = eidx[(i, i + 1)]
        Jr, pzr = S["shared_quake"][k].reshape(STEPS + 1, STEPS + 1), S["shared_rain"][k]
    else:
        Jr, pzr = one, 0.0
    if r < R - 1:
        k = eidx[(i, i + C)]
        Jd, pzd = S["shared_quake"][k].reshape(STEPS + 1, STEPS + 1), S["shared_rain"][k]
    else:
        Jd, pzd = one, 0.0
    po, pr = S["own_quake"][i], S["own_rain"][i]
    W = np.zeros((10, 10, 10, STEPS + 1, 2))
    # local quake: own + this zone's part of right and down edges
    for o, p_o in enumerate(po):
        if p_o == 0:
            continue
        for (xt, xr), p1 in np.ndenumerate(Jr):
            if p1 == 0:
                continue
            for (yt, yd), p2 in np.ndenumerate(Jd):
                if p2 == 0:
                    continue
                pq = p_o * p1 * p2
                for b, pb in ((0, 1 - pr), (1, pr)):
                    for zr, p3 in ((0, 1 - pzr), (1, pzr)):
                        for zd, p4 in ((0, 1 - pzd), (1, pzd)):
                            wt = pq * pb * p3 * p4
                            if wt == 0:
                                continue
                            for iq in range(STEPS + 1):
                                for ir in (0, 1):
                                    a = min(STEPS, iq + o + xt + yt)
                                    W[iq * 2 + ir, xr * 2 + zr, yd * 2 + zd, a, ir | b | zr | zd] += wt
    return W


def exact_loss_pmf(gp: GridPortfolio, max_rows: int = EXACT_MAX_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """Exact law of the portfolio loss: (values in USD, probabilities)."""
    R, C = gp.rows, gp.cols
    if R > max_rows:
        raise ValueError(f"exact sweep limited to {max_rows} rows (frontier 10^(rows+1) x loss bins)")
    qu, ru, u = gp.units
    nb = gp.n * gp.full_units + 1
    merge = np.array([[min(STEPS, (s // 2) + (v // 2)) * 2 + ((s % 2) | (v % 2)) for v in range(10)]
                      for s in range(10)])
    F = np.zeros((10,) * R + (10, nb))
    F[(0,) * (R + 1) + (0,)] = 1.0
    for c in range(C):
        for r in range(R):
            W = _zone_table(gp, r, c)
            F2 = np.moveaxis(F, [r, R], [0, 1])
            rest = F2.shape[2:-1]
            F2 = F2.reshape(10, 10, -1, nb)
            H = np.zeros((10,) + F2.shape[2:])
            for s in range(10):
                for v in range(10):
                    H[merge[s, v]] += F2[s, v]
            out = np.zeros_like(F2)
            for a in range(STEPS + 1):
                for rain in (0, 1):
                    li = a * qu + rain * ru
                    Wl = W[:, :, :, a, rain]                       # (inc, sout, vout)
                    if not Wl.any():
                        continue
                    Hs = np.zeros_like(H)
                    Hs[..., li:] = H[..., : nb - li]
                    out += np.tensordot(Wl, Hs, axes=(0, 0))
            F = np.moveaxis(out.reshape((10, 10) + rest + (nb,)), [0, 1], [r, R])
    pmf = F.reshape(-1, nb).sum(0)
    return u * np.arange(nb), pmf


# ---------------------------------------------------------------------------
# Sampling (inverse CDF from uniforms, so the same code serves MC, RQMC and IS)
# ---------------------------------------------------------------------------
def source_laws(gp: GridPortfolio) -> dict:
    """Categorical laws per source: own quake (n, 5), own rain (n, 2), shared quake (E, 25),
    shared rain (E, 2)."""
    S = gp.sources
    return {"own_quake": S["own_quake"].copy(),
            "own_rain": np.stack([1 - S["own_rain"], S["own_rain"]], 1),
            "shared_quake": S["shared_quake"].copy(),
            "shared_rain": np.stack([1 - S["shared_rain"], S["shared_rain"]], 1)}


KEYS = ("own_quake", "own_rain", "shared_quake", "shared_rain")


def _inv(laws: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Inverse CDF per column: laws (k, m), U (N, k) -> indices (N, k)."""
    cdf = np.cumsum(laws, 1); cdf[:, -1] = 1.0
    return (U[:, :, None] >= cdf[None, :, :]).sum(2)


def draw(gp: GridPortfolio, U: np.ndarray, laws: dict | None = None) -> dict:
    """Source outcomes from uniforms U (N, n_sources) under ``laws`` (default: the model)."""
    laws = laws or source_laws(gp)
    n, E = gp.n, len(gp.edges)
    cuts = np.cumsum([0, n, n, E, E])
    return {k: _inv(laws[k], U[:, cuts[t]:cuts[t + 1]]) for t, k in enumerate(KEYS)}


def losses(gp: GridPortfolio, X: dict) -> np.ndarray:
    """Portfolio loss in units u for each draw."""
    qu, ru, _ = gp.units
    N = X["own_quake"].shape[0]
    q = X["own_quake"].astype(int).copy()
    rain = X["own_rain"].astype(int).copy()
    for e, (i, j) in enumerate(gp.edges):
        xi, xj = np.divmod(X["shared_quake"][:, e], STEPS + 1)      # index = xi * 5 + xj
        q[:, i] += xi; q[:, j] += xj
        z = X["shared_rain"][:, e]
        rain[:, i] |= z; rain[:, j] |= z
    return (np.minimum(STEPS, q) * qu + rain * ru).sum(1)


def log_weight(gp: GridPortfolio, X: dict, laws_q: dict) -> np.ndarray:
    """log p(x) / q(x) for draws X made under laws_q."""
    lp = source_laws(gp)
    out = 0.0
    for k in KEYS:
        idx = X[k]
        rows = np.arange(idx.shape[1])[None, :]
        with np.errstate(divide="ignore"):
            out = out + (np.log(lp[k][rows, idx]) - np.log(laws_q[k][rows, idx])).sum(1)
    return out
