"""Boltzmann (Ising / Markov-random-field) law over peril-zone trigger bits on a 2-D grid.

Modelling choices (ours; the brief defines only the two-country model):

* Zones sit on a rows x cols grid. Bit b_k = 1 if zone k's quake cover triggers (at least
  one payout step) during the three-year term. Zones alternate between the brief's two
  calibrations in a checkerboard (KGZ at even r + c, TJK at odd), so every nearest-
  neighbour edge is a Kyrgyz-Tajik-type pair, exactly as in the brief.
* Calibration targets come from the brief's exact term law (model.py): the marginal
  P(b_k = 1) = P(a >= 1) of the zone's type, and for every grid edge the joint
  P(b_i = b_j = 1) = P(a >= 1, b >= 1) of the brief's pair (scenario dependent).
* The law is the maximum-entropy distribution with those moments:
      pi(b) = exp(-E(b)) / Z,   E(b) = sum_k theta_k b_k - sum_(i,j) J_ij b_i b_j,
  with one field per zone and one coupling per edge, fitted by convex moment matching.
  For a 1 x 2 grid this *is* the brief's joint trigger law (tests/test_grid.py).
* Contagion knob kappa: couplings J -> kappa J, then the fields are refitted so the
  marginals stay calibrated (as Notebook 19 of logistics_under_disruption does).
* Loss: a triggered zone pays its type's conditional mean term payout E[payout | trigger],
  rounded to the USD 2.5m lattice; L(b) = sum_k D_k b_k.

Conventions match logistics_under_disruption/src/supplychain_qmcmc (mrf.py, kernels.py):
energy as above, state ID with bit k at position n - 1 - k (zone 0 is the most significant
bit), so its kernels and MH machinery apply unchanged.

Exact engines
-------------
* ``enumerate_law``: all 2^n states (n <= ENUM_MAX_BITS); the independent ground truth.
* ``TransferMatrix``: column-by-column contraction. A rows x cols grid has treewidth
  min(rows, cols); the cost is about cols * 4^rows (times the loss bins for the loss law),
  so 8 x 8 is exact in under a second, 12 x 12 takes minutes, and wider grids are
  out of reach (TM_MAX_ROWS; measured in experiments/grid_exact.py).
  Both engines are cross-checked in the tests wherever both apply.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from .model import SCENARIOS, STEPS, Scenario, term_distribution_sources

ENUM_MAX_BITS = 22
TM_MAX_ROWS = 12
UNIT_USD = 2.5e6


# ---------------------------------------------------------------------------
# Calibration targets from the brief
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def brief_trigger_law(name: str = "adb-base") -> dict:
    """Term quake-trigger law of the brief's pair and conditional payouts (USD)."""
    s = SCENARIOS[name] if isinstance(name, str) else name
    q = term_distribution_sources(s).sum(axis=(2, 3))          # (a, b) quake steps
    p_a, p_b, p_ab = q[1:, :].sum(), q[:, 1:].sum(), q[1:, 1:].sum()
    step = s.quake_limit / STEPS
    pay_a = (np.arange(STEPS + 1)[:, None] * q).sum() / p_a * step
    pay_b = (np.arange(STEPS + 1)[None, :] * q).sum() / p_b * step
    return {"p": (float(p_a), float(p_b)), "p_both": float(p_ab),
            "joint": np.array([[q[0, 0], q[0, 1:].sum()], [q[1:, 0].sum(), p_ab]]),
            "payout": (float(pay_a), float(pay_b))}


# ---------------------------------------------------------------------------
# Grid geometry and the model
# ---------------------------------------------------------------------------
def zone(r: int, c: int, cols: int) -> int:
    return r * cols + c


def grid_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    """Nearest-neighbour edges (i < j): horizontal first, then vertical."""
    e = [(zone(r, c, cols), zone(r, c + 1, cols)) for r in range(rows) for c in range(cols - 1)]
    e += [(zone(r, c, cols), zone(r + 1, c, cols)) for r in range(rows - 1) for c in range(cols)]
    return e


@dataclass(frozen=True)
class GridModel:
    rows: int
    cols: int
    theta: np.ndarray                    # (n,) fields
    J: np.ndarray                        # (n_edges,) couplings, order of grid_edges
    payout_units: np.ndarray             # (n,) loss per triggered zone, in UNIT_USD
    p_target: np.ndarray                 # (n,) calibrated marginals
    pair_target: np.ndarray              # (n_edges,) calibrated P(b_i = b_j = 1)
    scenario: str = "adb-base"
    kappa: float = 1.0

    @property
    def n(self) -> int:
        return self.rows * self.cols

    @property
    def edge_list(self) -> list[tuple[int, int]]:
        return grid_edges(self.rows, self.cols)

    @property
    def edges(self) -> tuple[tuple[int, int, float], ...]:
        """(i, j, J_ij) triples in the logistics project's format."""
        return tuple((i, j, float(Jv)) for (i, j), Jv in zip(self.edge_list, self.J))

    @property
    def max_loss_units(self) -> int:
        return int(self.payout_units.sum())


def zone_types(rows: int, cols: int) -> np.ndarray:
    """0 = KGZ-type, 1 = TJK-type, checkerboard."""
    r, c = np.indices((rows, cols))
    return ((r + c) % 2).reshape(-1)


def calibrated_grid(rows: int, cols: int, scenario: str = "adb-base", kappa: float = 1.0,
                    tol: float = 1e-11) -> GridModel:
    """Maximum-entropy grid law matching the brief's marginals and edge joints exactly."""
    law = brief_trigger_law(scenario)
    types = zone_types(rows, cols)
    p = np.array(law["p"])[types]
    E = grid_edges(rows, cols)
    pair = np.full(len(E), law["p_both"])
    pay = np.rint(np.array(law["payout"]) / UNIT_USD).astype(int)[types]
    # initial point: the isolated-pair solution
    j0 = np.log(law["joint"][0, 0] * law["joint"][1, 1] / (law["joint"][0, 1] * law["joint"][1, 0]))
    theta0 = -np.log(p / (1 - p))
    J0 = np.full(len(E), j0)
    if abs(j0) < 1e-12 or rows * cols == 1:
        J0[:] = 0.0
    m = GridModel(rows, cols, theta0, J0, pay, p, pair, scenario, kappa)
    if np.allclose(J0, 0):                         # independent scenario: exact product law
        return m
    m = fit_moments(m, fit_couplings=True, tol=tol)
    if kappa != 1.0:
        m = fit_moments(GridModel(rows, cols, m.theta, kappa * m.J, pay, p, pair, scenario, kappa),
                        fit_couplings=False, tol=tol)
    return m


# ---------------------------------------------------------------------------
# Exact engine 1: enumeration
# ---------------------------------------------------------------------------
def all_bits(n: int) -> np.ndarray:
    if n > ENUM_MAX_BITS:
        raise ValueError(f"enumeration limited to {ENUM_MAX_BITS} bits")
    ids = np.arange(2**n, dtype=np.int64)
    return ((ids[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.int8)


def energies(m: GridModel, bits: np.ndarray | None = None) -> np.ndarray:
    b = all_bits(m.n) if bits is None else bits
    e = b @ m.theta
    for (i, j), Jv in zip(m.edge_list, m.J):
        e -= Jv * (b[:, i] & b[:, j])
    return e


def enumerate_law(m: GridModel) -> dict:
    """States' energies, probabilities, losses (USD), log Z, marginals and edge joints."""
    b = all_bits(m.n)
    e = energies(m, b)
    w = np.exp(-(e - e.min()))
    Z = w.sum()
    pi = w / Z
    E = m.edge_list
    return {"energies": e, "pi": pi, "logZ": float(np.log(Z) - e.min()),
            "loss": (b @ m.payout_units) * UNIT_USD, "loss_units": b @ m.payout_units,
            "marginals": pi @ b, "pair": np.array([pi @ (b[:, i] & b[:, j]) for i, j in E])}


# ---------------------------------------------------------------------------
# Exact engine 2: column transfer matrix
# ---------------------------------------------------------------------------
class TransferMatrix:
    """Exact contraction over columns of ``rows`` bits (column state x: bit r = zone (r, c))."""

    def __init__(self, m: GridModel):
        if m.rows > TM_MAX_ROWS:
            raise ValueError(f"transfer matrix limited to {TM_MAX_ROWS} rows (cost ~ 4^rows)")
        self.m = m
        R, C = m.rows, m.cols
        self.xs = ((np.arange(2**R)[:, None] >> np.arange(R)) & 1).astype(float)   # (2^R, R)
        J = dict(zip(m.edge_list, m.J))
        self.col_logw, self.col_loss, self.hlog = [], [], []
        for c in range(C):
            ks = [zone(r, c, C) for r in range(R)]
            lw = -self.xs @ m.theta[ks]
            for r in range(R - 1):
                lw += J[(ks[r], ks[r + 1])] * self.xs[:, r] * self.xs[:, r + 1]
            self.col_logw.append(lw)
            self.col_loss.append((self.xs @ m.payout_units[ks]).astype(int))
            if c < C - 1:
                jh = np.array([J[(ks[r], zone(r, c + 1, C))] for r in range(R)])
                self.hlog.append((self.xs * jh) @ self.xs.T)                       # (2^R, 2^R)
        self._run()

    def _run(self):
        C = self.m.cols
        self.alpha, self.beta, logs = [], [None] * C, 0.0
        a = np.exp(self.col_logw[0] - self.col_logw[0].max()); logs += self.col_logw[0].max()
        s = a.sum(); a /= s; logs += np.log(s)
        self.alpha.append(a)
        for c in range(1, C):
            lw = self.col_logw[c]
            a = (a @ np.exp(self.hlog[c - 1])) * np.exp(lw - lw.max()); logs += lw.max()
            s = a.sum(); a /= s; logs += np.log(s)
            self.alpha.append(a)
        self.logZ = float(logs)
        b = np.ones(2 ** self.m.rows)
        self.beta[C - 1] = b
        for c in range(C - 2, -1, -1):
            w = np.exp(self.col_logw[c + 1] - self.col_logw[c + 1].max())
            b = np.exp(self.hlog[c]) @ (w * b)
            b /= b.sum()
            self.beta[c] = b

    def column_marginal(self, c: int) -> np.ndarray:
        p = self.alpha[c] * self.beta[c]
        return p / p.sum()

    def moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact marginals P(b_k = 1) and edge joints P(b_i = b_j = 1) (grid_edges order)."""
        m, R, C = self.m, self.m.rows, self.m.cols
        marg = np.zeros(m.n)
        pair = {}
        for c in range(C):
            pc = self.column_marginal(c)
            for r in range(R):
                marg[zone(r, c, C)] = pc @ self.xs[:, r]
            for r in range(R - 1):
                pair[(zone(r, c, C), zone(r + 1, c, C))] = pc @ (self.xs[:, r] * self.xs[:, r + 1])
            if c < C - 1:
                w = np.exp(self.col_logw[c + 1] - self.col_logw[c + 1].max())
                P2 = self.alpha[c][:, None] * np.exp(self.hlog[c]) * (w * self.beta[c + 1])[None, :]
                P2 /= P2.sum()
                for r in range(R):
                    pair[(zone(r, c, C), zone(r, c + 1, C))] = self.xs[:, r] @ P2 @ self.xs[:, r]
        return marg, np.array([pair[e] for e in m.edge_list])

    def loss_pmf(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact law of L = sum_k D_k b_k: (values in USD, probabilities)."""
        nb = self.m.max_loss_units + 1
        A = np.zeros((2 ** self.m.rows, nb))
        w0 = np.exp(self.col_logw[0] - self.col_logw[0].max())
        A[np.arange(len(w0)), self.col_loss[0]] = w0
        A /= A.sum()
        for c in range(1, self.m.cols):
            M = np.exp(self.hlog[c - 1]).T @ A                       # (2^R, nb), indexed by new x
            w = np.exp(self.col_logw[c] - self.col_logw[c].max())
            new = np.zeros_like(A)
            for x in range(len(w)):
                sh = self.col_loss[c][x]
                new[x, sh:] = w[x] * M[x, : nb - sh]
            A = new / new.sum()
        pmf = A.sum(axis=0)
        return UNIT_USD * np.arange(nb), pmf / pmf.sum()


def exact_moments(m: GridModel) -> tuple[np.ndarray, np.ndarray, float]:
    """(marginals, edge joints, log Z) with the cheapest exact engine."""
    if m.rows <= TM_MAX_ROWS:
        tm = TransferMatrix(m)
        mg, pr = tm.moments()
        return mg, pr, tm.logZ
    if m.n <= ENUM_MAX_BITS:
        d = enumerate_law(m)
        return d["marginals"], d["pair"], d["logZ"]
    raise ValueError("no exact engine for this grid")


def exact_loss_pmf(m: GridModel) -> tuple[np.ndarray, np.ndarray]:
    if m.rows <= TM_MAX_ROWS:
        return TransferMatrix(m).loss_pmf()
    if m.n <= ENUM_MAX_BITS:
        d = enumerate_law(m)
        pmf = np.bincount(d["loss_units"], weights=d["pi"], minlength=m.max_loss_units + 1)
        return UNIT_USD * np.arange(len(pmf)), pmf
    raise ValueError("no exact engine for this grid")


# ---------------------------------------------------------------------------
# Moment matching (convex maximum-entropy dual)
# ---------------------------------------------------------------------------
def fit_moments(m: GridModel, fit_couplings: bool = True, tol: float = 1e-11) -> GridModel:
    """Minimise F = log Z + theta . p - J . pair (convex); grad = (p - E b, E bb - pair).

    With fit_couplings=False only the fields move (J fixed), matching the marginals."""
    nE = len(m.J)

    def unpack(x):
        return (x[: m.n], x[m.n:]) if fit_couplings else (x, m.J)

    def f(x):
        th, J = unpack(x)
        mm = GridModel(m.rows, m.cols, th, J, m.payout_units, m.p_target, m.pair_target, m.scenario, m.kappa)
        mg, pr, logZ = exact_moments(mm)
        val = logZ + th @ m.p_target - (J @ m.pair_target if fit_couplings else 0.0)
        g = m.p_target - mg
        if fit_couplings:
            g = np.concatenate([g, pr - m.pair_target])
        return val, g

    x0 = np.concatenate([m.theta, m.J]) if fit_couplings else m.theta.copy()
    res = minimize(f, x0, jac=True, method="L-BFGS-B",
                   options={"gtol": tol, "ftol": 1e-15, "maxiter": 5000, "maxcor": 30})
    th, J = unpack(res.x)
    out = GridModel(m.rows, m.cols, th, np.asarray(J, float), m.payout_units, m.p_target,
                    m.pair_target, m.scenario, m.kappa)
    resid = np.abs(res.jac).max()
    if resid > 1e-7:
        raise RuntimeError(f"moment matching did not converge (max residual {resid:.2e})")
    return out


def calibration_residual(m: GridModel) -> float:
    mg, pr, _ = exact_moments(m)
    r = np.abs(mg - m.p_target).max()
    if m.kappa == 1.0:
        r = max(r, np.abs(pr - m.pair_target).max())
    return float(r)
