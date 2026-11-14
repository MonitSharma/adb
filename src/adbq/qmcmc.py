"""Quantum-enhanced MCMC (Layden et al. 2023) for CVaR on the grid Boltzmann model.

Reuses the team's logistics_under_disruption project (kernels.py: exact MH matrices,
asymptotic variances, spectral gaps; quantum.py: Layden Hamiltonian, alpha, parameter
ensemble, Trotter unitaries), vendored unchanged in adbq._vendor.supplychain_qmcmc.

Target. The stress law tilts the grid law towards large losses,
    pi_lam(b) ∝ exp(-E(b) + lam * L(b) / USD 100m),
and CVaR_alpha under the true law pi is read from the self-normalised weighted histogram
with weights pi / pi_lam ∝ exp(-lam L / 100m) (exact importance correction, as in
Notebooks 17 and 23 of the logistics project).

Moves (all symmetric unless stated; acceptance min(1, exp(-dE_lam))):
  local     flip one uniformly chosen zone;
  pair      flip both zones of a uniformly chosen grid edge (structure-aware, same
            couplings as the quantum Hamiltonian);
  plaquette flip a uniformly chosen 2 x 2 block (four bits);
  uniform   propose a uniformly random configuration (classical non-local control);
  wolff     Wolff cluster on the couplings, bond probability 1 - exp(-J_ij / 2) between
            equal bits, field-only acceptance min(1, exp(-(dE_lam - dE_coupling)))
            (detailed balance checked exactly in tests/test_qmcmc.py);
  quantum   Layden proposal: prepare |s>, evolve under (1-g) alpha H_prob + g sum X with
            palindromic Trotter steps, measure; averaged over a fixed ensemble of (g, t)
            draws. U = U^T, so the proposal is exactly symmetric. 'quantum_mismatch' uses a
            Hamiltonian built from a randomly relabelled grid (wrong landscape control).

The quantum emulator evolves only the basis states it needs (columns), with the logistics
project's blocked RX layer (quantum._apply_rx_all); it is checked against
quantum.trotter_unitary. No 2^n x 2^n matrix is formed, so 16 qubits are cheap per state. Proposal rows are sampled through per-state pools of
i.i.d. draws from the exact row, so every proposal is an exact draw from Q[s, .].

Cost unit: one loss evaluation = one proposed state (parallel tempering: one per replica).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .grid import GridModel, all_bits, energies as grid_energies, enumerate_law, zone
from .risk import var_cvar

LOGISTICS_SRC = os.environ.get("ADBQ_LOGISTICS_SRC")      # optional override: an external checkout


def logistics():
    """(kernels, quantum) from the team's earlier project, vendored in adbq._vendor.
    Set ADBQ_LOGISTICS_SRC to a logistics_under_disruption/src checkout to use that instead."""
    if LOGISTICS_SRC:
        if LOGISTICS_SRC not in sys.path:
            sys.path.insert(0, LOGISTICS_SRC)
        from supplychain_qmcmc import kernels, quantum
        return kernels, quantum
    from ._vendor.supplychain_qmcmc import kernels, quantum
    return kernels, quantum


LOSS_SCALE = 1e8      # lam is per USD 100m of loss


# ---------------------------------------------------------------------------
# Target law, tilt and the CVaR influence function
# ---------------------------------------------------------------------------
@dataclass
class Target:
    model: GridModel
    lam: float = 0.0
    alpha: float = 0.95

    def __post_init__(self):
        d = enumerate_law(self.model)
        self.E, self.pi, self.L = d["energies"], d["pi"], d["loss"].astype(float)
        self.n = self.model.n
        self.E_lam = self.E - self.lam * self.L / LOSS_SCALE
        p = np.exp(-(self.E_lam - self.E_lam.min()))
        self.pi_lam = p / p.sum()
        self.var, self.cvar = var_cvar(self.L, self.pi, self.alpha)
        w = self.pi / self.pi_lam
        self.w = w / (self.pi_lam @ w)                     # E_{pi_lam} w = 1
        self.psi = self.var + np.maximum(self.L - self.var, 0) / (1 - self.alpha) - self.cvar

    def iid_variance(self) -> float:
        """N Var(CVaR-hat) for independent draws from pi_lam (perfect mixing)."""
        return float(self.pi_lam @ (self.w * self.psi) ** 2)

    def cvar_from_states(self, states: np.ndarray) -> float:
        wts = np.exp(-self.lam * self.L[states] / LOSS_SCALE)
        p = np.bincount(states, weights=wts, minlength=len(self.L))
        return var_cvar(self.L, p / p.sum(), self.alpha)[1]


def best_iid_lambda(model: GridModel, alpha: float = 0.95, grid=np.linspace(0, 3, 31)) -> float:
    """Tilt minimising the perfect-mixing variance; shared by every sampler (fair default)."""
    v = [Target(model, lam, alpha).iid_variance() for lam in grid]
    return float(grid[int(np.argmin(v))])


# ---------------------------------------------------------------------------
# Fast quantum proposal emulator
# ---------------------------------------------------------------------------
def evolve_basis(E_h: np.ndarray, states: np.ndarray, gamma: float, t: float, steps: int,
                 alpha: float, dtype=complex) -> np.ndarray:
    """Columns U|s> for each s in ``states``: r palindromic Strang steps of
    exp(-i H t), H = (1 - gamma) alpha diag(E_h - mean) + gamma sum_k X_k
    (identical to quantum.trotter_unitary applied to basis vectors)."""
    _, q = logistics()
    N = len(E_h)
    n = int(np.log2(N))
    dt = t / steps
    half = np.exp(-1j * (1 - gamma) * alpha * (E_h - E_h.mean()) * dt / 2).astype(dtype)
    X = np.zeros((N, len(states)), dtype)
    X[states, np.arange(len(states))] = 1.0
    for _ in range(steps):
        X *= half[:, None]
        X = q._apply_rx_all(X, 2 * gamma * dt, n)
        X *= half[:, None]
    return X


@dataclass
class QuantumProposal:
    """Ensemble-averaged Layden proposal with per-state pools of exact i.i.d. draws."""
    E_h: np.ndarray                    # landscape encoded in the Hamiltonian
    params: np.ndarray                 # (K, 2) fixed (gamma, t) draws
    steps: int = 6
    pool: int = 512
    alpha: float | None = None
    dtype: type = complex          # np.complex64 halves the cost; rows still sum to 1 within 1e-6
    _pools: dict = field(default_factory=dict)
    rows_computed: int = 0

    def __post_init__(self):
        _, q = logistics()
        if self.alpha is None:
            self.alpha = q.layden_alpha(self.E_h)

    def rows(self, states: np.ndarray) -> np.ndarray:
        """Exact proposal rows Q[s, .] for the given states, shape (len(states), 2^n)."""
        acc = np.zeros((len(self.E_h), len(states)))
        for g, t in self.params:
            acc += np.abs(evolve_basis(self.E_h, states, g, t, self.steps, self.alpha, self.dtype)) ** 2
        self.rows_computed += len(states)
        R = (acc / len(self.params)).T
        return R / R.sum(axis=1, keepdims=True)

    def _fill(self, states, rng):
        for k in range(0, len(states), 64):
            batch = np.asarray(states[k:k + 64], np.int64)
            for s, row in zip(batch, self.rows(batch)):
                c = np.cumsum(row); c[-1] = 1.0
                self._pools[int(s)] = [np.searchsorted(c, rng.random(self.pool), side="right").astype(np.int32), 0]

    def prefill(self, states, pool_sizes, rng, batch: int = 64, log=None):
        """Precompute pools for many states in full batches (much faster than lazy refills).
        ``pool_sizes`` gives one pool length per state (e.g. proportional to expected visits)."""
        states = np.asarray(states, np.int64)
        for k in range(0, len(states), batch):
            b = states[k:k + batch]
            for s, row, size in zip(b, self.rows(b), pool_sizes[k:k + batch]):
                c = np.cumsum(row); c[-1] = 1.0
                self._pools[int(s)] = [np.searchsorted(c, rng.random(int(size)), side="right").astype(np.int32), 0]
            if log and (k // batch) % 50 == 0:
                log(f"    prefill {k + len(b)}/{len(states)}")

    def propose(self, current: np.ndarray, rng) -> np.ndarray:
        """One exact draw from Q[s, .] per entry (pool draws are i.i.d. and each used once)."""
        exhausted = lambda s: s not in self._pools or self._pools[s][1] >= len(self._pools[s][0])
        self._fill(sorted({int(s) for s in current if exhausted(int(s))}), rng)
        out = np.empty(len(current), np.int64)
        for i, s in enumerate(map(int, current)):
            if exhausted(s):                            # same state many times in one batch
                self._fill([s], rng)
            pool = self._pools[s]
            out[i] = pool[0][pool[1]]
            pool[1] += 1
        return out


def quantum_params(kind: str, rng, K: int = 6) -> np.ndarray:
    """'default': Layden's ensemble gamma ~ U(0.25, 0.6), t ~ U(2, 20);
    'short': gamma ~ U(0.25, 0.6), t ~ U(0.5, 2) (closer to local moves)."""
    _, q = logistics()
    tr = q.TIME_RANGE if kind == "default" else (0.5, 2.0)
    return q.draw_parameters(K, rng, time_range=tr)


def mismatched_landscape(model: GridModel, lam: float, rng) -> np.ndarray:
    """E_lam of the same grid with zones randomly relabelled (wrong structure, same scales)."""
    perm = rng.permutation(model.n)
    t = Target(model, lam)
    b = all_bits(model.n)
    ids = (b[:, perm] * (1 << np.arange(model.n - 1, -1, -1))).sum(1)
    return t.E_lam[ids]


# ---------------------------------------------------------------------------
# Classical moves
# ---------------------------------------------------------------------------
def masks(model: GridModel, kind: str) -> np.ndarray:
    n, R, C = model.n, model.rows, model.cols
    bit = lambda k: 1 << (n - 1 - k)
    if kind == "pair":
        return np.array([bit(i) | bit(j) for i, j in model.edge_list], np.int64)
    if kind == "plaquette":
        return np.array([bit(zone(r, c, C)) | bit(zone(r, c + 1, C)) | bit(zone(r + 1, c, C))
                         | bit(zone(r + 1, c + 1, C)) for r in range(R - 1) for c in range(C - 1)], np.int64)
    raise ValueError(kind)


class Wolff:
    """Wolff cluster proposal for E = theta.b - sum J b_i b_j with J > 0 (field-only accept)."""

    def __init__(self, model: GridModel):
        self.n = model.n
        self.nbrs = [[] for _ in range(model.n)]
        for (i, j), Jv in zip(model.edge_list, model.J):
            p = 1 - np.exp(-max(Jv, 0.0) / 2)
            self.nbrs[i].append((j, p, Jv)); self.nbrs[j].append((i, p, Jv))

    def bits(self, s):
        return [(s >> (self.n - 1 - k)) & 1 for k in range(self.n)]

    def propose_one(self, s: int, rng) -> tuple[int, float]:
        """(candidate, dE_coupling) for one state."""
        b = self.bits(s)
        seed = int(rng.integers(self.n))
        cl, stack = {seed}, [seed]
        while stack:
            i = stack.pop()
            for j, p, _ in self.nbrs[i]:
                if j not in cl and b[j] == b[i] and rng.random() < p:
                    cl.add(j); stack.append(j)
        mask = sum(1 << (self.n - 1 - k) for k in cl)
        # Spin form: J b_i b_j = (J/4)(s_i s_j + s_i + s_j + 1), s = 2b - 1. The cluster
        # construction cancels only the spin-coupling part K s_i s_j (K = J/4); flipping the
        # cluster changes it by 2K s_i s_j on each boundary bond. The rest of dE (fields,
        # tilt, and the s_i + s_j parts) is left to the acceptance step.
        dEc = 0.0
        for i in cl:
            for j, _, Jv in self.nbrs[i]:
                if j not in cl:
                    dEc += 0.5 * Jv * (2 * b[i] - 1) * (2 * b[j] - 1)
        return s ^ mask, dEc


# ---------------------------------------------------------------------------
# Chain runner (vectorised over chains)
# ---------------------------------------------------------------------------
def run_chains(target: Target, moves, starts, n_steps: int, rng, model: GridModel | None = None,
               beta: float = 1.0):
    """MH chains on pi_lam with a mixture of moves.

    ``moves`` = list of (weight, kind) with kind in {'local','pair','plaquette','uniform',
    'wolff'} or a QuantumProposal. Returns (states (chains, steps), acceptance per chain).
    One step = one proposed state = one loss evaluation."""
    model = model or target.model
    n = target.n
    e = beta * target.E_lam
    w = np.array([m[0] for m in moves], float); w /= w.sum()
    mk = {k: masks(model, k) for _, k in moves if isinstance(k, str) and k in ("pair", "plaquette")}
    wolff = Wolff(model) if any(isinstance(k, str) and k == "wolff" for _, k in moves) else None
    if wolff is not None and beta != 1.0:
        raise ValueError("the Wolff bond probabilities assume beta = 1")
    cur = np.asarray(starts, np.int64).copy()
    Rn = len(cur)
    out = np.empty((Rn, n_steps), np.int64)
    acc = np.zeros(Rn)
    for t in range(n_steps):
        pick = rng.choice(len(moves), size=Rn, p=w)
        cand = cur.copy()
        corr = np.zeros(Rn)
        for m, (_, kind) in enumerate(moves):
            sel = np.flatnonzero(pick == m)
            if not len(sel):
                continue
            if not isinstance(kind, str):
                cand[sel] = kind.propose(cur[sel], rng)
            elif kind == "local":
                cand[sel] = cur[sel] ^ (1 << (n - 1 - rng.integers(n, size=len(sel))))
            elif kind in mk:
                cand[sel] = cur[sel] ^ mk[kind][rng.integers(len(mk[kind]), size=len(sel))]
            elif kind == "uniform":
                cand[sel] = rng.integers(2**n, size=len(sel))
            elif kind == "wolff":
                for i in sel:
                    cand[i], dEc = wolff.propose_one(int(cur[i]), rng)
                    corr[i] = dEc
            else:
                raise ValueError(kind)
        ok = rng.random(Rn) < np.exp(-np.clip(e[cand] - e[cur] - corr, 0, None))
        cur = np.where(ok, cand, cur)
        acc += ok
        out[:, t] = cur
    return out, acc / n_steps


def run_tempering(target: Target, moves, betas, starts, n_sweeps: int, rng):
    """Parallel tempering on pi_lam^beta (betas[0] = 1), one mixture-move step per replica per
    sweep, then neighbour swaps. Cost: len(betas) loss evaluations per sweep."""
    betas = np.asarray(betas, float)
    K, Rn = len(betas), len(starts)
    st = np.tile(np.asarray(starts, np.int64), (K, 1))
    cold = np.empty((Rn, n_sweeps), np.int64)
    e = target.E_lam
    for t in range(n_sweeps):
        for k in range(K):
            st[k] = run_chains(target, moves, st[k], 1, rng, beta=betas[k])[0][:, 0]
        for k in range(K - 1):
            a = (betas[k] - betas[k + 1]) * (e[st[k]] - e[st[k + 1]])
            ok = rng.random(Rn) < np.exp(np.clip(a, None, 0))
            tmp = st[k][ok].copy(); st[k][ok] = st[k + 1][ok]; st[k + 1][ok] = tmp
        cold[:, t] = st[0]
    return cold


# ---------------------------------------------------------------------------
# Exact matrices (n <= 12): proposal matrices and steps-to-accuracy
# ---------------------------------------------------------------------------
def proposal_matrices(model: GridModel, target: Target, rng, K: int = 6, steps: int = 6) -> dict:
    """Dense symmetric proposal matrices for the exact scorecard (2^n x 2^n)."""
    kern, qm = logistics()
    n, N = model.n, 2**model.n
    if n > 12:
        raise ValueError("dense matrices limited to 12 bits")
    ids = np.arange(N)
    Q = {"local": kern.local_proposal(n), "uniform": kern.uniform_proposal(n)}
    for kind in ("pair", "plaquette"):
        M = np.zeros((N, N))
        for mk in masks(model, kind):
            M[ids, ids ^ mk] += 1.0 / len(masks(model, kind))
        Q[kind] = M
    for kind in ("default", "short"):
        prm = quantum_params(kind, rng, K)
        Q[f"quantum_{kind}"] = QuantumProposal(target.E_lam, prm, steps).rows(ids)
    Q["quantum_mismatch"] = QuantumProposal(mismatched_landscape(model, target.lam, rng),
                                            quantum_params("short", rng, K), steps).rows(ids)
    return Q


def steps_to_accuracy(P_or_Q, target: Target, rel: float = 0.1, is_proposal: bool = True) -> float:
    """Steps for relative RMSE ``rel`` on CVaR: sigma^2_asym(w psi) / (rel CVaR)^2 (exact)."""
    kern, _ = logistics()
    P = kern.mh_transition(P_or_Q, target.E_lam) if is_proposal else P_or_Q
    s2 = kern.asymptotic_variance_solve(P, target.pi_lam, target.w * target.psi)
    return float(s2 / (rel * target.cvar) ** 2)


def spectral(Q, target: Target) -> dict:
    kern, _ = logistics()
    return kern.spectral_gaps(kern.mh_transition(Q, target.E_lam), target.pi_lam)
