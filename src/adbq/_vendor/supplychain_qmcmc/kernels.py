"""Proposal kernels, exact Metropolis-Hastings transition matrices and spectral diagnostics.

Convention (as in Notebooks 13-14): row = current state, column = next state, rows sum to 1.
Every proposal here is symmetric, Q[s, s'] = Q[s', s], so the MH acceptance reduces to
min(1, exp(-beta * [E(s') - E(s)])) and never needs the partition function.
"""

import numpy as np

from . import mrf

TOL = 1e-12


def local_proposal(n=mrf.N_BITS):
    """Flip one uniformly chosen bit (Notebook 13)."""
    N = 2**n
    Q = np.zeros((N, N))
    ids = np.arange(N)
    for k in range(n):
        Q[ids, ids ^ (1 << (n - 1 - k))] = 1.0 / n
    return Q


def uniform_proposal(n=mrf.N_BITS):
    """Propose any configuration uniformly at random (the 'fully depolarized' limit)."""
    N = 2**n
    return np.full((N, N), 1.0 / N)


def mixture(weights_and_kernels):
    """Pick kernel k with probability w_k at every step; symmetric if every kernel is."""
    total = sum(w for w, _ in weights_and_kernels)
    if not np.isclose(total, 1.0):
        raise ValueError("Mixture weights must sum to one")
    return sum(w * Q for w, Q in weights_and_kernels)


def symmetry_residual(Q):
    return float(np.abs(Q - Q.T).max())


def mh_transition(Q, energies, beta=1.0):
    """Exact MH matrix for a symmetric proposal Q and target exp(-beta E)."""
    Q = np.asarray(Q, dtype=float)
    if symmetry_residual(Q) > 1e-10:
        raise ValueError("The simplified acceptance rule needs a symmetric proposal")
    e = beta * np.asarray(energies, dtype=float)
    acceptance = np.exp(-np.clip(e[None, :] - e[:, None], 0, None))
    P = Q * acceptance
    np.fill_diagonal(P, 0.0)
    rejection = 1.0 - P.sum(axis=1)
    if rejection.min() < -1e-10:
        raise ValueError("Negative rejection mass")
    P[np.diag_indices_from(P)] = np.clip(rejection, 0.0, None)
    return P


def audit(P, pi):
    """Row-sum, detailed-balance and stationarity residuals."""
    flux = pi[:, None] * P
    return {
        "row_sum": float(np.abs(P.sum(axis=1) - 1).max()),
        "detailed_balance": float(np.abs(flux - flux.T).max()),
        "stationarity": float(np.abs(pi @ P - pi).max()),
    }


def _symmetrized(P, pi):
    s = np.sqrt(pi)
    S = s[:, None] * P / s[None, :]
    return (S + S.T) / 2


def spectrum(P, pi):
    """Eigenvalues (descending) and orthonormal eigenvectors of D^1/2 P D^-1/2."""
    w, V = np.linalg.eigh(_symmetrized(P, pi))
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def spectral_gaps(P, pi):
    """Ordinary gap 1 - lambda_2, absolute gap 1 - max_k>=2 |lambda_k| and relaxation time."""
    w = np.sort(np.linalg.eigvalsh(_symmetrized(P, pi)))[::-1]
    absolute = 1.0 - max(abs(w[1]), abs(w[-1]))
    return {"gap": float(1.0 - w[1]), "absolute_gap": float(absolute), "relaxation_time": float(1.0 / absolute)}


def asymptotic_variance(P, pi, g, spectral=None):
    """lim N Var(mean of g(B_t)) for the stationary reversible chain (exact, all lags summed).

    Uses gamma(t) = sum_k c_k^2 lambda_k^t, so sigma^2 = sum_k c_k^2 (1 + lambda_k) / (1 - lambda_k).
    ``g`` may be a vector or a (states, observables) matrix.
    """
    w, V = spectral if spectral is not None else spectrum(P, pi)
    g = np.asarray(g, dtype=float)
    centered = g - pi @ g
    c = V.T @ (np.sqrt(pi)[:, None] * centered.reshape(len(pi), -1))
    keep = w < 1 - 1e-10
    sigma2 = ((c[keep] ** 2) * ((1 + w[keep]) / (1 - w[keep]))[:, None]).sum(axis=0)
    return sigma2 if g.ndim > 1 else float(sigma2[0])


def iact(P, pi, f, spectral=None):
    """Exact integrated autocorrelation time tau_f = sigma_f^2 / Var_pi(f)."""
    f = np.asarray(f, dtype=float)
    var = pi @ (f - pi @ f) ** 2
    return asymptotic_variance(P, pi, f, spectral) / var


def run_chain(Q, energies, start, n_steps, rng, beta=1.0):
    """Simulate MH with a tabulated symmetric proposal (an exact emulator of an ideal QPU).

    Each step draws s' from row Q[s] - exactly what one shot of the ideal proposal circuit
    returns - then applies the classical accept/reject rule. Rejections are recorded as
    repeated states. Returns (post-transition state IDs, proposed IDs, accepted flags).
    """
    Q = np.asarray(Q)
    cdf = np.cumsum(Q, axis=1)
    cdf[:, -1] = 1.0
    e = beta * np.asarray(energies, dtype=float)
    states = np.empty(n_steps, dtype=np.int64)
    proposed = np.empty(n_steps, dtype=np.int64)
    accepted = np.empty(n_steps, dtype=bool)
    current = int(start)
    u_prop, u_acc = rng.random(n_steps), rng.random(n_steps)
    for t in range(n_steps):
        candidate = int(np.searchsorted(cdf[current], u_prop[t], side="right"))
        ok = u_acc[t] < np.exp(-max(e[candidate] - e[current], 0.0))
        if ok:
            current = candidate
        states[t], proposed[t], accepted[t] = current, candidate, ok
    return states, proposed, accepted


def run_chains(Q, energies, starts, n_steps, rng, beta=1.0, return_proposals=False):
    """Vectorized ``run_chain`` for many independent chains (one per entry of ``starts``).

    Returns an (n_chains, n_steps) array of post-transition state IDs and the acceptance
    fraction of each chain (self-proposals count as accepted, as they leave the state unchanged).
    With ``return_proposals=True`` the proposed IDs are returned as a third array; under a
    loss-tilted target every proposed state needs a loss evaluation, accepted or not.
    """
    Q = np.asarray(Q)
    cdf = np.cumsum(Q, axis=1)
    cdf[:, -1] = 1.0
    e = beta * np.asarray(energies, dtype=float)
    current = np.asarray(starts, dtype=np.int64).copy()
    R = len(current)
    out = np.empty((R, n_steps), dtype=np.int64)
    proposed = np.empty((R, n_steps), dtype=np.int64) if return_proposals else None
    accepted = np.zeros(R)
    for t in range(n_steps):
        u = rng.random(R)
        candidate = (cdf[current] <= u[:, None]).sum(axis=1)
        ok = rng.random(R) < np.exp(-np.clip(e[candidate] - e[current], 0, None))
        current = np.where(ok, candidate, current)
        accepted += ok
        out[:, t] = current
        if return_proposals:
            proposed[:, t] = candidate
    if return_proposals:
        return out, accepted / n_steps, proposed
    return out, accepted / n_steps


def asymptotic_variance_solve(P, pi, g):
    """Same quantity as ``asymptotic_variance`` via one linear solve (faster for a single observable).

    With Z = (I - P + 1 pi^T)^-1 the fundamental matrix and g centred under pi,
    sigma^2 = 2 <g, Z g>_pi - <g, g>_pi.
    """
    from scipy.linalg import solve

    g = np.asarray(g, dtype=float) - pi @ g
    A = np.eye(len(pi)) - P + np.outer(np.ones(len(pi)), pi)
    x = solve(A, g, check_finite=False)
    return float(2 * (pi * g) @ x - pi @ g**2)


def pair_flip_proposal(edges, n):
    """Flip both endpoints of a uniformly chosen coupling (symmetric, structure-aware classical move)."""
    N = 2**n
    Q = np.zeros((N, N))
    ids = np.arange(N)
    for i, j, _J in edges:
        Q[ids, ids ^ ((1 << (n - 1 - i)) | (1 << (n - 1 - j)))] += 1.0 / len(edges)
    return Q


def block_flip_proposal(n, sizes=(2, 3, 4)):
    """Flip a uniformly chosen set of k random bits, k uniform over ``sizes`` (generic multi-bit classical move)."""
    from itertools import combinations

    N = 2**n
    Q = np.zeros((N, N))
    ids = np.arange(N)
    for k in sizes:
        combos = list(combinations(range(n), k))
        for c in combos:
            Q[ids, ids ^ sum(1 << (n - 1 - x) for x in c)] += 1.0 / (len(sizes) * len(combos))
    return Q


def _propose(kind, current, rng, n, masks=None, cdf=None):
    """Vectorised proposals for one move type over an array of current states."""
    R = len(current)
    if kind == "local":
        return current ^ (1 << (n - 1 - rng.integers(n, size=R)))
    if kind == "pair":
        return current ^ masks[rng.integers(len(masks), size=R)]
    if kind == "uniform":
        return rng.integers(2**n, size=R)
    u = rng.random(R)                                   # tabulated proposal matrix (e.g. an ideal quantum ensemble)
    return np.array([np.searchsorted(cdf[c], x, side="right") for c, x in zip(current, u)], dtype=np.int64)


def run_mixture_chains(moves, energies, starts, n_steps, rng, n, edges=None, beta=1.0):
    """Independent MH chains whose proposal is a mixture of move types.

    ``moves`` is a list of (weight, kind) with kind in {"local", "pair", "uniform"} or a symmetric proposal matrix.
    Exactly equivalent to ``run_chains`` with the mixed matrix, but much faster for the cheap classical moves.
    Returns (states (chains, steps), acceptance fraction per chain).
    """
    weights = np.array([w for w, _ in moves], dtype=float)
    weights /= weights.sum()
    masks = None if edges is None else np.array([(1 << (n - 1 - i)) | (1 << (n - 1 - j)) for i, j, _ in edges])
    cdfs = []
    for _, kind in moves:
        if isinstance(kind, str):
            cdfs.append(None)
        else:
            c = np.cumsum(np.asarray(kind, dtype=float), axis=1)
            c[:, -1] = 1.0
            cdfs.append(c)
    e = beta * np.asarray(energies, dtype=float)
    current = np.asarray(starts, dtype=np.int64).copy()
    R = len(current)
    out = np.empty((R, n_steps), dtype=np.int64)
    accepted = np.zeros(R)
    for t in range(n_steps):
        pick = rng.choice(len(moves), size=R, p=weights)
        candidate = current.copy()
        for m, (_, kind) in enumerate(moves):
            sel = np.flatnonzero(pick == m)
            if len(sel):
                candidate[sel] = _propose(kind if isinstance(kind, str) else "matrix", current[sel], rng, n, masks,
                                          cdfs[m])
        ok = rng.random(R) < np.exp(-np.clip(e[candidate] - e[current], 0, None))
        current = np.where(ok, candidate, current)
        accepted += ok
        out[:, t] = current
    return out, accepted / n_steps


def run_parallel_tempering(moves, energies, betas, starts, n_sweeps, rng, n, edges=None):
    """Parallel tempering: one mixture-move MH chain per inverse temperature, neighbour swaps every sweep.

    ``betas`` must start at 1.0 (the target). Returns the target (beta = 1) chain, shape (chains, sweeps), and the
    swap acceptance rate per neighbouring pair. Each sweep costs ``len(betas)`` energy (loss) evaluations.
    """
    betas = np.asarray(betas, dtype=float)
    if betas[0] != 1.0:
        raise ValueError("betas[0] must be 1.0 (the target distribution)")
    e = np.asarray(energies, dtype=float)
    K, R = len(betas), len(starts)
    state = np.tile(np.asarray(starts, dtype=np.int64), (K, 1))
    cold = np.empty((R, n_sweeps), dtype=np.int64)
    swaps = np.zeros(K - 1)
    for t in range(n_sweeps):
        for k in range(K):
            new, _ = run_mixture_chains(moves, e, state[k], 1, rng, n, edges, beta=betas[k])
            state[k] = new[:, 0]
        for k in range(K - 1):
            a = (betas[k] - betas[k + 1]) * (e[state[k]] - e[state[k + 1]])
            ok = rng.random(R) < np.exp(np.clip(a, None, 0))
            tmp = state[k][ok].copy()
            state[k][ok] = state[k + 1][ok]
            state[k + 1][ok] = tmp
            swaps[k] += ok.mean()
        cold[:, t] = state[0]
    return cold, swaps / n_sweeps
