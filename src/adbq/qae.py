"""Maximum-likelihood amplitude estimation (Suzuki et al. 2020) with an exponential schedule.

Measuring the objective qubit after Q^m A |0> gives 1 with probability
sin^2((2m + 1) theta), a = sin^2(theta). This holds for any A (Brassard et al. 2002) and
is checked against the gate-level simulator in ``tests/test_oracle.py``. The emulator
below therefore samples those Bernoulli outcomes directly: noiseless, shot-by-shot.

Query count convention: one query = one application of A or A^dagger; Q uses two, so a
shot at depth m costs 2m + 1 queries.
"""
from __future__ import annotations

import numpy as np


def schedule(k_max: int) -> np.ndarray:
    return np.array([0] + [2**j for j in range(k_max)])


def queries(ms: np.ndarray, shots: int) -> int:
    return int(shots * (2 * ms + 1).sum())


def mle_theta(ms: np.ndarray, hits: np.ndarray, shots: int, grid: int = 400) -> float:
    """Sequential maximum likelihood: each depth narrows the search window."""
    lo, hi = 0.0, np.pi / 2
    est = None
    for j in range(len(ms)):
        if est is not None:
            half = np.pi / (2 * (2 * ms[j] + 1))
            lo, hi = max(0.0, est - half), min(np.pi / 2, est + half)
        th = np.linspace(lo, hi, grid)
        ll = np.zeros_like(th)
        for m, h in zip(ms[: j + 1], hits[: j + 1]):
            p = np.clip(np.sin((2 * m + 1) * th) ** 2, 1e-300, 1 - 1e-16)
            ll += h * np.log(p) + (shots - h) * np.log1p(-p)
        i = int(np.argmax(ll))
        # local refinement
        a, b = th[max(0, i - 1)], th[min(grid - 1, i + 1)]
        th2 = np.linspace(a, b, 200)
        ll2 = np.zeros_like(th2)
        for m, h in zip(ms[: j + 1], hits[: j + 1]):
            p = np.clip(np.sin((2 * m + 1) * th2) ** 2, 1e-300, 1 - 1e-16)
            ll2 += h * np.log(p) + (shots - h) * np.log1p(-p)
        est = float(th2[np.argmax(ll2)])
    return est


def mlae(a_true: float, k_max: int, shots: int, rng) -> float:
    theta = np.arcsin(np.sqrt(a_true))
    ms = schedule(k_max)
    hits = rng.binomial(shots, np.sin((2 * ms + 1) * theta) ** 2)
    return float(np.sin(mle_theta(ms, hits, shots)) ** 2)


def cramer_rao_a(a: float, k_max: int, shots: int) -> float:
    """Cramer-Rao standard deviation of a-hat for the schedule (Suzuki et al. 2020)."""
    ms = schedule(k_max)
    fisher = shots * (4 * (2 * ms + 1) ** 2).sum()
    theta = np.arcsin(np.sqrt(a))
    return float(np.sin(2 * theta) / np.sqrt(fisher))


# ---------------------------------------------------------------------------
# Iterative QAE (Grinko, Gacon, Zoufal, Woerner 2021), Chernoff-Hoeffding variant
# ---------------------------------------------------------------------------
def _find_next_k(k: int, up: bool, th_l: float, th_u: float, r: float = 2.0):
    K_i = 4 * k + 2
    K_max = int(np.pi / (th_u - th_l))
    K = K_max - (K_max - 2) % 4
    while K >= r * K_i:
        lo, hi = (K * th_l) % (2 * np.pi), (K * th_u) % (2 * np.pi)
        if lo <= np.pi and hi <= np.pi and hi >= lo:
            return (K - 2) // 4, True
        if lo >= np.pi and hi >= np.pi and hi >= lo:
            return (K - 2) // 4, False
        K -= 4
    return k, up


def _cp_bounds(h: int, n: int, level: float) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval for a binomial proportion at failure ``level``."""
    from scipy.stats import beta
    lo = 0.0 if h == 0 else float(beta.ppf(level / 2, h, n - h + 1))
    hi = 1.0 if h == n else float(beta.ppf(1 - level / 2, h + 1, n - h))
    return lo, hi


def iqae(a_true: float, epsilon: float, rng, alpha: float = 0.05, shots: int = 100,
         max_rounds: int = 200, method: str = "ch", return_shots: bool = False):
    """Return (estimate of a, queries used, confidence interval on a).

    ``epsilon`` is the target half-width of the confidence interval on a.
    ``method``: "ch" = Chernoff-Hoeffding bounds per round (Grinko et al., Algorithm 1);
    "cp" = exact Clopper-Pearson bounds per round. Both split the failure budget alpha over
    the T rounds of the schedule (union bound), so the final interval holds with probability
    at least 1 - alpha whenever A is exact. ``return_shots=True`` appends the number of
    circuit executions (shots of Q^k A) to the returned tuple."""
    theta = np.arcsin(np.sqrt(a_true))
    T = int(np.ceil(np.log2(np.pi / (8 * epsilon)))) + 1
    eps_log = np.log(2 * T / alpha)
    k, up = 0, True
    th_l, th_u = 0.0, np.pi / 2
    n_queries = 0
    n_shots = 0
    hits_k, shots_k = 0, 0
    for _ in range(max_rounds):
        if (np.sin(th_u) ** 2 - np.sin(th_l) ** 2) / 2 <= epsilon:
            break
        k_new, up = _find_next_k(k, up, th_l, th_u)
        if k_new != k:
            hits_k, shots_k = 0, 0
        k = k_new
        K = 4 * k + 2
        h = rng.binomial(shots, np.sin((2 * k + 1) * theta) ** 2)
        n_queries += shots * (2 * k + 1)
        n_shots += shots
        hits_k += h
        shots_k += shots
        p = hits_k / shots_k
        if method == "cp":
            p_min, p_max = _cp_bounds(hits_k, shots_k, alpha / T)
        else:
            e = np.sqrt(eps_log / (2 * shots_k))
            p_min, p_max = max(0.0, p - e), min(1.0, p + e)
        if up:
            ph_min, ph_max = np.arccos(1 - 2 * p_min), np.arccos(1 - 2 * p_max)
        else:
            ph_min, ph_max = 2 * np.pi - np.arccos(1 - 2 * p_max), 2 * np.pi - np.arccos(1 - 2 * p_min)
        base = np.floor(K * th_l / (2 * np.pi)) * 2 * np.pi
        new_l, new_u = (base + ph_min) / K, (base + ph_max) / K
        th_l, th_u = max(th_l, new_l), min(th_u, new_u)
    a_l, a_u = np.sin(th_l) ** 2, np.sin(th_u) ** 2
    if return_shots:
        return float((a_l + a_u) / 2), int(n_queries), (float(a_l), float(a_u)), int(n_shots)
    return float((a_l + a_u) / 2), int(n_queries), (float(a_l), float(a_u))
