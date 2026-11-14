"""Classical sampling baselines from the brief (step 9) for the two-country model.

A term is a path of three annual states drawn from the 100-state annual law. One
*query* = one sampled term path = one evaluation of the payoff f.

For Monte Carlo, stratified and importance sampling the per-query variance is computed
exactly over all 100^3 paths, so their RMSE curves carry no simulation noise. Each
estimator is also implemented as a sampler, and ``tests/`` checks the sampler against
the exact variance. Scrambled Sobol has no closed-form variance and is measured
empirically.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from .model import CAP, STEPS, Scenario, annual_distribution, payoff_arrays

SHAPE = (STEPS + 1, STEPS + 1, 2, 2)
NA = int(np.prod(SHAPE))  # 100 annual states


class PathSpace:
    """Annual law, per-path payoff tensors and the brief's proposals."""

    def __init__(self, s: Scenario):
        self.s = s
        self.p = annual_distribution(s).reshape(-1)
        self.p = self.p / self.p.sum()
        states = np.array(np.unravel_index(np.arange(NA), SHAPE)).T            # (100, 4)
        term = np.minimum(states[:, None, None, :] + states[None, :, None, :] + states[None, None, :, :], CAP)
        self.term_index = np.ravel_multi_index(term.reshape(-1, 4).T, SHAPE).reshape(NA, NA, NA)
        self.f = {k: v.reshape(-1)[self.term_index] for k, v in payoff_arrays(s).items()}
        # annual principal loss of each annual state (for the loss-tilted proposal)
        a, b, r1, r2 = states.T
        self.ann_loss = s.quake_limit * (a + b) / STEPS + s.rain_limit * (r1 + r2)
        tilt = self.p * self.ann_loss
        self.q_is = 0.5 * self.p + 0.5 * tilt / tilt.sum()
        self.p0 = self.p[0]

    # ---------------- exact per-query variances ----------------
    def mu(self, key: str) -> float:
        P = np.einsum("i,j,k->ijk", self.p, self.p, self.p)
        return float((P * self.f[key]).sum())

    def var_mc(self, key: str) -> float:
        P = np.einsum("i,j,k->ijk", self.p, self.p, self.p)
        m = (P * self.f[key]).sum()
        return float((P * self.f[key] ** 2).sum() - m**2)

    def var_is(self, key: str) -> float:
        Q = np.einsum("i,j,k->ijk", self.q_is, self.q_is, self.q_is)
        P = np.einsum("i,j,k->ijk", self.p, self.p, self.p)
        w = np.divide(P, Q, out=np.zeros_like(P), where=Q > 0)
        m = (P * self.f[key]).sum()
        return float((Q * (self.f[key] * w) ** 2).sum() - m**2)

    def strata(self):
        """Stratum k = first year with any payout (k = 1, 2, 3); weights p0^(k-1)(1-p0)."""
        cond = self.p.copy(); cond[0] = 0; cond /= cond.sum()
        zero = np.zeros(NA); zero[0] = 1
        laws = [(cond, self.p, self.p), (zero, cond, self.p), (zero, zero, cond)]
        weights = [self.p0 ** (k) * (1 - self.p0) for k in range(3)]
        return laws, weights

    def var_strat(self, key: str, budget: int) -> float:
        """Variance of the stratified estimator with proportional allocation."""
        laws, wts = self.strata()
        tot = sum(wts)
        v = 0.0
        for (l1, l2, l3), wk in zip(laws, wts):
            Pk = np.einsum("i,j,k->ijk", l1, l2, l3)
            m = (Pk * self.f[key]).sum()
            var_k = (Pk * self.f[key] ** 2).sum() - m**2
            nk = max(1, budget * wk / tot)
            v += wk**2 * var_k / nk
        return float(v)

    # ---------------- samplers ----------------
    def _draw(self, law: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.minimum(np.searchsorted(np.cumsum(law), u, side="right"), NA - 1)

    def estimate_mc(self, key: str, n: int, rng) -> float:
        u = rng.random((n, 3))
        i, j, k = (self._draw(self.p, u[:, d]) for d in range(3))
        return float(self.f[key][i, j, k].mean())

    def estimate_sobol(self, key: str, n: int, rng, scrambles: int = 8) -> float:
        per = max(1, n // scrambles)
        ests = []
        for _ in range(scrambles):
            eng = qmc.Sobol(d=3, scramble=True, seed=rng)
            m = int(np.log2(per))
            u = eng.random_base2(m) if 2**m == per else eng.random(per)
            i, j, k = (self._draw(self.p, u[:, d]) for d in range(3))
            ests.append(self.f[key][i, j, k].mean())
        return float(np.mean(ests))

    def estimate_strat(self, key: str, n: int, rng) -> float:
        laws, wts = self.strata()
        tot = sum(wts)
        est = 0.0
        for (l1, l2, l3), wk in zip(laws, wts):
            nk = max(1, int(round(n * wk / tot)))
            u = rng.random((nk, 3))
            i, j, k = self._draw(l1, u[:, 0]), self._draw(l2, u[:, 1]), self._draw(l3, u[:, 2])
            est += wk * self.f[key][i, j, k].mean()
        return float(est)

    def estimate_is(self, key: str, n: int, rng) -> float:
        u = rng.random((n, 3))
        i, j, k = (self._draw(self.q_is, u[:, d]) for d in range(3))
        w = (self.p[i] * self.p[j] * self.p[k]) / (self.q_is[i] * self.q_is[j] * self.q_is[k])
        return float((self.f[key][i, j, k] * w).mean())
