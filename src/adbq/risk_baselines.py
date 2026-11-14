"""Classical estimators of VaR_alpha and CVaR_alpha for the two-country model.

Every estimator returns a *weighted sample of terms*: a weight per loss level that
estimates the loss PMF. VaR and CVaR are then read off with ``risk.var_cvar``
(weighted lower quantile; Rockafellar-Uryasev CVaR). For importance sampling the weights
are self-normalised. One query = one sampled three-year path, as in ``baselines.py``.

Estimators
----------
* ``mc``      plain Monte Carlo under p;
* ``sobol``   scrambled Sobol in [0,1)^3 (8 scrambles pooled into one weighted sample);
* ``strat``   the brief's three strata (first payout year), proportional allocation;
              the all-zero path (probability p0^3, loss 0) is known exactly;
* ``is``      the brief's proposal q = (p + p * loss / E[loss]) / 2 per year;
* ``tilt``    loss-tilted IS tuned for CVaR: q_theta(x) = (1 - lam) p(x) e^{theta l(x)} / Z
              + lam p(x) per year, (theta, lam) chosen on a grid to minimise the exact
              asymptotic variance of the CVaR estimator *at the true VaR*. This tuning
              uses the answer, so the tilted baseline is optimistic for the classical side.

Exact asymptotic variances (``cvar_asymptotic_var``) hold once the VaR atom is
identified, which happens exponentially fast in N when F(VaR-) < alpha < F(VaR); the
empirical RMSE (experiments/risk_accuracy.py) includes the small-N misidentification.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from .baselines import NA, PathSpace
from .model import Scenario
from .risk import loss_pmf_two_country, var_cvar

METHODS = ("mc", "sobol", "strat", "is", "tilt")


class RiskPathSpace(PathSpace):
    """Path space with path-level losses on the integer lattice and tail proposals."""

    def __init__(self, s: Scenario):
        super().__init__(s)
        self.values, self.pmf = loss_pmf_two_country(s)
        self.unit = float(self.values[1])
        self.L = np.rint(self.f["term_expected_principal_loss_usd"] / self.unit).astype(int)  # (NA,)*3
        self.Lflat = self.L.reshape(-1)
        self.P = np.einsum("i,j,k->ijk", self.p, self.p, self.p).reshape(-1)
        self.nb = len(self.values)
        self._cache = {}

    # ---------------- proposals ----------------
    def tilt_law(self, theta: float, lam: float) -> np.ndarray:
        """Per-year proposal; theta in 1 / (USD 10m) of annual principal loss."""
        e = self.p * np.exp(theta * self.ann_loss / 10e6)
        return (1 - lam) * e / e.sum() + lam * self.p

    def product(self, q: np.ndarray) -> np.ndarray:
        return np.einsum("i,j,k->ijk", q, q, q).reshape(-1)

    # ---------------- exact asymptotic variances ----------------
    def cvar_asymptotic_var(self, method: str, alpha: float, q: np.ndarray | None = None) -> float:
        """N * Var(CVaR-hat) as N -> infinity (VaR identified): for g = (L - VaR)+,
        MC Var_p(g); self-normalised IS E_q[w^2 (g - m)^2]; strat sum_k W_k Var_k(g)
        (proportional allocation over the strata's total weight). Divided by (1-alpha)^2."""
        v, _ = var_cvar(self.values, self.pmf, alpha)
        g = np.maximum(self.Lflat * self.unit - v, 0.0)
        m = float(self.P @ g)
        if method == "mc":
            out = float(self.P @ (g - m) ** 2)
        elif method in ("is", "tilt"):
            if q is None:
                q = self.q_is if method == "is" else self.tuned_tilt(alpha)[0]
            Q = self.product(q)
            w = np.divide(self.P, Q, out=np.zeros_like(self.P), where=Q > 0)
            out = float(Q @ (w * (g - m)) ** 2)
        elif method == "strat":
            laws, wts = self.strata()
            tot = sum(wts)
            G = g.reshape((NA,) * 3)
            out = 0.0
            for (l1, l2, l3), wk in zip(laws, wts):
                Pk = np.einsum("i,j,k->ijk", l1, l2, l3)
                mk = (Pk * G).sum()
                out += wk * tot * ((Pk * G**2).sum() - mk**2)
        else:
            raise ValueError(method)
        return out / (1 - alpha) ** 2

    def tuned_tilt(self, alpha: float, thetas=np.linspace(0.0, 3.0, 31),
                   lams=(0.0, 0.05, 0.1, 0.2, 0.5)) -> tuple[np.ndarray, float, float]:
        """(q, theta, lam) minimising the exact asymptotic CVaR variance on a grid."""
        if alpha not in self._cache:
            best = None
            for th in thetas:
                for lam in lams:
                    q = self.tilt_law(th, lam)
                    v = self.cvar_asymptotic_var("tilt", alpha, q)
                    if best is None or v < best[0]:
                        best = (v, q, float(th), float(lam))
            self._cache[alpha] = best[1:]
        return self._cache[alpha]

    # ---------------- weighted samples -> loss PMF estimate ----------------
    # The estimators only see (loss, weight) of each sampled path, so sampling paths is
    # equal in law to sampling the distinct (loss, weight) atoms with their total proposal
    # mass. ``_atoms`` groups the 10^6 paths once; each replication is then a multinomial
    # over a few hundred atoms instead of 10^6 paths. tests/test_risk_baselines.py checks
    # the grouped sampler against the exact asymptotic variances.
    def _atoms(self, key, Q: np.ndarray, w: np.ndarray):
        if key not in self._cache:
            wr = np.round(np.log(np.where(w > 0, w, 1.0)), 10)
            uniq, inv = np.unique(np.stack([self.Lflat, wr]), axis=1, return_inverse=True)
            mass = np.bincount(inv.reshape(-1), weights=Q)
            wmean = np.bincount(inv.reshape(-1), weights=Q * w) / mass
            self._cache[key] = (uniq[0].astype(int), wmean, mass / mass.sum())
        return self._cache[key]

    def _stratum_pmfs(self):
        if "strata" not in self._cache:
            laws, wts = self.strata()
            pm = [np.bincount(self.Lflat, weights=np.einsum("i,j,k->ijk", *l).reshape(-1),
                              minlength=self.nb) for l in laws]
            self._cache["strata"] = ([x / x.sum() for x in pm], wts)
        return self._cache["strata"]

    def sample_pmf(self, method: str, n: int, rng, alpha: float | None = None) -> np.ndarray:
        """Estimated loss PMF on the lattice from n queries (IS: unnormalised weights)."""
        if method == "mc":
            return rng.multinomial(n, self.pmf / self.pmf.sum()) / n
        if method in ("is", "tilt"):
            q = self.q_is if method == "is" else self.tuned_tilt(alpha)[0]
            key = ("is",) if method == "is" else ("tilt", alpha)
            if key not in self._cache:
                Q = self.product(q)
                w = np.divide(self.P, Q, out=np.zeros_like(self.P), where=Q > 0)
                self._atoms(key, Q, w)
            L, w, mass = self._cache[key]
            c = rng.multinomial(n, mass)
            return np.bincount(L, weights=c * w, minlength=self.nb) / n
        if method == "strat":
            pms, wts = self._stratum_pmfs()
            tot = sum(wts)
            est = np.zeros(self.nb)
            est[0] += 1 - tot                                   # the all-zero path, exactly
            for pk, wk in zip(pms, wts):
                nk = max(1, int(round(n * wk / tot)))
                est += wk * rng.multinomial(nk, pk) / nk
            return est
        if method == "sobol":
            scr = 8
            per = max(1, n // scr)
            m = int(np.log2(per))
            est = np.zeros(self.nb)
            for _ in range(scr):
                eng = qmc.Sobol(d=3, scramble=True, seed=rng)
                u = eng.random_base2(m) if 2**m == per else eng.random(per)
                i, j, k = (self._draw(self.p, u[:, d]) for d in range(3))
                est += np.bincount(self.L[i, j, k], minlength=self.nb)
            return est / est.sum()
        raise ValueError(method)

    def estimate(self, method: str, n: int, rng, alpha: float) -> tuple[float, float]:
        """(VaR-hat, CVaR-hat) in USD from n queries."""
        return var_cvar(self.values, self.sample_pmf(method, n, rng, alpha), alpha)
