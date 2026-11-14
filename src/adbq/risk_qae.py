"""Quantum estimation of VaR_alpha and CVaR_alpha (Woerner and Egger 2019) with IQAE.

VaR: bisection over the loss support l_0 < l_1 < ... < l_m. Each step asks whether
F(l_mid) >= alpha, using IQAE on the CDF indicator f(x) = 1{L(x) <= l_mid} (amplitude
a = F(l_mid)). The decision is taken only when the IQAE confidence interval excludes
alpha; otherwise IQAE is rerun with half the target half-width (every rerun is counted).
Below ``eps_min`` the point estimate decides. Each IQAE call runs at confidence
1 - conf / (ceil(log2(m + 1)) x reruns), a union bound over the bisection steps and over the
halving reruns within a step (reruns = ceil(log2(eps0 / eps_min)) + 1).

CVaR: with v = VaR-hat, IQAE on the excess payoff g = (L - v)+ / (L_max - v) gives
mu-hat = (L_max - v) a-hat and CVaR-hat = v + mu-hat / (1 - alpha). This equals the true
CVaR when v is the true VaR (Rockafellar-Uryasev minimum); if v is a wrong atom the RU
function is >= CVaR, so a VaR miss shows up as a positive bias, which the RMSE includes.

Query convention as in qae.py: one query = one application of A or A^dagger. The
emulator samples the Bernoulli outcomes of Q^k A|0> for the exact amplitude, which
tests/test_risk_qae.py checks against the gate-level oracle for these payoffs.

The loss support is taken as the set of lattice levels with positive probability; it is
fixed by the contract terms (quake steps and rain limits), not by the probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, log2

import numpy as np

from .qae import iqae


def cdf_amplitude(values, pmf, x: float) -> float:
    return float(np.asarray(pmf)[np.asarray(values) <= x].sum())


def excess_amplitude(values, pmf, v: float) -> tuple[float, float]:
    """(a, f_max) for g = (L - v)+: a = E[(L - v)+] / f_max with f_max = L_max - v."""
    values, pmf = np.asarray(values), np.asarray(pmf)
    f_max = float(values[pmf > 0].max() - v)
    if f_max <= 0:
        return 0.0, 0.0
    return float(np.dot(pmf, np.maximum(values - v, 0.0)) / f_max), f_max


@dataclass
class QuantumVaR:
    var: float
    queries: int
    steps: list = field(default_factory=list)    # (level, final eps, decision, queries)


def quantum_var(values, pmf, alpha: float, rng, eps0: float = 0.05, eps_min: float = 1e-7,
                conf: float = 0.01, shots: int = 100, support=None) -> QuantumVaR:
    values, pmf = np.asarray(values, float), np.asarray(pmf, float)
    sup = np.sort(values[pmf > 0]) if support is None else np.sort(np.asarray(support, float))
    # union bound over bisection steps and, within a step, over the halving reruns
    reruns = max(1, ceil(log2(eps0 / eps_min)) + 1)
    conf_step = conf / max(1, ceil(log2(len(sup)))) / reruns
    lo, hi = -1, len(sup) - 1                    # F(sup[hi]) = 1 >= alpha; F(sup[-1]) := 0
    total, log = 0, []
    while hi - lo > 1:
        mid = (lo + hi) // 2
        a = cdf_amplitude(values, pmf, sup[mid])
        eps, used = eps0, 0
        while True:
            est, q, (al, au) = iqae(a, eps, rng, alpha=conf_step, shots=shots)
            used += q
            if al >= alpha:
                yes = True; break
            if au < alpha:
                yes = False; break
            if eps <= eps_min:
                yes = est >= alpha; break
            eps /= 2
        total += used
        log.append((float(sup[mid]), eps, yes, used))
        if yes:
            hi = mid
        else:
            lo = mid
    return QuantumVaR(float(sup[hi]), total, log)


@dataclass
class QuantumRisk:
    var: float
    cvar: float
    queries_var: int
    queries_cvar: int

    @property
    def queries(self) -> int:
        return self.queries_var + self.queries_cvar


def matched_eps_min(values, pmf, alpha: float, eps_rel_cvar: float) -> float:
    """Bisection floor that keeps a VaR miss within the CVaR stage's own error.

    Between adjacent atoms the RU function has slope (F - alpha) / (1 - alpha), so picking
    a neighbouring atom v' with |F(v') - alpha| <= eta costs at most eta * dl / (1 - alpha)
    in CVaR (dl = atom spacing). The CVaR stage's half-width is eps_rel * mu / (1 - alpha)
    with mu = E[(L - VaR)+]; equating gives eta = eps_rel * mu / dl. As in the IQAE
    emulation, the sweep parameter is set from the exact mu (it fixes the schedule only).
    """
    values, pmf = np.asarray(values, float), np.asarray(pmf, float)
    sup = np.sort(values[pmf > 0])
    dl = float(np.diff(sup).max())
    from .risk import var_cvar
    v, _ = var_cvar(values, pmf, alpha)
    mu = float(np.dot(pmf, np.maximum(values - v, 0.0)))
    return max(1e-7, eps_rel_cvar * mu / dl)


def quantum_cvar(values, pmf, alpha: float, eps_rel_cvar: float, rng, conf: float = 0.01,
                 var_mode: str = "certify", shots: int = 100) -> QuantumRisk:
    """VaR by bisection, then CVaR. ``eps_rel_cvar`` is the IQAE target half-width on the
    excess amplitude, relative to that amplitude.

    var_mode = "certify": every bisection decision is taken at confidence 1 - conf/steps
    (cost set by the CDF gap at alpha); "matched": decisions stop refining at
    ``matched_eps_min`` (a wrong atom then costs no more than the CVaR stage's error)."""
    eps_min = 1e-7 if var_mode == "certify" else matched_eps_min(values, pmf, alpha, eps_rel_cvar)
    qv = quantum_var(values, pmf, alpha, rng, conf=conf, shots=shots, eps_min=eps_min)
    a, f_max = excess_amplitude(values, pmf, qv.var)
    if a == 0:
        return QuantumRisk(qv.var, qv.var, qv.queries, 0)
    est, q, _ = iqae(a, eps_rel_cvar * a, rng, alpha=conf, shots=shots)
    return QuantumRisk(qv.var, qv.var + f_max * est / (1 - alpha), qv.queries, q)
