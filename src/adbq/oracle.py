"""State preparation A for the two-country model, built source by source.

    A |0> = sum_x sqrt(p(x)) |x>_sources |a, b, r1, r2>_totals ( sqrt(1 - f~(x)) |0> + sqrt(f~(x)) |1> )_obj

with f~ = f / f_max, so the probability of measuring the objective qubit in |1> is
a = mu / f_max (Woerner and Egger 2019; Egger et al. 2021).

Registers (little-endian qubit indices):
    own quake 1 (3) | own quake 2 (3) | shared quake x (3) | shared quake y (3)
    own rain 1 | own rain 2 | shared rain | total a (3) | total b (3) | r1 | r2 | objective

Steps:
 1. load each independent source (Grover-Rudolph trees; sources are a product state);
 2. a = min(4, own1 + x), b = min(4, own2 + y): reversible saturating adders;
 3. r1 = own rain 1 OR shared rain, r2 likewise;
 4. rotate the objective qubit by f~(a, b, r1, r2).

``condition_rain=True`` loads the three rain qubits from their joint law conditioned
on r1 = r2 = 1 (a necessary condition for the 'both exhausted' target) and returns the
conditioning probability; the estimate is then mu = P(r1 = r2 = 1) * f_max * a.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuits import REV, Circuit, load_distribution, value_rotation
from .model import STEPS, Scenario, payoff_arrays, two_country_sources

O1, O2 = [0, 1, 2], [3, 4, 5]
SX, SY = [6, 7, 8], [9, 10, 11]
U1, U2, Z = 12, 13, 14
A_REG, B_REG = [15, 16, 17], [18, 19, 20]
R1, R2 = 21, 22
OBJ = 23
N_QUBITS = 24
TOTALS = A_REG + B_REG + [R1, R2]


@dataclass
class Oracle:
    A: Circuit
    f_max: float
    prefactor: float          # mu = prefactor * f_max * a
    key: str

    def good(self, x: int) -> bool:
        return bool((x >> OBJ) & 1)


def build_oracle(s: Scenario, key: str, condition_rain: bool = False,
                 payoff: np.ndarray | None = None) -> Oracle:
    """``payoff`` overrides the target table: any nonnegative f on the (5, 5, 2, 2) term
    grid (e.g. the CDF indicator 1{L <= x} or the excess (L - v)+ used for VaR / CVaR);
    ``key`` is then only a label."""
    src = two_country_sources(s)
    c = Circuit(N_QUBITS)
    # 1. sources
    c.extend(load_distribution(O1, src.own_quake[0]))
    c.extend(load_distribution(O2, src.own_quake[1]))
    shared = np.zeros(64)                      # x in low 3 bits, y in high 3 bits
    for (x, y), p in np.ndenumerate(src.shared_quake):
        shared[x + 8 * y] = p
    c.extend(load_distribution(SX + SY, shared))
    rain = np.zeros(8)                         # bits (u1, u2, z)
    for v in range(8):
        u1, u2, z = v & 1, (v >> 1) & 1, (v >> 2) & 1
        rain[v] = ((src.own_rain[0] if u1 else 1 - src.own_rain[0])
                   * (src.own_rain[1] if u2 else 1 - src.own_rain[1])
                   * (src.shared_rain if z else 1 - src.shared_rain))
    prefactor = 1.0
    if condition_rain:
        keep = np.array([((v & 1) | (v >> 2)) & (((v >> 1) & 1) | (v >> 2)) for v in range(8)], bool)
        prefactor = float(rain[keep].sum())
        rain = np.where(keep, rain, 0.0)
    c.extend(load_distribution([U1, U2, Z], rain))
    # 2-3. arithmetic
    sat = lambda v: min(STEPS, (v & 7) + (v >> 3))
    c.add(REV(tuple(O1 + SX), tuple(A_REG), sat, "a = min(4, own1 + x)"))
    c.add(REV(tuple(O2 + SY), tuple(B_REG), sat, "b = min(4, own2 + y)"))
    c.add(REV((U1, Z), (R1,), lambda v: (v & 1) | (v >> 1), "r1 = u1 OR z"))
    c.add(REV((U2, Z), (R2,), lambda v: (v & 1) | (v >> 1), "r2 = u2 OR z"))
    # 4. objective rotation on (a, b, r1, r2)
    f = payoff_arrays(s)[key] if payoff is None else np.asarray(payoff, float)
    if f.shape != (STEPS + 1, STEPS + 1, 2, 2) or (f < 0).any() or f.max() <= 0:
        raise ValueError("payoff must be nonnegative on the (5, 5, 2, 2) grid and not all zero")
    f_max = float(f.max())
    table = np.zeros(2**8)
    for (a, b, r1, r2), val in np.ndenumerate(f):
        table[a + 8 * b + 64 * r1 + 128 * r2] = val / f_max
    c.extend(value_rotation(TOTALS, OBJ, table))
    return Oracle(c, f_max, prefactor, key)
