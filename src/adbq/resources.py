"""Logical resource estimates for one application of A on an n-country chain portfolio.

These are *estimates from standard constructions*, not a compiled circuit. Each line
names the construction it assumes, so a Phase 2 compilation (Qualtran, Classiq or
pytket) can replace any line with a measured count.

Constructions assumed
---------------------
* Loading an m-qubit source register: Grover-Rudolph tree of uniformly controlled RY
  rotations, Moettoenen decomposition: 2^m - 2 CNOT and 2^m - 1 single-qubit RY.
* Out-of-place addition of a b-bit register: Gidney (2018) adder, b - 1 Toffoli
  (temporary-AND) to compute; uncomputation by measurement costs no Toffolis.
* Saturation min(4, v) for v < 16: 3 Toffoli (OR of high bits, two masked copies).
* Constant multiply-and-accumulate of a country loss into the portfolio register:
  two controlled constant additions, 2B Toffoli for a B-bit register.
* Comparator L > T: B Toffoli. Counter capped at 2: 2 Toffoli per increment.
* Objective rotation for value targets (E[L], stop-loss): uniformly controlled RY over
  the B-bit loss register, 2^B CNOT + 2^B RY; indicator targets need one Toffoli.
* Each arbitrary RY synthesised to precision delta costs about 3 log2(1/delta) T gates
  (Ross and Selinger 2016); a Toffoli costs 4 T with temporary-AND tricks (Gidney 2018).

Every gate is counted once for A; Q = A S0 A^dagger S_chi costs 2 A plus reflections.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil, log2

from .model import STEPS


@dataclass
class Resources:
    n: int
    target: str
    logical_qubits: int
    toffoli: int
    cnot: int
    rotations: int
    t_count: int
    toffoli_depth: int

    def as_dict(self):
        return asdict(self)


def estimate(n: int, target: str = "P(>=2 exhausted)", loss_units_full: int = 32,
             rot_precision: float = 1e-10,
             edges: int | None = None, degree: int = 2) -> Resources:
    """One application of A. ``edges`` (default n - 1, the chain) and ``degree`` (shared
    streams per country: 2 on the chain, up to 4 on a grid) generalise to other graphs."""
    edges = n - 1 if edges is None else edges
    B = ceil(log2(loss_units_full * n + 1))                    # portfolio loss register
    # --- qubits ---
    src_q = n * (3 + 1) + edges * (6 + 1)                      # own quake/rain, shared quake/rain
    tot_q = n * (3 + 1)                                        # a_i, r_i
    work_q = 5 + 5                                             # adder partial sums (reused)
    loss_q = B + B                                             # running loss + one carry/ancilla row
    count_q = 2 + 2                                            # >=2 pay, >=2 exhausted counters
    qubits = src_q + tot_q + work_q + loss_q + count_q + 1     # + objective
    # --- loading ---
    cnot = n * (2**3 - 2) + edges * (2**6 - 2)
    rot = n * (2**3 - 1) + edges * (2**6 - 1) + n + edges      # + rain RY
    # --- arithmetic per country ---
    tof = 0
    per_country = 0
    per_country += 2 + 3 * (degree - 1)                        # own + each incident edge (3-bit adds)
    per_country += 3                                           # saturate to 0..4
    per_country += degree                                      # rain OR of up to degree + 1 bits
    per_country += 2 * B                                       # loss MAC into portfolio register
    per_country += 1 + 1 + 2 + 2                               # pay, exhausted flags, two counters
    tof += n * per_country
    depth = 2 * B * ceil(log2(max(n, 2))) + 12                 # loss accumulation as an adder tree
    # --- objective ---
    if target in ("E[L]", "stop-loss"):
        cnot += 2**B
        rot += 2**B
        if target == "stop-loss":
            tof += B
    elif target == "P(L>T)":
        tof += B + 1
    else:
        tof += 1
    t = 4 * tof + rot * ceil(3 * log2(1 / rot_precision))
    return Resources(n, target, qubits, tof, cnot + 6 * tof, rot, t, depth)


def iqae_queries(a: float, eps_rel: float, c: float = 21.3) -> float:
    """Queries for IQAE to reach relative RMSE eps_rel; c fitted on the n = 2 runs
    (results/accuracy_per_query.json, median over all scenarios and targets)."""
    return c * ((1 - a) / a) ** 0.5 / eps_rel


def mc_samples(a: float, eps_rel: float) -> float:
    """Plain Monte Carlo samples for relative RMSE eps_rel on a probability a."""
    return (1 - a) / (a * eps_rel**2)
