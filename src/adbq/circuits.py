"""A minimal gate-level circuit model with an exact sparse state-vector simulator.

No external quantum SDK is needed: the circuits here use only

* ``MCRY`` -- an RY rotation with any number of controls, each on a 0 or 1 value
  (the building block of Grover-Rudolph / Moettoenen state preparation and of the
  objective rotation), and
* ``REV``  -- a reversible classical function ``out ^= fn(in)`` (saturating adders,
  OR gates, comparators); it is a permutation of basis states.

The state is a dict {basis index -> real amplitude}. Every gate here maps real
amplitudes to real amplitudes, so the simulation is exact in double precision.
Because the loaded state is a product over a few small source registers, the support
stays at a few thousand basis states and full Grover iterates are cheap to simulate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class MCRY:
    controls: tuple[tuple[int, int], ...]   # (qubit, required value)
    target: int
    theta: float

    def inverse(self) -> "MCRY":
        return MCRY(self.controls, self.target, -self.theta)


@dataclass(frozen=True)
class REV:
    inputs: tuple[int, ...]                 # little-endian
    outputs: tuple[int, ...]                # little-endian
    fn: Callable[[int], int]
    label: str = ""

    def inverse(self) -> "REV":
        return self                         # XOR into the output is self-inverse


@dataclass
class Circuit:
    n_qubits: int
    gates: list = field(default_factory=list)

    def add(self, g):
        self.gates.append(g)

    def extend(self, gs):
        self.gates.extend(gs)

    def inverse(self) -> "Circuit":
        return Circuit(self.n_qubits, [g.inverse() for g in reversed(self.gates)])


def _bits(x: int, qubits) -> int:
    v = 0
    for i, q in enumerate(qubits):
        v |= ((x >> q) & 1) << i
    return v


def _set_bits(x: int, qubits, value: int) -> int:
    for i, q in enumerate(qubits):
        if (value >> i) & 1:
            x ^= 1 << q
    return x


def apply(state: dict[int, float], circ: Circuit) -> dict[int, float]:
    for g in circ.gates:
        if isinstance(g, MCRY):
            mask = 1 << g.target
            c, s = np.cos(g.theta / 2), np.sin(g.theta / 2)
            new: dict[int, float] = {}
            for x, amp in state.items():
                if any(((x >> q) & 1) != v for q, v in g.controls):
                    new[x] = new.get(x, 0.0) + amp
                    continue
                x0, x1 = x & ~mask, x | mask
                if x & mask:   # |1> -> -s|0> + c|1>
                    new[x0] = new.get(x0, 0.0) - s * amp
                    new[x1] = new.get(x1, 0.0) + c * amp
                else:          # |0> ->  c|0> + s|1>
                    new[x0] = new.get(x0, 0.0) + c * amp
                    new[x1] = new.get(x1, 0.0) + s * amp
            state = {k: v for k, v in new.items() if abs(v) > 1e-17}
        elif isinstance(g, REV):
            new = {}
            for x, amp in state.items():
                y = _set_bits(x, g.outputs, g.fn(_bits(x, g.inputs)))
                new[y] = new.get(y, 0.0) + amp
            state = new
        else:
            raise TypeError(g)
    return state


# ---------------------------------------------------------------------------
# Distribution loading (Grover-Rudolph binary tree of multi-controlled RY)
# ---------------------------------------------------------------------------
def load_distribution(qubits: list[int], probs: np.ndarray) -> list[MCRY]:
    """Gates that map |0...0> on ``qubits`` (little-endian) to sum_x sqrt(p_x)|x>.

    The tree branches on the most significant qubit first; each node is one RY
    controlled on the already-fixed higher bits. Zero-probability subtrees are skipped.
    """
    m = len(qubits)
    p = np.zeros(2**m)
    p[: len(probs)] = probs
    p = p / p.sum()
    gates = []
    for level in range(m):                    # level 0 = MSB
        tq = qubits[m - 1 - level]
        span = 2 ** (m - level)               # values sharing a prefix
        for prefix in range(2**level):
            lo = prefix * span
            block = p[lo: lo + span]
            tot = block.sum()
            if tot <= 0:
                continue
            p1 = block[span // 2:].sum() / tot
            if p1 == 0:
                continue
            theta = 2 * np.arcsin(np.sqrt(p1))
            ctrls = tuple((qubits[m - 1 - j], (prefix >> (level - 1 - j)) & 1) for j in range(level))
            gates.append(MCRY(ctrls, tq, theta))
    return gates


def value_rotation(ctrl_qubits: list[int], target: int, values: np.ndarray) -> list[MCRY]:
    """Uniformly controlled RY: rotate ``target`` so P(1 | x) = values[x] in [0, 1]."""
    gates = []
    for x, v in enumerate(values):
        if v <= 0:
            continue
        theta = 2 * np.arcsin(np.sqrt(min(1.0, v)))
        ctrls = tuple((q, (x >> i) & 1) for i, q in enumerate(ctrl_qubits))
        gates.append(MCRY(ctrls, target, theta))
    return gates


def prob_good(state: dict[int, float], good: Callable[[int], bool]) -> float:
    return float(sum(a * a for x, a in state.items() if good(x)))


def grover_iterate(state, A: Circuit, Ainv: Circuit, good: Callable[[int], bool]):
    """Q = A S0 A^dagger S_chi  (Brassard et al. 2002), applied to ``state``."""
    st = {x: (-a if good(x) else a) for x, a in state.items()}       # S_chi
    st = apply(st, Ainv)
    st = {x: (a if x == 0 else -a) for x, a in st.items()}           # S0 up to global phase
    st = {x: -a for x, a in st.items()}
    return apply(st, A)
