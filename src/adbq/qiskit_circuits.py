"""Qiskit implementations of the state preparation A and the Grover operator Q.

Three circuit families:

1. ``to_qiskit``: a gate-for-gate translation of adbq's own circuits (circuits.py). MCRY
   becomes a multi-controlled RY with the same control values; a reversible classical block
   out ^= fn(in) is synthesised exactly from its truth table as multi-controlled X gates.
   Qubit q of adbq is qubit q of Qiskit (both little-endian), so basis indices agree.
   Used for the full two-country A (24 qubits) and checked against adbq's simulator.

2. ``chain_indicator_circuit``: A for P(L > T) on the n-country chain, built from Qiskit's
   arithmetic library so it scales: Grover-Rudolph source loading (adbq), WeightedAdder
   for each country's quake steps, a 4-to-3 saturation table, OR of rain triggers, a
   WeightedAdder for the portfolio loss, IntegerComparator for L > T. Used for compiled
   resource counts n = 2..10. Verified by composition: loading per register by statevector,
   the arithmetic as a classical permutation on every source configuration (n = 2, 3).

3. ``rain_joint_circuit`` / ``quake_trigger_circuit``: small sub-circuits of the two-country
   A (joint rain trigger, 6 qubits; joint quake trigger, 15 qubits) small enough for noisy
   simulation and a first hardware run of Q^k A, k <= 2.

Grover operator: Q = A S_0 A^dagger S_chi with S_chi = Z on the objective qubit
(qiskit.circuit.library.grover_operator), so P(objective = 1 after Q^k A) = sin^2((2k+1) theta).
"""
from __future__ import annotations

from math import ceil, log2

import numpy as np

from .circuits import MCRY, REV, Circuit, load_distribution
from .model import STEPS, Scenario, two_country_sources
from .portfolio import ChainPortfolio, _units


def _qk():
    import qiskit
    from qiskit.circuit import library
    return qiskit, library


# ---------------------------------------------------------------------------
# 1. Gate-for-gate translation
# ---------------------------------------------------------------------------
def _mcry(qc, g: MCRY):
    _, lib = _qk()
    if not g.controls:
        qc.ry(g.theta, g.target)
        return
    ctrls = [q for q, _ in g.controls]
    state = sum(v << i for i, (_, v) in enumerate(g.controls))
    qc.append(lib.RYGate(g.theta).control(len(ctrls), ctrl_state=state, annotated=False), ctrls + [g.target])


def _rev(qc, g: REV):
    _, lib = _qk()
    nin = len(g.inputs)
    for x in range(2**nin):
        y = g.fn(x)
        for j, out in enumerate(g.outputs):
            if (y >> j) & 1:
                qc.append(lib.MCXGate(nin, ctrl_state=x), list(g.inputs) + [out])


def to_qiskit(circ: Circuit, name: str = "A"):
    qiskit, _ = _qk()
    qc = qiskit.QuantumCircuit(circ.n_qubits, name=name)
    for g in circ.gates:
        (_mcry if isinstance(g, MCRY) else _rev)(qc, g)
    return qc


def grover(A, objective: int):
    """Q = A S_0 A^dagger S_chi with S_chi = Z on ``objective``."""
    qiskit, lib = _qk()
    oracle = qiskit.QuantumCircuit(A.num_qubits)
    oracle.z(objective)
    return lib.grover_operator(oracle, state_preparation=A)


def amplified(A, objective: int, k: int, measure: bool = False):
    """Q^k A, optionally measuring the objective qubit only."""
    qiskit, _ = _qk()
    qc = qiskit.QuantumCircuit(A.num_qubits, 1 if measure else 0)
    qc.compose(A, inplace=True)
    if k:
        Q = grover(A, objective)
        for _ in range(k):
            qc.compose(Q, inplace=True)
    if measure:
        qc.measure(objective, 0)
    return qc


def prob_objective(qc, objective: int) -> float:
    """Exact P(objective = 1) from Aer's statevector (decomposes multi-controlled gates)."""
    qiskit, _ = _qk()
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="statevector")
    c = qc.copy()
    c.save_statevector()
    sv = np.asarray(sim.run(qiskit.transpile(c, sim, optimization_level=0)).result().get_statevector())
    idx = np.arange(len(sv))
    return float((np.abs(sv[(idx >> objective) & 1 == 1]) ** 2).sum())


# ---------------------------------------------------------------------------
# 3. Small sub-circuits of the two-country A for noisy simulation and hardware
# ---------------------------------------------------------------------------
def _add_loader(qc, qubits, probs):
    for g in load_distribution(list(qubits), np.asarray(probs)):
        _mcry(qc, g)


def rain_joint_circuit(s: Scenario = Scenario()):
    """P(r1 = r2 = 1): three rain sources (u1, u2, z), r_i = u_i OR z, objective = r1 AND r2.
    Qubits: 0 u1, 1 u2, 2 z, 3 r1, 4 r2, 5 objective. Returns (circuit, objective, exact)."""
    qiskit, _ = _qk()
    src = two_country_sources(s)
    qc = qiskit.QuantumCircuit(6, name="A_rain")
    qc.ry(2 * np.arcsin(np.sqrt(src.own_rain[0])), 0)
    qc.ry(2 * np.arcsin(np.sqrt(src.own_rain[1])), 1)
    qc.ry(2 * np.arcsin(np.sqrt(src.shared_rain)), 2)
    for u, r in ((0, 3), (1, 4)):                      # r = u OR z = NOT(NOT u AND NOT z)
        qc.x([u, 2]); qc.ccx(u, 2, r); qc.x([u, 2]); qc.x(r)
    qc.ccx(3, 4, 5)
    pu1, pu2, pz = src.own_rain[0], src.own_rain[1], src.shared_rain
    exact = pz + (1 - pz) * pu1 * pu2
    return qc, 5, float(exact)


def quake_trigger_circuit(s: Scenario = Scenario()):
    """P(a >= 1 and b >= 1) over the term: own quake 1 (q0-2), own quake 2 (q3-5), shared
    (x: q6-8, y: q9-11); f1 = (own1 != 0) OR (x != 0) on q12, f2 likewise on q13,
    objective = f1 AND f2 on q14. Returns (circuit, objective, exact)."""
    qiskit, lib = _qk()
    src = two_country_sources(s)
    qc = qiskit.QuantumCircuit(15, name="A_quake")
    _add_loader(qc, [0, 1, 2], src.own_quake[0])
    _add_loader(qc, [3, 4, 5], src.own_quake[1])
    shared = np.zeros(64)
    for (x, y), p in np.ndenumerate(src.shared_quake):
        shared[x + 8 * y] = p
    _add_loader(qc, [6, 7, 8, 9, 10, 11], shared)
    for regs, f in (([0, 1, 2, 6, 7, 8], 12), ([3, 4, 5, 9, 10, 11], 13)):
        qc.append(lib.MCXGate(6, ctrl_state=0), regs + [f])   # f = 1 iff all six bits are 0
        qc.x(f)                                               # -> f = 1 iff any bit set
    qc.ccx(12, 13, 14)
    q = src.shared_quake
    o1, o2 = src.own_quake
    p_a0 = o1[0] * q[0, :].sum()          # a = 0: own1 = 0 and x = 0
    p_b0 = o2[0] * q[:, 0].sum()
    p_both0 = o1[0] * o2[0] * q[0, 0]
    exact = 1 - p_a0 - p_b0 + p_both0
    return qc, 14, float(exact)


# ---------------------------------------------------------------------------
# 2. Scalable A for P(L > T) on the chain, from Qiskit's arithmetic library
# ---------------------------------------------------------------------------
def chain_indicator_circuit(pf: ChainPortfolio):
    """A for P(L > T) on the n-country chain. Returns (circuit, objective, layout dict)."""
    qiskit, lib = _qk()
    from .model import capped_compound_poisson, capped_shared_pair
    n, s, H = pf.n, pf.scen, pf.scen.years
    qu, ru, Tu, u = _units(s)
    regs = {}
    nq = 0

    def alloc(name, k):
        nonlocal nq
        regs[name] = list(range(nq, nq + k)); nq += k
        return regs[name]

    for i in range(n):
        alloc(f"own_q{i}", 3); alloc(f"own_r{i}", 1)
    for e in range(n - 1):
        alloc(f"sh_q{e}", 6); alloc(f"sh_r{e}", 1)
    # per-country quake sum (WeightedAdder over 9 bits -> 4 sum bits + carries), a_i, r_i
    sum_adders = []
    for i in range(n):
        wa = lib.WeightedAdder(9, [1, 2, 4] * 3)
        sum_adders.append(wa)
        alloc(f"sum{i}", wa.num_sum_qubits); alloc(f"carry{i}", wa.num_carry_qubits)
        alloc(f"ctrl{i}", wa.num_control_qubits)
        alloc(f"a{i}", 3); alloc(f"r{i}", 1)
    weights = []
    for i in range(n):
        weights += [qu * 1, qu * 2, qu * 4, ru]
    loss_add = lib.WeightedAdder(4 * n, weights)
    alloc("loss", loss_add.num_sum_qubits); alloc("loss_carry", loss_add.num_carry_qubits)
    alloc("loss_ctrl", loss_add.num_control_qubits)
    cmp_ = lib.IntegerComparator(loss_add.num_sum_qubits, Tu + 1, geq=True)
    alloc("cmp_out", 1); alloc("cmp_anc", cmp_.num_ancillas)
    qc = qiskit.QuantumCircuit(nq, name=f"A_chain{n}")
    # 1. sources
    for i in range(n):
        _add_loader(qc, regs[f"own_q{i}"], capped_compound_poisson(pf.own_quake_rate[i] * pf.w[i] * H))
        qc.ry(2 * np.arcsin(np.sqrt(1 - np.exp(-pf.own_rain_rate[i] * H))), regs[f"own_r{i}"][0])
    for e in range(n - 1):
        J = capped_shared_pair(pf.c[e] * H, pf.w[e], pf.w[e + 1])
        flat = np.zeros(64)
        for (x, y), p in np.ndenumerate(J):
            flat[x + 8 * y] = p
        _add_loader(qc, regs[f"sh_q{e}"], flat)
        qc.ry(2 * np.arcsin(np.sqrt(1 - np.exp(-pf.cr[e] * H))), regs[f"sh_r{e}"][0])
    regs["_n_load"] = len(qc.data)          # instructions before this index are the loaders
    # 2. per country: s_i = own + left part + right part; a_i = min(4, s_i); r_i = OR
    for i in range(n):
        left = regs[f"sh_q{i - 1}"][3:] if i > 0 else None       # y-part of edge (i-1, i)
        right = regs[f"sh_q{i}"][:3] if i < n - 1 else None      # x-part of edge (i, i+1)
        state = regs[f"own_q{i}"] + (left if left else []) + (right if right else [])
        wts = [1, 2, 4] * (len(state) // 3)
        wa = lib.WeightedAdder(len(state), wts)
        need = wa.num_sum_qubits + wa.num_carry_qubits + wa.num_control_qubits
        work = regs[f"sum{i}"] + regs[f"carry{i}"] + regs[f"ctrl{i}"]
        sum_q = regs[f"sum{i}"][: wa.num_sum_qubits]
        rest = [q for q in work if q not in sum_q][: need - wa.num_sum_qubits]
        qc.compose(wa, state + sum_q + rest, inplace=True)
        sat = REV(tuple(sum_q), tuple(regs[f"a{i}"]), lambda v: min(STEPS, v), f"a{i} = min(4, s{i})")
        _rev(qc, sat)
        rain_in = regs[f"own_r{i}"] + (regs[f"sh_r{i - 1}"] if i > 0 else []) + (regs[f"sh_r{i}"] if i < n - 1 else [])
        r = regs[f"r{i}"][0]
        qc.append(lib.MCXGate(len(rain_in), ctrl_state=0), rain_in + [r]); qc.x(r)
    # 3. portfolio loss and comparator
    state = []
    for i in range(n):
        state += regs[f"a{i}"] + regs[f"r{i}"]
    qc.compose(loss_add, state + regs["loss"] + regs["loss_carry"] + regs["loss_ctrl"], inplace=True)
    qc.compose(cmp_, regs["loss"] + regs["cmp_out"] + regs["cmp_anc"], inplace=True)
    return qc, regs["cmp_out"][0], regs


# ---------------------------------------------------------------------------
# Classical permutation check of the arithmetic (verification by composition)
# ---------------------------------------------------------------------------
def classical_eval(qc, bits: np.ndarray) -> np.ndarray:
    """Run a circuit built from X-type gates (X, CX, CCX, MCX with any control states, and
    blocks whose definitions reduce to them) on a batch of classical bit-strings
    ``bits`` (shape (N, num_qubits)). Raises on any gate that is not a classical permutation."""
    from qiskit.circuit import ControlledGate
    b = bits.copy()

    def run(circ, qmap):
        for ins in circ.data:
            op = ins.operation
            qs = [qmap[circ.find_bit(q).index] for q in ins.qubits]
            if op.name in ("barrier", "id", "delay"):
                continue
            if op.name == "x" and not isinstance(op, ControlledGate):
                b[:, qs[0]] ^= 1
            elif isinstance(op, ControlledGate) and op.base_gate.name == "x":
                nc = op.num_ctrl_qubits
                want = np.array([(op.ctrl_state >> i) & 1 for i in range(nc)], np.int8)
                on = np.all(b[:, qs[:nc]] == want, axis=1)
                b[on, qs[nc]] ^= 1
            elif op.definition is not None:
                run(op.definition, qs)
            else:
                raise ValueError(f"non-classical gate {op.name}")

    run(qc, list(range(qc.num_qubits)))
    return b


def arithmetic_part(qc, n_load_ops: int):
    """Split a circuit after its first ``n_load_ops`` instructions (the loaders)."""
    qiskit, _ = _qk()
    out = qiskit.QuantumCircuit(qc.num_qubits)
    for ins in qc.data[n_load_ops:]:
        out.append(ins.operation, ins.qubits, ins.clbits)
    return out


# ---------------------------------------------------------------------------
# Resource counts from compiled circuits
# ---------------------------------------------------------------------------
def compiled_counts(qc, rot_precision: float = 1e-10) -> dict:
    """Compile to {cx, u} for CX count and depth, and to {ccx, cx, u} for a T estimate:
    T = 7 per Toffoli (textbook; 4 with Gidney's measurement-based uncomputation) +
    1 per exact pi/4-type rotation + ceil(3 log2(1/delta)) per arbitrary rotation
    (Ross and Selinger 2016)."""
    qiskit, _ = _qk()
    c1 = qiskit.transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    ops1 = c1.count_ops()
    c2 = qiskit.transpile(qc, basis_gates=["ccx", "cx", "u"], optimization_level=1)
    ops2 = c2.count_ops()
    t_exact, arb = 0, 0
    for ins in c2.data:
        if ins.operation.name != "u":
            continue
        angles = np.array([float(p) for p in ins.operation.params])
        k = angles / (np.pi / 4)
        if np.allclose(k, np.round(k), atol=1e-9):
            t_exact += int(np.any(np.round(k).astype(int) % 2 == 1))
        else:
            arb += 1
    t_rot = ceil(3 * log2(1 / rot_precision))
    return {"qubits": qc.num_qubits, "cx": int(ops1.get("cx", 0)), "depth": int(c1.depth()),
            "ccx": int(ops2.get("ccx", 0)), "u_exact_t": t_exact, "arbitrary_rotations": arb,
            "t_estimate_textbook": 7 * int(ops2.get("ccx", 0)) + t_exact + arb * t_rot,
            "t_estimate_gidney": 4 * int(ops2.get("ccx", 0)) + t_exact + arb * t_rot}


def verify_chain_arithmetic(pf: ChainPortfolio, qc, objective: int, regs: dict, batch: int = 50_000):
    """Exact P(L > T) of the chain A by composition: every source configuration (with its
    probability) is pushed through the circuit's arithmetic as a classical permutation and
    the objective bit is compared with 1{L > T}. Returns (P from the circuit, mismatches)."""
    import itertools
    from .model import capped_compound_poisson, capped_shared_pair
    n, s, H = pf.n, pf.scen, pf.scen.years
    qu, ru, Tu, u = _units(s)
    ar = arithmetic_part(qc, regs["_n_load"])
    laws = []          # (register qubits, list of (value, prob)) per source
    for i in range(n):
        po = capped_compound_poisson(pf.own_quake_rate[i] * pf.w[i] * H)
        laws.append((regs[f"own_q{i}"], [(v, p) for v, p in enumerate(po) if p > 0]))
        pr = 1 - np.exp(-pf.own_rain_rate[i] * H)
        laws.append((regs[f"own_r{i}"], [(0, 1 - pr), (1, pr)]))
    for e in range(n - 1):
        J = capped_shared_pair(pf.c[e] * H, pf.w[e], pf.w[e + 1])
        laws.append((regs[f"sh_q{e}"], [(x + 8 * y, p) for (x, y), p in np.ndenumerate(J) if p > 0]))
        pz = 1 - np.exp(-pf.cr[e] * H)
        laws.append((regs[f"sh_r{e}"], [(0, 1 - pz), (1, pz)]))
    combos = itertools.product(*[l[1] for l in laws])
    p_circ, mism = 0.0, 0
    while True:
        chunk = list(itertools.islice(combos, batch))
        if not chunk:
            break
        B = np.zeros((len(chunk), qc.num_qubits), np.int8)
        P = np.ones(len(chunk))
        for k, (qs, _) in enumerate(laws):
            vals = np.array([c[k][0] for c in chunk]); P *= np.array([c[k][1] for c in chunk])
            for j, q in enumerate(qs):
                B[:, q] = (vals >> j) & 1
        # reference loss of each configuration
        steps = np.zeros((len(chunk), n), int); rain = np.zeros((len(chunk), n), int)
        for i in range(n):
            steps[:, i] += sum(B[:, q] << j for j, q in enumerate(regs[f"own_q{i}"]))
            rain[:, i] |= B[:, regs[f"own_r{i}"][0]]
        for e in range(n - 1):
            v = sum(B[:, q].astype(int) << j for j, q in enumerate(regs[f"sh_q{e}"]))
            steps[:, e] += v & 7; steps[:, e + 1] += v >> 3
            rain[:, e] |= B[:, regs[f"sh_r{e}"][0]]; rain[:, e + 1] |= B[:, regs[f"sh_r{e}"][0]]
        L = (np.minimum(STEPS, steps) * qu + rain * ru).sum(1)
        out = classical_eval(ar, B)[:, objective]
        mism += int((out != (L > Tu)).sum())
        p_circ += float(P[out == 1].sum())
    return p_circ, mism
