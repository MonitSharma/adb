import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from adbq.circuits import REV, Circuit, apply, load_distribution          # noqa: E402
from adbq.model import SCENARIOS, TARGETS, exact                          # noqa: E402
from adbq.portfolio import chain_portfolio, exact_chain                   # noqa: E402
from adbq.qiskit_circuits import (amplified, chain_indicator_circuit, classical_eval,  # noqa: E402
                                  compiled_counts, prob_objective, quake_trigger_circuit,
                                  rain_joint_circuit, to_qiskit, verify_chain_arithmetic)


def test_translation_of_loader_and_reversible_block():
    import qiskit
    from qiskit_aer import AerSimulator
    c = Circuit(6)
    c.extend(load_distribution([0, 1, 2], np.array([0.5, 0.2, 0.1, 0.05, 0.15])))
    c.add(REV((0, 1, 2), (3, 4, 5), lambda v: min(4, v + 1)))
    ref = apply({0: 1.0}, c)
    qc = to_qiskit(c)
    qc.save_statevector()
    sim = AerSimulator(method="statevector")
    sv = np.asarray(sim.run(qiskit.transpile(qc, sim)).result().get_statevector())
    want = np.zeros(64); [want.__setitem__(k, v) for k, v in ref.items()]
    assert np.abs(sv - want).max() < 1e-12


@pytest.mark.parametrize("make", [rain_joint_circuit, quake_trigger_circuit])
def test_small_circuits_and_grover(make):
    qc, obj, ex = make()
    th = np.arcsin(np.sqrt(ex))
    for k in (0, 1, 2):
        assert prob_objective(amplified(qc, obj, k), obj) == pytest.approx(np.sin((2 * k + 1) * th) ** 2, abs=1e-12)


def test_small_circuit_targets_match_the_model():
    from adbq.grid import brief_trigger_law
    from adbq.model import term_distribution_sources
    assert quake_trigger_circuit()[2] == pytest.approx(brief_trigger_law()["p_both"], rel=1e-12)
    t = term_distribution_sources(SCENARIOS["adb-base"])
    assert rain_joint_circuit()[2] == pytest.approx(t[:, :, 1, 1].sum(), rel=1e-12)


def test_classical_evaluator_handles_open_controls():
    import qiskit
    from qiskit.circuit.library import MCXGate
    qc = qiskit.QuantumCircuit(3)
    qc.append(MCXGate(2, ctrl_state=0b01), [0, 1, 2])           # fires when q0 = 1, q1 = 0
    b = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], np.int8)
    assert classical_eval(qc, b)[:, 2].tolist() == [1, 0, 0]
    qc.ry(0.3, 0)
    with pytest.raises(ValueError):
        classical_eval(qc, b)


def test_chain_circuit_by_composition_n2():
    pf = chain_portfolio(2)
    qc, obj, regs = chain_indicator_circuit(pf)
    p, mism = verify_chain_arithmetic(pf, qc, obj, regs)
    assert mism == 0
    assert p == pytest.approx(exact_chain(pf)["P(L>T)"], rel=1e-12)


def test_compiled_counts_grow_with_n():
    c2 = compiled_counts(chain_indicator_circuit(chain_portfolio(2))[0])
    c4 = compiled_counts(chain_indicator_circuit(chain_portfolio(4))[0])
    assert c4["qubits"] > c2["qubits"] and c4["cx"] > c2["cx"] and c4["t_estimate_gidney"] > c2["t_estimate_gidney"]


@pytest.mark.slow
def test_full_two_country_A_statevector():
    from adbq.oracle import OBJ, build_oracle
    o = build_oracle(SCENARIOS["adb-base"], TARGETS[2])
    a = prob_objective(to_qiskit(o.A), OBJ)
    assert o.f_max * a == pytest.approx(exact("adb-base")[TARGETS[2]], rel=1e-10)
