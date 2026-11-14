"""Quantinuum H-series / Helios helpers: native compilation, resource and cost estimates, noise.

Compilation uses pytket-quantinuum's ``QuantinuumBackend(..., machine_debug=True)`` which compiles
offline to the native gate set (single-qubit ``PhasedX`` and ``Rz``, arbitrary-angle two-qubit
``ZZPhase``) without an account. Submitting to an emulator or hardware goes through Quantinuum Nexus
(``qnexus``); see ``notebooks/part_05/run_on_quantinuum.py``.

Hardware numbers below are *published* figures (data sheets / launch material, cited in
``SPECS``) used only to build an illustrative noise model. They are not measurements made here and
should be refreshed from Quantinuum's current documentation before any hardware run.
"""

from collections import Counter

import numpy as np

from . import quantum

SPECS = {
    "H2-1": {
        "qubits": 56, "p1": 3e-5, "p2": 1.1e-3, "spam_0": 6.0e-4, "spam_1": 1.4e-3,
        "source": "Quantinuum H2 product data sheet / performance validation (2Q error 1.1e-3; SPAM |0> 6.0e-4, |1> 1.4e-3)",
    },
    "Helios-1": {
        "qubits": 98, "p1": 2.5e-5, "p2": 7.9e-4, "spam_0": 6.0e-4, "spam_1": 1.4e-3,
        "source": "Helios launch, Nov 2025 (1Q fidelity 99.9975%, 2Q fidelity 99.921%); SPAM assumed equal to H2-1",
    },
}

TWO_QUBIT = {"ZZPhase", "ZZMax", "TK2", "CX", "CZ"}
ONE_QUBIT = {"PhasedX", "Rz", "Rx", "Ry", "X", "Y", "Z", "H", "U3", "U1", "SX", "TK1"}


def to_tket(qiskit_circuit):
    from pytket.extensions.qiskit import qiskit_to_tk

    return qiskit_to_tk(qiskit_circuit)


def compile_native(circuit, device="H2-1", optimisation_level=2):
    """Compile a pytket circuit offline to the H-series native gate set."""
    from pytket.extensions.quantinuum import QuantinuumBackend

    backend = QuantinuumBackend(device_name=device, machine_debug=True)
    return backend.get_compiled_circuit(circuit, optimisation_level=optimisation_level)


def gate_counts(circuit):
    """Native one-qubit, two-qubit and SPAM operation counts as used by the HQC formula."""
    ops = Counter(cmd.op.type.name for cmd in circuit.get_commands())
    n2 = sum(v for k, v in ops.items() if k in TWO_QUBIT)
    n_meas = ops.get("Measure", 0) + ops.get("Reset", 0)
    n1 = sum(v for k, v in ops.items() if k not in TWO_QUBIT and k not in {"Measure", "Reset", "Barrier"})
    return {"1q": n1, "2q": n2, "spam": circuit.n_qubits + n_meas, "depth_2q": circuit.depth_2q(), "ops": dict(ops)}


def hqc(n1q, n2q, n_spam, shots):
    """Published H-series cost model: HQC = 5 + C (N1q + 10 N2q + 5 Nm) / 5000 (one job item)."""
    return 5 + shots * (n1q + 10 * n2q + 5 * n_spam) / 5000


def spam_channel(spec, n):
    """Classical readout channel R[read, true] with P(read 1 | 0) = spam_0 and P(read 0 | 1) = spam_1."""
    return quantum.bit_flip_channel(spec["spam_0"], spec["spam_1"], n)


def effective_depolarizing(spec, n1q, n2q):
    """Probability that at least one gate error occurs: 1 - (1 - p1)^N1 (1 - p2)^N2."""
    return 1 - (1 - spec["p1"]) ** n1q * (1 - spec["p2"]) ** n2q


def aer_noise_model(spec):
    """Qiskit Aer noise model with depolarizing gate errors and asymmetric readout from ``spec``."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(spec["p1"], 1), ["rx", "ry", "r", "u", "u3", "x", "sx"])
    model.add_all_qubit_quantum_error(depolarizing_error(spec["p2"], 2), ["rzz", "cx", "cz", "ecr"])
    model.add_all_qubit_readout_error(ReadoutError([[1 - spec["spam_0"], spec["spam_0"]],
                                                    [spec["spam_1"], 1 - spec["spam_1"]]]))
    return model


def twirled_readout_and_input_flip(rng, bits, spec):
    """Randomization that makes asymmetric readout symmetric and restores exact proposal symmetry.

    Returns (bits to prepare, readout flip mask). Prepare the returned bits; apply X before measuring
    on qubits where the mask is 1 and flip those measured bits back classically. The input flips use
    the averaged readout error rate so the effective proposal is R Q R with symmetric R (Notebook 18).
    """
    bits = np.asarray(bits)
    e = 0.5 * (spec["spam_0"] + spec["spam_1"])
    input_flip = (rng.random(len(bits)) < e).astype(int)
    readout_mask = rng.integers(0, 2, len(bits))
    return bits ^ input_flip, readout_mask
