"""Layden et al. (2023) quantum proposal for the supply-chain MRF.

Qubit k (Qiskit index) carries component ``mrf.COMPONENTS[n-1-k]``, so the Qiskit basis
index of a measured bitstring equals Notebook 12's state ID: S1 is the leftmost character
of a Qiskit counts key and the most significant bit.

With b = (1 - z) / 2, where z = +-1 is the Z eigenvalue, Notebook 12's spin variable is
s = 2b - 1 = -z, so the problem Hamiltonian is

    H_prob = sum_i h_i Z_i - sum_(i,j) K_ij Z_i Z_j   (+ C * identity, a global phase),

whose diagonal is E(b) - C. One proposal step prepares |b>, evolves under
H = (1 - gamma) * alpha * H_prob + gamma * sum_j X_j for time t, and measures every qubit.
"""

import numpy as np

from . import mrf

GAMMA_RANGE = (0.25, 0.6)   # Layden et al., Algorithm S1
TIME_RANGE = (2.0, 20.0)


# ---------------------------------------------------------------- Hamiltonian pieces
def pauli_coefficients(theta=mrf.THETA, edges=mrf.EDGES):
    """Z-field coefficients hz[i] and ZZ couplings [(i, j, Jzz)] of H_prob (component indices)."""
    h, K, _ = mrf.ising_coefficients(theta, edges)
    return np.asarray(h), tuple((i, j, -k) for i, j, k in K)


def layden_alpha(energies):
    """alpha = ||H_mix||_F / ||H_prob||_F for a traceless diagonal H_prob.

    For an n-qubit diagonal Hamiltonian, ||H||_F / sqrt(2^n) is the RMS of its centred
    diagonal, which equals sqrt(sum of squared Pauli coefficients). ||H_mix||_F / sqrt(2^n) = sqrt(n).
    """
    e = np.asarray(energies, dtype=float)
    n = int(np.log2(len(e)))
    return float(np.sqrt(n) / np.sqrt(np.mean((e - e.mean()) ** 2)))


def mixer_matrix(n=mrf.N_BITS):
    """Dense sum_j X_j; entry [s, s'] is 1 exactly when s and s' differ in one bit."""
    N = 2**n
    M = np.zeros((N, N))
    ids = np.arange(N)
    for k in range(n):
        M[ids, ids ^ (1 << k)] = 1.0
    return M


def hamiltonian(energies, gamma, alpha=None):
    """Dense real symmetric H(gamma) for an enumerated energy vector (problem diagonal centred)."""
    e = np.asarray(energies, dtype=float)
    alpha = layden_alpha(e) if alpha is None else alpha
    n = int(np.log2(len(e)))
    return (1 - gamma) * alpha * np.diag(e - e.mean()) + gamma * mixer_matrix(n)


def transition_probabilities(energies, gamma, t, alpha=None):
    """|<s'|exp(-iHt)|s>|^2 for all pairs (exact time evolution)."""
    w, V = np.linalg.eigh(hamiltonian(energies, gamma, alpha))
    re = (V * np.cos(w * t)) @ V.T
    im = (V * np.sin(w * t)) @ V.T
    return re**2 + im**2


def draw_parameters(n_draws, rng, gamma_range=GAMMA_RANGE, time_range=TIME_RANGE):
    """Layden's randomized ensemble: gamma ~ U(0.25, 0.6), t ~ U(2, 20) independently."""
    return np.column_stack([rng.uniform(*gamma_range, n_draws), rng.uniform(*time_range, n_draws)])


def proposal_matrix(energies, parameters, alpha=None, trotter_steps=None):
    """Ensemble-averaged proposal Q[s, s'] over a fixed list of (gamma, t) draws.

    Each unitary satisfies U = U^T, so every |U[s', s]|^2 table and their average are
    exactly symmetric. ``trotter_steps=None`` uses exact evolution; an integer r uses r
    palindromic (Strang) Trotter steps, the circuit that a quantum computer would run.
    """
    e = np.asarray(energies, dtype=float)
    alpha = layden_alpha(e) if alpha is None else alpha
    Q = np.zeros((len(e), len(e)))
    for gamma, t in parameters:
        if trotter_steps is None:
            Q += transition_probabilities(e, gamma, t, alpha)
        else:
            Q += np.abs(trotter_unitary(e, gamma, t, trotter_steps, alpha)) ** 2
    return Q / len(parameters)


# ---------------------------------------------------------------- Trotterized circuit
def _apply_rx_all(U, angle, n):
    """Left-multiply U by prod_k RX(angle) = exp(-i angle/2 X) on every qubit.

    The n-fold tensor product is applied in blocks of at most six qubits (a 64 x 64 dense
    factor per block), which turns the work into a few large BLAS matrix products.
    """
    c, s = np.cos(angle / 2), -1j * np.sin(angle / 2)
    rx = np.array([[c, s], [s, c]])
    cols = U.shape[1]
    out = U
    done = 0
    while done < n:
        k = min(6, n - done)
        block = np.array([[1.0 + 0j]])
        for _ in range(k):
            block = np.kron(block, rx)
        left, right = 2**done, 2 ** (n - done - k)
        T = out.reshape(left, 2**k, right * cols)
        out = np.matmul(block[None, :, :], T).reshape(U.shape) if left > 1 else (block @ T[0]).reshape(U.shape)
        done += k
    return out


def trotter_unitary(energies, gamma, t, steps, alpha=None):
    """Matrix of r Strang steps exp(-iA dt/2) exp(-iB dt) exp(-iA dt/2), A diagonal, B mixer.

    Each factor is a symmetric matrix and the step is a palindrome, so (ABA)^r equals its own
    transpose and the Trotterized proposal |U|^2 is *exactly* symmetric however coarse dt is.
    (A first-order product (DM)^r is also safe: its transpose is D^-1 (DM)^r D, which has the same
    moduli. What breaks symmetry is a non-palindromic sequence of *different* layers, such as a
    ramped schedule; see Notebook 18.)
    """
    e = np.asarray(energies, dtype=float)
    alpha = layden_alpha(e) if alpha is None else alpha
    n = int(np.log2(len(e)))
    dt = t / steps
    half_diag = np.exp(-1j * (1 - gamma) * alpha * (e - e.mean()) * dt / 2)
    U = np.eye(len(e), dtype=complex)
    for _ in range(steps):
        U = half_diag[:, None] * U
        U = _apply_rx_all(U, 2 * gamma * dt, n)
        U = half_diag[:, None] * U
    return U


def qubit_of(component_index, n=mrf.N_BITS):
    return n - 1 - component_index


def proposal_circuit(bits, gamma, t, steps, alpha=None, theta=mrf.THETA, edges=mrf.EDGES, measure=True):
    """Qiskit circuit: prepare |b>, apply r Strang steps of H(gamma) for time t, measure.

    Adjacent half-steps of the diagonal layer are merged, so r steps cost r+1 diagonal
    layers (each: n RZ and |E| RZZ gates) and r mixer layers (n RX gates).
    """
    from itertools import product

    from qiskit import QuantumCircuit

    n = len(bits)
    if len(theta) != n:
        raise ValueError("theta must have one field per qubit")
    hz, zz = pauli_coefficients(theta, edges)
    if alpha is None:
        alpha = layden_alpha(mrf.energy(np.array(list(product([0, 1], repeat=n))), theta, edges))
    a, dt = (1 - gamma) * alpha, t / steps
    qc = QuantumCircuit(n, n if measure else 0, name="qemcmc_proposal")
    for k, b in enumerate(bits):
        if b:
            qc.x(qubit_of(k, n))

    def diagonal_layer(tau):
        # exp(-i tau a hz Z) = RZ(2 tau a hz); exp(-i tau a J ZZ) = RZZ(2 tau a J)
        for k in range(n):
            qc.rz(2 * tau * a * hz[k], qubit_of(k, n))
        for i, j, J in zz:
            qc.rzz(2 * tau * a * J, qubit_of(i, n), qubit_of(j, n))

    for step in range(steps):
        diagonal_layer(dt / 2 if step == 0 else dt)
        for k in range(n):
            qc.rx(2 * gamma * dt, k)
    diagonal_layer(dt / 2)
    if measure:
        qc.measure(range(n), range(n))
    return qc


def counts_key_to_state(key):
    """Qiskit counts key (qubit n-1 leftmost) -> Notebook 12 state ID."""
    return int(key.replace(" ", ""), 2)


def schedule_unitary(energies, gammas, t, alpha=None):
    """Strang steps with a per-step mixing weight gamma_k (e.g. a ramped or annealing-like schedule).

    With a constant schedule this equals ``trotter_unitary``. A palindromic schedule
    (gammas == gammas[::-1]) gives U = U^T; a one-way ramp generally does not, so its proposal
    is asymmetric and the simplified Metropolis rule becomes biased.
    """
    e = np.asarray(energies, dtype=float)
    alpha = layden_alpha(e) if alpha is None else alpha
    n = int(np.log2(len(e)))
    dt = t / len(gammas)
    U = np.eye(len(e), dtype=complex)
    for gamma in gammas:
        half_diag = np.exp(-1j * (1 - gamma) * alpha * (e - e.mean()) * dt / 2)
        U = half_diag[:, None] * U
        U = _apply_rx_all(U, 2 * gamma * dt, n)
        U = half_diag[:, None] * U
    return U


def bit_flip_channel(flip_0_to_1, flip_1_to_0, n=mrf.N_BITS):
    """Classical n-bit flip channel R[read, true] built from independent per-bit flip rates."""
    r = np.array([[1 - flip_0_to_1, flip_1_to_0], [flip_0_to_1, 1 - flip_1_to_0]])
    R = np.array([[1.0]])
    for _ in range(n):
        R = np.kron(R, r)
    return R
