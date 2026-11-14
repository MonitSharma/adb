"""Notebook 12's nine-bit correlated-disruption MRF, frozen.

Bit convention: b_i = 0 operational, 1 disrupted. Component order is S1, S2, S3, S4, SEA,
AIR, WH, F1, F2. States are enumerated lexicographically with S1 as the most significant
bit, so ``state_id(b) = sum_i b_i 2**(8-i)`` and ID 0 is the fully operational network.
"""

from itertools import product

import numpy as np

COMPONENTS = ("S1", "S2", "S3", "S4", "SEA", "AIR", "WH", "F1", "F2")
INDEX = {name: i for i, name in enumerate(COMPONENTS)}
N_BITS = len(COMPONENTS)
N_STATES = 2**N_BITS

THETA = np.array([3.0, 3.0, 3.1, 3.4, 3.0, 3.1, 3.0, 3.0, 3.0])
COUPLINGS = (
    ("S1", "S2", 1.6), ("S2", "S3", 1.6), ("S3", "S1", 1.6),
    ("S1", "SEA", 0.6), ("S2", "SEA", 0.6), ("S3", "AIR", 0.6), ("S4", "AIR", 0.7),
    ("SEA", "WH", 1.5), ("WH", "AIR", 1.5), ("SEA", "AIR", 0.8),
    ("WH", "F1", 1.1), ("WH", "F2", 1.1), ("F1", "F2", 3.2),
)
EDGES = tuple((INDEX[a], INDEX[b], J) for a, b, J in COUPLINGS)

# Reference constants from Notebook 12, used by tests and notebook assertions.
REFERENCE_Z = 1.7716211099738939
REFERENCE_P_OPERATIONAL = 0.5644547778134884

BIT_WEIGHTS = 2 ** np.arange(N_BITS - 1, -1, -1)


def all_states():
    """All 512 configurations in Notebook 12's lexicographic order (row k has ID k)."""
    return np.array(list(product([0, 1], repeat=N_BITS)), dtype=int)


def state_id(bits):
    """Integer ID of one configuration or of each row of an array of configurations."""
    return np.asarray(bits) @ BIT_WEIGHTS


def bits_of(state):
    """Binary configuration(s) for integer ID(s)."""
    state = np.asarray(state)
    return ((state[..., None] >> np.arange(N_BITS - 1, -1, -1)) & 1).astype(int)


def configuration(disrupted):
    """Binary vector with exactly the named components disrupted."""
    unknown = set(disrupted) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown components: {sorted(unknown)}")
    return np.array([int(name in disrupted) for name in COMPONENTS])


def label(bits):
    """Readable disrupted-component list, e.g. 'S1+S3' or 'none'."""
    names = [c for c, b in zip(COMPONENTS, np.asarray(bits)) if b]
    return "+".join(names) if names else "none"


def energy(bits, theta=THETA, edges=EDGES):
    """E(b) = sum_i theta_i b_i - sum_(i,j) J_ij b_i b_j for one state or an array of states."""
    b = np.asarray(bits)
    pair = sum(J * b[..., i] * b[..., j] for i, j, J in edges)
    return b @ theta - pair


def neighbor_ids():
    """(512, 9) array: entry [s, k] is the ID reached from s by flipping component k."""
    ids = np.arange(N_STATES)
    return np.column_stack([ids ^ (1 << (N_BITS - 1 - k)) for k in range(N_BITS)])


def boltzmann(energies, beta=1.0):
    """Normalized exp(-beta E) over an enumerated energy vector (shifted for stability)."""
    e = np.asarray(energies, dtype=float)
    w = np.exp(-beta * (e - e.min()))
    return w / w.sum()


def exact_target():
    """Enumerate the frozen target: states, energies, probabilities and Z."""
    states = all_states()
    energies = energy(states)
    weights = np.exp(-energies)
    Z = float(weights.sum())
    return {"states": states, "energies": energies, "pi": weights / Z, "Z": Z}


def ising_coefficients(theta=THETA, edges=EDGES):
    """Notebook 12's spin form E(s) = C - sum h_i s_i - sum K_ij s_i s_j with s = 2b - 1."""
    incident = np.zeros(len(theta))
    for i, j, J in edges:
        incident[i] += J
        incident[j] += J
    h = -np.asarray(theta) / 2 + incident / 4
    K = tuple((i, j, J / 4) for i, j, J in edges)
    C = float(np.sum(theta) / 2 - sum(J for _, _, J in edges) / 4)
    return h, K, C


def local_minima(energies, tol=1e-12):
    """IDs whose nine one-bit neighbours all have energy >= their own."""
    e = np.asarray(energies)
    nbr = neighbor_ids()
    return np.flatnonzero((e[nbr] >= e[:, None] - tol).all(axis=1))


FAMILIES = ("Supplier compound", "Infrastructure compound", "Supplier + infrastructure", "Other")


def failure_family(bits):
    """Coarse failure mode used to read tails and count mode switches.

    Supplier compound: at least two of S1-S3 disrupted. Infrastructure compound: at least two
    of SEA, AIR, WH disrupted. Both, or neither ('Other'), otherwise.
    """
    b = np.asarray(bits)
    supplier = b[..., 0:3].sum(axis=-1) >= 2
    infra = b[..., 4:7].sum(axis=-1) >= 2
    code = np.where(supplier & infra, 2, np.where(supplier, 0, np.where(infra, 1, 3)))
    return code
