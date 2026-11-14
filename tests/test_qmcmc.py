import itertools

import numpy as np
import pytest

from adbq.grid import calibrated_grid
from adbq.qmcmc import (QuantumProposal, Target, Wolff, evolve_basis, logistics, masks,
                        mismatched_landscape, quantum_params, run_chains, run_tempering,
                        steps_to_accuracy)


@pytest.fixture(scope="module")
def small():
    m = calibrated_grid(2, 3, kappa=3.0)
    return m, Target(m, lam=1.0)


def test_fast_emulator_matches_logistics_trotter(small):
    _, q = logistics()
    m, t = small
    ids = np.arange(2**m.n)
    U = q.trotter_unitary(t.E_lam, 0.4, 3.7, 5)
    X = evolve_basis(t.E_lam, ids, 0.4, 3.7, 5, q.layden_alpha(t.E_lam))
    assert np.abs(U - X).max() < 1e-12


def test_quantum_proposal_is_symmetric_and_stochastic(small):
    m, t = small
    rng = np.random.default_rng(0)
    for E_h in (t.E_lam, mismatched_landscape(m, t.lam, rng)):
        Q = QuantumProposal(E_h, quantum_params("default", rng, 4), steps=4).rows(np.arange(2**m.n))
        assert np.abs(Q - Q.T).max() < 1e-13
        assert np.allclose(Q.sum(1), 1)


def test_quantum_pool_sampler_draws_from_the_row(small):
    m, t = small
    rng = np.random.default_rng(1)
    qp = QuantumProposal(t.E_lam, quantum_params("short", rng, 3), steps=4, pool=256)
    row = qp.rows(np.array([5]))[0]
    draws = np.concatenate([qp.propose(np.full(400, 5), rng) for _ in range(20)])
    emp = np.bincount(draws, minlength=len(row)) / len(draws)
    assert 0.5 * np.abs(emp - row).sum() < 0.03


def _wolff_matrix(m, E):
    """Exact Wolff-MH transition matrix by enumerating seeds and bond configurations."""
    n, N = m.n, 2**m.n
    w = Wolff(m)
    edges = list(zip(m.edge_list, m.J))
    P = np.zeros((N, N))
    for s in range(N):
        b = w.bits(s)
        for seed in range(n):
            # bonds that can open: equal bits; enumerate open/closed for each
            cand = [(i, j, 1 - np.exp(-Jv / 2), Jv) for (i, j), Jv in edges if b[i] == b[j]]
            for open_ in itertools.product((0, 1), repeat=len(cand)):
                pr = np.prod([p if o else 1 - p for (_, _, p, _), o in zip(cand, open_)]) / n
                adj = {k: [] for k in range(n)}
                for (i, j, _, _), o in zip(cand, open_):
                    if o:
                        adj[i].append(j); adj[j].append(i)
                cl, st = {seed}, [seed]
                while st:
                    for j in adj[st.pop()]:
                        if j not in cl:
                            cl.add(j); st.append(j)
                s2 = s ^ sum(1 << (n - 1 - k) for k in cl)
                dEc = sum(0.5 * Jv * (2 * b[i] - 1) * (2 * b[j] - 1) for (i, j), Jv in edges
                          if (i in cl) != (j in cl))
                a = min(1.0, np.exp(-(E[s2] - E[s] - dEc)))
                P[s, s2] += pr * a
                P[s, s] += pr * (1 - a)
    return P


def test_wolff_satisfies_detailed_balance_exactly():
    m = calibrated_grid(2, 2, kappa=3.0)
    t = Target(m, lam=1.5)
    P = _wolff_matrix(m, t.E_lam)
    kern, _ = logistics()
    aud = kern.audit(P, t.pi_lam)
    assert aud["row_sum"] < 1e-12 and aud["detailed_balance"] < 1e-14 and aud["stationarity"] < 1e-13


@pytest.mark.parametrize("moves", [
    [(1.0, "local")], [(0.5, "local"), (0.5, "pair")], [(0.5, "local"), (0.5, "plaquette")],
    [(0.5, "local"), (0.5, "wolff")], [(0.5, "local"), (0.5, "uniform")]])
def test_chains_target_the_tilted_law(small, moves):
    m, t = small
    rng = np.random.default_rng(2)
    S, acc = run_chains(t, moves, np.zeros(400, int), 600, rng)
    emp = np.bincount(S[:, 100:].ravel(), minlength=2**m.n) / S[:, 100:].size
    assert 0.5 * np.abs(emp - t.pi_lam).sum() < 0.03


def test_quantum_chain_and_tempering_target_the_law(small):
    m, t = small
    rng = np.random.default_rng(3)
    qp = QuantumProposal(t.E_lam, quantum_params("short", rng, 3), steps=4)
    S, _ = run_chains(t, [(0.5, "local"), (0.5, qp)], np.zeros(300, int), 400, rng)
    emp = np.bincount(S[:, 100:].ravel(), minlength=2**m.n) / S[:, 100:].size
    assert 0.5 * np.abs(emp - t.pi_lam).sum() < 0.03
    C = run_tempering(t, [(0.5, "local"), (0.5, "pair")], [1.0, 0.6, 0.3], np.zeros(300, int), 300, rng)
    emp = np.bincount(C[:, 50:].ravel(), minlength=2**m.n) / C[:, 50:].size
    assert 0.5 * np.abs(emp - t.pi_lam).sum() < 0.03


def test_importance_corrected_cvar_and_exact_steps(small):
    m, t = small
    t0 = Target(m, lam=0.0)
    assert t.cvar == t0.cvar                                # tilt does not change the target CVaR
    rng = np.random.default_rng(4)
    S = rng.choice(2**m.n, size=200_000, p=t.pi_lam)
    assert t.cvar_from_states(S) == pytest.approx(t.cvar, rel=0.01)
    kern, _ = logistics()
    s_loc = steps_to_accuracy(kern.local_proposal(m.n), t)
    assert s_loc > t.iid_variance() / (0.1 * t.cvar) ** 2   # a Markov chain cannot beat i.i.d. here
