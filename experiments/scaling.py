"""Scaling study for n-country chain portfolios.

1. Exact generalised targets by transfer-matrix contraction (verified against full
   CTMC enumeration for n = 2, 3, 4) and its wall-clock time.
2. Logical resources of A per n (adbq.resources) and the queries IQAE and plain Monte
   Carlo need for 1% relative error on the rare tail target P(>=2 exhausted).

Writes results/scaling.json. Run: python experiments/scaling.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adbq.portfolio import GEN_TARGETS, brute_force_chain, chain_portfolio, exact_chain  # noqa: E402
from adbq.resources import estimate, iqae_queries, mc_samples                           # noqa: E402

NS = [2, 3, 4, 5, 10, 20, 50]


def main():
    rows = []
    for n in NS:
        pf = chain_portfolio(n)
        t0 = time.perf_counter()
        ex = exact_chain(pf)
        t_tm = time.perf_counter() - t0
        t_bf = None
        if n <= 4:
            t0 = time.perf_counter()
            bf = brute_force_chain(pf)
            t_bf = time.perf_counter() - t0
            assert all(abs(ex[k] - bf[k]) <= 1e-9 * abs(bf[k]) + 1e-300 for k in ex)
        res = {k: estimate(n, k).as_dict() for k in GEN_TARGETS}
        a_tail = ex["P(>=2 exhausted)"]
        rows.append({
            "n": n, "exact": ex, "t_transfer_s": t_tm, "t_bruteforce_s": t_bf,
            "annual_states_naive": 10**n, "resources": res,
            "tail_amplitude": a_tail,
            "iqae_queries_1pct": iqae_queries(a_tail, 0.01),
            "mc_samples_1pct": mc_samples(a_tail, 0.01),
        })
        r = res["P(>=2 exhausted)"]
        print(f"n={n:3d} tail={a_tail:.3e} exact {t_tm:6.2f}s  qubits={r['logical_qubits']:5d} "
              f"Toffoli={r['toffoli']:6d} T={r['t_count']:8d}  IQAE 1%: {rows[-1]['iqae_queries_1pct']:.2e}  "
              f"MC 1%: {rows[-1]['mc_samples_1pct']:.2e}", flush=True)
    (ROOT / "results" / "scaling.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
