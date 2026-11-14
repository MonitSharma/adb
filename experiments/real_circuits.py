"""Real circuits: Qiskit A and Q, verification, compiled resources, noisy simulation,
and a Quantinuum job plan (prepared, NOT submitted).

1. Verification (noiseless): the gate-for-gate Qiskit translation of the two-country A
   (24 qubits) against adbq's exact values for all five targets, and Q^k A for k = 1, 2
   against sin^2((2k+1) theta); the two small hardware sub-circuits likewise; the scalable
   chain circuit for P(L > T) by composition at n = 2, 3 (exact, every source configuration).
2. Compiled resources of the chain circuit, n = 2..10 (Qiskit transpiler): qubits, CX,
   depth, Toffolis, T estimate; next to the analytic resource model of resources.py.
3. Noisy Aer simulation of Q^k A, k = 0, 1, 2, for the two small sub-circuits, with the
   logistics project's published-spec noise models (H2-1, Helios-1): depolarizing gate
   errors and asymmetric readout. For the full 24-qubit A only the predicted success
   probability from its compiled gate counts is reported (it is far too deep).
4. Quantinuum job plan: pytket compilation to the H-series native gate set (offline,
   machine_debug), native gate counts and the published HQC formula per circuit.
   Nothing is submitted; experiments/quantinuum_job.py holds the guarded submission path.

Writes results/real_circuits.json, results/real_circuits.md, results/quantinuum_job_plan.json.
Run: python experiments/real_circuits.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adbq.model import SCENARIOS, SHORT, TARGETS, exact                         # noqa: E402
from adbq.oracle import OBJ, build_oracle                                         # noqa: E402
from adbq.portfolio import chain_portfolio, exact_chain                            # noqa: E402
from adbq.qae import mle_theta                                                    # noqa: E402
from adbq.qiskit_circuits import (amplified, chain_indicator_circuit, compiled_counts,  # noqa: E402
                                  prob_objective, quake_trigger_circuit, rain_joint_circuit,
                                  to_qiskit, verify_chain_arithmetic)
from adbq.resources import estimate                                               # noqa: E402

SEED = 20261117
SHOTS = 4000
SHOTS_HW = 500
KS = (0, 1, 2)


def log(m):
    print(m, flush=True)


def lq():
    """Quantinuum helpers from the team's earlier project (vendored in adbq._vendor)."""
    from adbq._vendor.supplychain_qmcmc import quantinuum
    return quantinuum


def verification():
    out = {"two_country_full_A": {}, "small": {}, "chain": {}}
    s = SCENARIOS["adb-base"]
    for key, short in zip(TARGETS, SHORT):
        o = build_oracle(s, key)
        A = to_qiskit(o.A)
        t0 = time.time()
        a = prob_objective(A, OBJ)
        out["two_country_full_A"][short] = {"qubits": A.num_qubits, "circuit": o.f_max * a,
                                            "exact": exact("adb-base")[key],
                                            "rel_err": abs(o.f_max * a / exact("adb-base")[key] - 1),
                                            "seconds": time.time() - t0}
        log(f"  full A {short}: rel err {out['two_country_full_A'][short]['rel_err']:.1e}")
    o = build_oracle(s, TARGETS[2])
    A = to_qiskit(o.A)
    th = np.arcsin(np.sqrt(exact("adb-base")[TARGETS[2]] / o.f_max))
    out["two_country_full_A"]["grover"] = {
        str(k): abs(prob_objective(amplified(A, OBJ, k), OBJ) - np.sin((2 * k + 1) * th) ** 2) for k in (1, 2)}
    log(f"  full A Grover: {out['two_country_full_A']['grover']}")
    for name, f in (("rain_joint", rain_joint_circuit), ("quake_trigger", quake_trigger_circuit)):
        qc, obj, ex = f()
        th = np.arcsin(np.sqrt(ex))
        out["small"][name] = {"qubits": qc.num_qubits, "exact": ex,
                              "abs_err_k": {str(k): abs(prob_objective(amplified(qc, obj, k), obj)
                                                        - np.sin((2 * k + 1) * th) ** 2) for k in KS}}
    for n in (2, 3):
        pf = chain_portfolio(n)
        qc, obj, regs = chain_indicator_circuit(pf)
        p, mism = verify_chain_arithmetic(pf, qc, obj, regs)
        out["chain"][str(n)] = {"circuit": p, "exact": exact_chain(pf)["P(L>T)"], "mismatches": mism}
    return out


def resources_table():
    rows = []
    for n in range(2, 11):
        pf = chain_portfolio(n)
        qc, _, _ = chain_indicator_circuit(pf)
        cc = compiled_counts(qc)
        model = estimate(n, "P(L>T)")
        rows.append({"n": n, **cc, "model_qubits": model.logical_qubits, "model_toffoli": model.toffoli,
                     "model_t": model.t_count})
        log(f"  n={n}: {cc['qubits']} qubits, {cc['cx']} CX, T~{cc['t_estimate_gidney']}")
    o = build_oracle(SCENARIOS["adb-base"], TARGETS[2])
    full = compiled_counts(to_qiskit(o.A))
    return rows, {"two_country_full_A_truth_table": full}


def noisy():
    import qiskit
    from qiskit_aer import AerSimulator
    q = lq()
    rng = np.random.default_rng(SEED)
    out = {}
    for name, f in (("rain_joint", rain_joint_circuit), ("quake_trigger", quake_trigger_circuit)):
        qc, obj, ex = f()
        th = np.arcsin(np.sqrt(ex))
        out[name] = {"exact": ex, "devices": {}}
        for dev in ("H2-1", "Helios-1"):
            spec = q.SPECS[dev]
            sim = AerSimulator(noise_model=q.aer_noise_model(spec), seed_simulator=int(rng.integers(2**31)))
            rows = []
            hits = []
            for k in KS:
                c = qiskit.transpile(amplified(qc, obj, k, measure=True), basis_gates=["cx", "u"], optimization_level=1)
                ops = c.count_ops()
                counts = sim.run(c, shots=SHOTS).result().get_counts()
                h = counts.get("1", 0)
                hits.append(h)
                n1, n2 = sum(v for g, v in ops.items() if g == "u"), ops.get("cx", 0)
                rows.append({"k": k, "cx": int(n2), "u": int(n1), "ideal": float(np.sin((2 * k + 1) * th) ** 2),
                             "noisy": h / SHOTS, "pred_no_error": float(1 - q.effective_depolarizing(spec, n1, n2))})
            ms = np.array(KS)
            a_mle = float(np.sin(mle_theta(ms, np.array(hits), SHOTS)) ** 2)
            a_k0 = hits[0] / SHOTS
            out[name]["devices"][dev] = {"rows": rows, "a_mle_k012": a_mle, "a_k0_only": a_k0,
                                         "mle_rel_err": a_mle / ex - 1}
            log(f"  {name} {dev}: MLE {a_mle:.5f} vs exact {ex:.5f}")
    # full two-country A: predicted probability that no gate error occurs
    o = build_oracle(SCENARIOS["adb-base"], TARGETS[2])
    full = qiskit.transpile(to_qiskit(o.A), basis_gates=["cx", "u"], optimization_level=1).count_ops()
    out["full_A_prediction"] = {dev: {"cx": int(full.get("cx", 0)),
                                      "p_no_error_A": float(1 - q.effective_depolarizing(q.SPECS[dev], full.get("u", 0), full.get("cx", 0)))}
                                for dev in ("H2-1", "Helios-1")}
    return out


def job_plan():
    """pytket native compilation and HQC per circuit: 2 small circuits x k = 0, 1, 2."""
    import qiskit
    q = lq()
    items = []
    for name, f in (("rain_joint", rain_joint_circuit), ("quake_trigger", quake_trigger_circuit)):
        qc, obj, ex = f()
        for k in KS:
            c = qiskit.transpile(amplified(qc, obj, k, measure=True), basis_gates=["cx", "u", "measure"],
                                 optimization_level=1)
            native = q.compile_native(q.to_tket(c), device="H2-1")
            g = q.gate_counts(native)
            items.append({"circuit": name, "k": k, "qubits": native.n_qubits, "native_1q": g["1q"],
                          "native_2q": g["2q"], "depth_2q": g["depth_2q"], "shots": SHOTS_HW,
                          "hqc": q.hqc(g["1q"], g["2q"], g["spam"], SHOTS_HW)})
            log(f"  {name} k={k}: {g['2q']} ZZ, HQC {items[-1]['hqc']:.1f}")
    rain = sum(i["hqc"] for i in items if i["circuit"] == "rain_joint")
    return {"device_compiled_for": "H2-1 (pytket-quantinuum, machine_debug, offline)",
            "recommended": {"circuits": "rain_joint, k = 0, 1, 2", "hqc": rain},
            "hqc_formula": "HQC = 5 + C (N1q + 10 N2q + 5 Nm) / 5000 per job item (published H-series model)",
            "shots_per_circuit": SHOTS_HW, "items": items, "total_hqc": sum(i["hqc"] for i in items),
            "status": "PREPARED - NOT SUBMITTED. Hardware runs need explicit approval.",
            "note": "Helios pricing and limits may differ; get a syntax-checker quote before any run."}


def render(d):
    v = d["verification"]
    L = ["# Real circuits: verification, resources, noise, hardware plan", "",
         "Generated by `experiments/real_circuits.py`.", "",
         "## Verification (noiseless, Qiskit Aer statevector)", "",
         "| circuit | qubits | quantity | circuit | exact | error |", "|---|---:|---|---:|---:|---:|"]
    for k, r in v["two_country_full_A"].items():
        if k == "grover":
            continue
        L.append(f"| two-country A | {r['qubits']} | {k} | {r['circuit']:.6g} | {r['exact']:.6g} | {r['rel_err']:.1e} (rel) |")
    for k, e in v["two_country_full_A"]["grover"].items():
        L.append(f"| two-country Q^{k} A | 24 | P(obj=1) vs sin^2((2k+1)theta) | | | {e:.1e} (abs) |")
    for name, r in v["small"].items():
        for k, e in r["abs_err_k"].items():
            L.append(f"| {name} Q^{k} A | {r['qubits']} | P(obj=1) vs sin^2 | | {r['exact']:.6g} (a) | {e:.1e} (abs) |")
    for n, r in v["chain"].items():
        L.append(f"| chain n={n}, P(L>T), by composition | - | all source configurations | {r['circuit']:.6g} | "
                 f"{r['exact']:.6g} | {r['mismatches']} mismatches |")
    L += ["", "## Compiled resources, chain A for P(L > T) (one application of A)", "",
          "| n | qubits | CX | depth | Toffoli | T (Gidney) | T (textbook) | model qubits | model T |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in d["resources"]:
        L.append(f"| {r['n']} | {r['qubits']} | {r['cx']:,} | {r['depth']:,} | {r['ccx']:,} | {r['t_estimate_gidney']:,} | "
                 f"{r['t_estimate_textbook']:,} | {r['model_qubits']} | {r['model_t']:,} |")
    f = d["resources_full"]["two_country_full_A_truth_table"]
    L += ["", f"Two-country A as translated gate for gate (truth-table arithmetic): {f['qubits']} qubits, "
          f"{f['cx']:,} CX, T ~ {f['t_estimate_gidney']:,}. The library-arithmetic circuit uses more qubits "
          "(carries, comparator ancillas) but scales linearly. Arbitrary rotations are costed at "
          "3 log2(1/delta) T each (delta = 1e-10); Toffoli at 4 T (Gidney) or 7 T (textbook). "
          "The analytic model of resources.py assumes leaner, uncompiled constructions.", "",
          "## Noisy simulation of Q^k A (Aer, published-spec noise models)", "",
          "| circuit | device model | k | CX | ideal P(1) | noisy P(1) | P(no gate error) |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for name, r in d["noisy"].items():
        if name == "full_A_prediction":
            continue
        for dev, x in r["devices"].items():
            for row in x["rows"]:
                L.append(f"| {name} | {dev} | {row['k']} | {row['cx']} | {row['ideal']:.4f} | {row['noisy']:.4f} | "
                         f"{row['pred_no_error']:.3f} |")
    L += ["", "| circuit | device model | exact a | MLE from k = 0, 1, 2 | relative error |", "|---|---|---:|---:|---:|"]
    for name, r in d["noisy"].items():
        if name == "full_A_prediction":
            continue
        for dev, x in r["devices"].items():
            L.append(f"| {name} | {dev} | {r['exact']:.5f} | {x['a_mle_k012']:.5f} | {x['mle_rel_err']:+.1%} |")
    fp = d["noisy"]["full_A_prediction"]
    L += ["", "Full two-country A (24 qubits): " + "; ".join(
        f"{dev}: {x['cx']:,} CX, P(no gate error in one A) = {x['p_no_error_A']:.1e}" for dev, x in fp.items())
          + ". The full model is not a near-term hardware circuit; it needs fault tolerance.", "",
          f"{SHOTS} shots per point. Noise models are the logistics project's illustrative published-spec models, "
          "not device measurements.", "",
          "## Quantinuum job plan (prepared, not submitted)", "",
          "| circuit | k | qubits | native 1q | native ZZ | 2q depth | shots | HQC |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for it in d["job_plan"]["items"]:
        L.append(f"| {it['circuit']} | {it['k']} | {it['qubits']} | {it['native_1q']} | {it['native_2q']} | {it['depth_2q']} | "
                 f"{it['shots']} | {it['hqc']:.1f} |")
    jp = d["job_plan"]
    rain = sum(i["hqc"] for i in jp["items"] if i["circuit"] == "rain_joint")
    L += ["", f"All six circuits: {jp['total_hqc']:.0f} HQC ({jp['hqc_formula']}).",
          "", f"**Recommended first run: the rain sub-circuit only, k = 0, 1, 2: {rain:.0f} HQC.** In the noisy "
          "simulation it recovers the amplitude within a few percent; the quake sub-circuit is dominated "
          "by gate noise at k >= 1 and would spend most of the budget on noise "
          "(experiments/quantinuum_job.py runs it only with --include-quake).",
          "", f"{jp['status']} {jp['note']}"]
    return "\n".join(L) + "\n"


def main():
    d = {"seed": SEED}
    log("verification"); d["verification"] = verification()
    log("resources"); d["resources"], d["resources_full"] = resources_table()
    log("noisy"); d["noisy"] = noisy()
    log("job plan"); d["job_plan"] = job_plan()
    return d


if __name__ == "__main__":
    d = main()
    (ROOT / "results" / "real_circuits.json").write_text(json.dumps(d, indent=1))
    (ROOT / "results" / "quantinuum_job_plan.json").write_text(json.dumps(d["job_plan"], indent=1))
    (ROOT / "results" / "real_circuits.md").write_text(render(d), encoding="utf-8")
    print(render(d))
