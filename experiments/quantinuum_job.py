"""Quantinuum job for Q^k A, k = 0, 1, 2, on small sub-circuits of the ADB model (default:
the 6-qubit joint rain trigger; --include-quake adds the 15-qubit joint quake trigger).

STATUS: prepared, never submitted from this repository. The default mode is a dry run.

Modes
-----
  (default) --dry-run   Compile locally to the H-series native gate set (pytket-quantinuum,
                        machine_debug: no account, no network), print native gate counts and
                        the published HQC estimate. Nothing leaves this machine.
  --quote               Upload the circuits to Quantinuum Nexus and run them on the syntax
                        checker (<device>SC) for a cost quote. Network access; no hardware.
  --emulator            Run on the hosted emulator (<device>E). Costs credits.
  --hardware            Run on hardware. Costs HQC.

Every network mode requires BOTH --approved (explicit sign-off by the project owner for this
specific run) and --max-hqc (a hard cap; the job is refused if the local estimate exceeds it).
Authentication is qnexus's own browser login; no credentials are stored or printed here.

The Nexus calls follow the qnexus API used in the sibling logistics project; this repository
has only exercised the dry run. Check the qnexus version before a real run.

Run: python experiments/quantinuum_job.py [--dry-run | --quote | --emulator | --hardware]
         [--device H2-1] [--shots 500] [--approved --max-hqc 300]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from real_circuits import KS, lq                                                    # noqa: E402
from adbq.qiskit_circuits import amplified, quake_trigger_circuit, rain_joint_circuit  # noqa: E402


def build(device: str, shots: int, include_quake: bool = False):
    import qiskit
    q = lq()
    jobs = []
    circuits = [("rain_joint", rain_joint_circuit)]
    if include_quake:          # not recommended: gate noise dominates at k >= 1 (results/real_circuits.md)
        circuits.append(("quake_trigger", quake_trigger_circuit))
    for name, f in circuits:
        qc, obj, ex = f()
        for k in KS:
            c = qiskit.transpile(amplified(qc, obj, k, measure=True), basis_gates=["cx", "u", "measure"],
                                 optimization_level=1)
            tk = q.to_tket(c)
            native = q.compile_native(tk, device=device)
            g = q.gate_counts(native)
            jobs.append({"name": f"adbq-{name}-k{k}", "tket": tk, "native": native, "exact_a": ex, "k": k,
                         "counts": g, "hqc": q.hqc(g["1q"], g["2q"], g["spam"], shots)})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--quote", action="store_true")
    mode.add_argument("--emulator", action="store_true")
    mode.add_argument("--hardware", action="store_true")
    ap.add_argument("--device", default="H2-1")
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--approved", action="store_true", help="explicit owner sign-off for this run")
    ap.add_argument("--max-hqc", type=float, default=None)
    ap.add_argument("--include-quake", action="store_true", help="also the 15-qubit quake sub-circuit (costly, noisy)")
    a = ap.parse_args()
    jobs = build(a.device, a.shots, a.include_quake)
    total = sum(j["hqc"] for j in jobs)
    for j in jobs:
        print(f"{j['name']:28s} qubits {j['native'].n_qubits:3d}  ZZ {j['counts']['2q']:4d}  HQC {j['hqc']:.1f}")
    print(f"estimated total: {total:.1f} HQC for {len(jobs)} circuits x {a.shots} shots on {a.device}")
    network = a.quote or a.emulator or a.hardware
    if not network:
        print("dry run only: nothing submitted.")
        return
    if not a.approved or a.max_hqc is None:
        sys.exit("refused: network modes need --approved and --max-hqc (hardware runs require the owner's approval).")
    if total > a.max_hqc:
        sys.exit(f"refused: estimate {total:.1f} HQC exceeds --max-hqc {a.max_hqc}.")
    import qnexus as qnx
    from qnexus.models import QuantinuumConfig
    if not qnx.auth.is_logged_in():
        qnx.auth.login()
    target = a.device + ("SC" if a.quote else "E" if a.emulator else "")
    cfg = QuantinuumConfig(device_name=target)
    project = qnx.projects.get_or_create(name="adb-joint-loss-qae")
    refs = [qnx.circuits.upload(circuit=j["tket"], name=j["name"], project=project) for j in jobs]
    comp = qnx.compile(programs=refs, backend_config=cfg, name="adbq-compile", project=project, optimisation_level=2)
    job = qnx.start_execute_job(programs=comp, n_shots=[a.shots] * len(comp), backend_config=cfg,
                                name=f"adbq-qae-k012-{target}", project=project)
    rec = {"job_id": str(job.id), "target": target, "shots": a.shots, "estimated_hqc": total, "max_hqc": a.max_hqc}
    out = ROOT / "results" / f"quantinuum_job_{target}.json"
    out.write_text(json.dumps(rec, indent=1))
    print(f"submitted {rec}; record in {out}")


if __name__ == "__main__":
    main()
