#!/usr/bin/env python3
"""Team solver for the challenge evaluator (joint-loss-baseline EVALUATION.md, section 2).

    python team_solver.py --request request.json --output result.json [--method iqae|exact]

The evaluator runs this in a fresh directory with a request naming one case file. The
default method is iterative QAE on the emulated state preparation (execution
``quantum_simulator``); ``--method exact`` gives the classical exact reference
(``classical_cpu``). Only the ``adb_drb`` track is supported; a ``worldbank_elt`` case
exits with status 3 and a message on stderr.
"""
import time

START = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from adbq.benchmark import case_from_config, check_result, solve  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--request", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--method", choices=["iqae", "exact"], default="iqae")
    args = ap.parse_args()
    request = json.loads(Path(args.request).read_text())
    base = Path(args.request).resolve().parent
    try:
        case = case_from_config(base / request["config_file"])
    except NotImplementedError as error:
        print(f"team_solver: {error}", file=sys.stderr)
        return 3
    if case.problem_id != request["problem_id"]:
        raise ValueError("input problem ID does not match")
    prepared = time.perf_counter()
    out = solve(case, int(request["seed"]), args.method)
    sub = {"problem_id": case.problem_id, "method": out["method"], "execution": out["execution"],
           "estimates": out["estimates"], "intervals": out["intervals"],
           "resources": {"elapsed_seconds": 0.0, "preparation_seconds": prepared - START,
                         **out["resources"]},
           "run_id": request["run_id"], "seed": request["seed"]}
    if "evidence" in out:
        sub["evidence"] = out["evidence"]
    sub["resources"]["elapsed_seconds"] = time.perf_counter() - START
    check_result(sub, case, request)
    Path(args.output).write_text(json.dumps(sub, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
