# Phase 1 submission map

ADB Quantum Innovation Challenge, *Joint-Loss Modelling for Disaster Risk Financing*. Team SMU (ID 6738).
Deadline for the application: 22 November 2026.

## Deliverables required by the brief (Section 4, Phase 1)

| Deliverable | Where |
|---|---|
| Scientific proposal (method, encoding, pre/post-processing, citations) | `docs/proposal.pdf` (built from `docs/proposal.md`), Sections 1-2 and 8 |
| Resource estimation (qubits, depth, two-qubit gates, shots, error mitigation, modality) | `docs/proposal.md` Section 3; `results/real_circuits.md`, `results/ft_estimate.md` |
| Lightweight prototype: source code, environment, hardware/simulator details, AI use | this repository; `README.md` (Reproduce, Environment, AI use); `requirements-lock.txt`; `results/environment.json` |
| Phase 2 plan of action | `docs/proposal.md` Section 4 |
| Team | `docs/proposal.md` Section 6; application form (`docs/application_form.template.md`) |
| Comparison with the benchmark results | `docs/proposal.md` Section 2.9 onward; `results/runs/`; `results/benchmark_submission/` |

## Selection criteria (challenge site, Timeline and Selection Criteria)

| Criterion | Weight | Main evidence |
|---|---:|---|
| Problem & Approach | 25% | proposal Sections 1, 2.15 |
| Technical Rigor | 20% | benchmarks against MC, Sobol, stratified, importance and tuned samplers; `results/baseline_sensitivity.md`; benchmark interface |
| Proof of Concept | 20% | `adbq run`, `adbq twin`, `team_solver.py`; tests |
| Work Plan | 15% | proposal Section 4 |
| Resource Estimation | 10% | proposal Section 3 |
| Team Skills | 10% | proposal Section 6 |

## Benchmark compatibility (joint-loss-baseline)

* `team_solver.py` follows the evaluator interface (EVALUATION.md, section 2) and writes the README submission
  format with `run_id`, `seed`, `model_evaluations` and quantum `evidence`.
* `results/benchmark_submission/<scenario>.json`: one result file per ADB scenario.
* Inputs in `benchmark_inputs/` come from a copy of the benchmark; replace them with the official copy and rerun.
* World Bank track: not covered (the solver exits with status 3 on such a case).

## Before submitting

1. Fill the FILL IN fields of the application form and rebuild: `python docs/build_docs.py`.
2. Rebuild `docs/proposal.pdf` on Windows: `python docs/build_pdf.py`.
3. Run `pytest` and `python experiments/benchmark_submission.py`.
4. Make the GitHub repository public and add its URL to the form.
5. Build the ZIP: `python tools/package_submission.py`.
