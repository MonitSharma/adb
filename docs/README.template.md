# Quantum amplitude estimation for joint catastrophe-bond losses

Phase 1 project for the ADB Quantum Innovation Challenge, *Joint-Loss Modelling for Disaster
Risk Financing* (Kyrgyz Republic / Tajikistan disaster relief bond). Proposal draft:
[`docs/proposal.md`](docs/proposal.md).

**Question.** Can quantum methods estimate the joint loss and payout risk of a multi-country
catastrophe-bond portfolio more accurately per query than strong classical samplers, and does
that hold as the portfolio grows?

![Accuracy per query](docs/figures/accuracy_per_query.png)

## Results at a glance

Base scenario, queries to 1% relative RMSE (one query = one sampled term, or one application of A):

{{table_accuracy_base}}

* Across the five scenarios, IQAE needs {{acc_prob_lo}}-{{acc_prob_hi}} fewer queries than the best
  classical sampler (including a loss tilt tuned per target) on the joint-payout and exceedance
  probabilities. The tuned tilt beats it on stop-loss ({{acc_stop_lo}}-{{acc_stop_hi}}), and
  Sobol wins on E[L].
* Rarest target: IQAE needs {{acc_rare_brief_lo}}-{{acc_rare_brief_hi}} fewer queries than the brief's samplers.
  When both sides condition on the independent rain triggers and the classical side uses a tuned tilt, the
  margin shrinks to {{acc_rare_lo}}-{{acc_rare_hi}}.
* Exact VaR99 = {{var99_base}}, CVaR99 = {{cvar99_base}} (base). CVaR99 moves from {{cvar99_indep}}
  (independent) to {{cvar99_common}} (common shock); VaR95 stays at {{var95_base}}.
* On 2-D grids with shared peril streams, exact computation stops at {{grid_exact_rows}} rows (the
  nearest-neighbour Ising grid, a different model, contracts exactly to {{ising_tm_rows_max}} x {{ising_tm_rows_max}}). IQAE keeps
  an edge on a moderate tail ({{grid_t3_last}} at {{grid_last}}) and loses to importance sampling
  on rarer tails ({{grid_t5_last}}).
* Quantum-enhanced MCMC beats local MCMC under high contagion, but Wolff cluster moves beat every
  quantum sampler (the best quantum sampler needs {{qmcmc_k4_quantum_needs}} the evaluations).
* Certified IQAE intervals covered the exact value in at least {{cov_min}} of {{cov_runs}} runs per cell
  (nominal {{cov_nominal}}). Clopper-Pearson misses can be confidently wrong (worst {{cov_worst_miss}}), so the
  digital model uses Chernoff-Hoeffding bounds (worst miss in the digital model {{ti_ch_worst}}, at {{ti_cost_median}} the queries).
* **Digital model.** Interval transport saves {{twin_save_lo}}-{{twin_save_hi}} of the queries on a fine
  shared-quake sweep at 10% half-width; a conditional tail-scenario library moved by exact thinning
  saves {{lib_saving}}; budget allocation cuts weighted error by {{alloc_vs_prop}} against a
  difficulty-proportional split. Classical paired MC beats the paired quantum circuit for structure
  choice ({{crn_win_lo}}-{{crn_win_hi}}).
* The Qiskit circuit matches exact values to {{qiskit_full_A_err}}. The full A needs fault tolerance
  ({{full_A_cx}} CX; surface-code estimate up to {{ft_worst_qubits}} physical qubits and
  {{ft_worst_days}} for the rarest target at 1%); a 6-qubit sub-circuit is costed at {{hqc_recommended}} HQC.
  **Nothing has been submitted to hardware.**

Claim level: accuracy per query in noiseless emulation; utility, not advantage. One quantum
query is about {{t_per_query_n2}} T gates at n = 2. Negative results are in
[`docs/proposal.md`](docs/proposal.md#6-claim-boundary-and-negative-results).

## Run a portfolio

```bash
python -m adbq run configs/adb-base.yaml        # or: pip install -e . && adbq run configs/*.yaml
python -m adbq validate configs/*.yaml
```

The digital model (certified answers, a dependence sweep with interval transport, a structure
comparison, and a conditional tail-scenario library with drivers, all with a query ledger):

```bash
python -m adbq twin configs/twin-adb-base.yaml  # writes results/twin_runs/<name>/report.json, summary.md
```

Each run writes `results/runs/<name>/report.json`, `summary.md` and figures. Shipped configs:

{{table_runs}}

## Challenge benchmark interface

The challenge's benchmark (`joint-loss-baseline`, CLI `qjlb`) evaluates a team solver on private cases:
`team_solver.py --request request.json --output result.json`. This repository's solver handles the ADB
track for any hold-out case (term of one to ten years, per-country sub-limits, rain inputs and
threshold) and checks its own output against the evaluator's rules before writing it.

```bash
python team_solver.py --request request.json --output result.json        # IQAE, quantum_simulator
python team_solver.py --request request.json --output result.json --method exact   # classical_cpu
python experiments/benchmark_submission.py   # results/benchmark_submission/<scenario>.json + summary.md
```

* `problem_id` is computed exactly as the benchmark computes it; the shipped `adb-base` result carries the
  benchmark's own published ID.
* The interval width of each IQAE run is set from the evaluator's tolerance for that target, so every
  target passes its RMSE check (see `tests/test_benchmark.py`).
* Evidence: `circuit_sha256` hashes a canonical specification of the state preparation; the backend is
  the emulator in this repository. No hardware is used.
* Not covered yet: the World Bank event-loss-table track (`worldbank_elt`); such a case exits with status 3.

Inputs: [`benchmark_inputs/`](benchmark_inputs/README.md). Results: [`summary.md`](results/benchmark_submission/summary.md).
Phase 1 deliverables map: [`SUBMISSION.md`](SUBMISSION.md).

## What is in here

| module | contents |
|---|---|
| `model.py` | the brief's model; annual CTMC and independent-source routes; Table 5 |
| `baselines.py`, `risk_baselines.py` | MC, Sobol, stratified, the brief's importance sampler, tuned and rain-conditioned loss tilts, all with exact variances |
| `circuits.py`, `oracle.py` | gate-level A, exact sparse simulator |
| `qae.py`, `risk_qae.py` | IQAE (Clopper-Pearson or Chernoff-Hoeffding), MLAE; VaR bisection and CVaR with IQAE |
| `risk.py` | exact loss laws (two-country, chain), VaR and CVaR |
| `portfolio.py` | n-country chain and its exact transfer-matrix contraction |
| `grid.py`, `qmcmc.py` | calibrated Ising trigger grid; quantum-enhanced MCMC |
| `twin.py` | the certified portfolio digital model (`adbq twin`) |
| `_vendor/supplychain_qmcmc` | four modules from the team's earlier project `logistics_under_disruption`, unchanged |
| `gridsources.py`, `grid_estimators.py` | 2-D shared-peril grid, exact sweep, MC / RQMC / tilt / cross-entropy, IQAE |
| `qiskit_circuits.py` | Qiskit A and Q, compiled counts, verification by composition |
| `resources.py` | logical resource model |
| `twin.py` | digital model: versioned source laws, exact ratio bounds, interval transport, scenario library, paired comparisons, allocation, ledger |
| `pipeline.py`, `cli.py` | `adbq run config.yaml` |
| `benchmark.py`, `team_solver.py` | the challenge benchmark interface: general ADB cases, IQAE with tolerance-driven intervals, result checks |

Generated result summaries: [`risk_summary.md`](results/risk_summary.md),
[`grid_summary.md`](results/grid_summary.md), [`qmcmc_exact.md`](results/qmcmc_exact.md),
[`qmcmc_chains.md`](results/qmcmc_chains.md), [`grid_qae.md`](results/grid_qae.md),
[`real_circuits.md`](results/real_circuits.md), [`coverage.md`](results/coverage.md),
[`twin.md`](results/twin.md), [`ft_estimate.md`](results/ft_estimate.md),
[`baseline_sensitivity.md`](results/baseline_sensitivity.md).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # or requirements-lock.txt for the exact versions used
pytest -q                                             # fast suite; pytest -m slow for the 24-qubit check
python experiments/environment.py                     # records versions and machine -> results/environment.json
python experiments/accuracy_per_query.py              # two-country accuracy per query
python experiments/scaling.py                         # chain scaling and resource model
python experiments/cvar.py                            # exact VaR / CVaR (second implementation)
python experiments/risk_accuracy.py && python experiments/risk_report.py
python experiments/grid_exact.py                      # Ising grid, exact
python experiments/qmcmc_exact.py                     # QeMCMC exact scorecard (3 x 4)
python experiments/qmcmc_chains.py --default          # QeMCMC chains (4 x 4), about an hour
python experiments/grid_qae.py                        # high-width comparison, about an hour
python experiments/real_circuits.py                   # Qiskit / pytket / noisy Aer
python experiments/coverage.py                        # certified-interval coverage
python experiments/twin.py                            # digital-model experiments
python experiments/twin_intervals.py                  # interval method for the digital model (CP vs CH)
python experiments/ft_estimate.py                     # surface-code estimate
python experiments/benchmark_submission.py           # benchmark-format result files (ADB track)
python experiments/baseline_sensitivity.py            # open baseline settings (strata allocation, IS mix)
python experiments/quantinuum_job.py                  # dry run only: native counts and HQC
python -m adbq run configs/*.yaml                      # digital-model configs are skipped; use adbq twin
python -m adbq twin configs/twin-adb-base.yaml
python experiments/make_figures.py
python experiments/make_paper_figures.py                # publication figures -> docs/figures/ (PDF + 600 dpi PNG)
python docs/build_docs.py                             # README.md and docs/proposal.md from results
python docs/build_pdf.py                              # docs/proposal.pdf (Windows with Microsoft Word)
python tools/package_submission.py                    # submission ZIP = the files git would publish
```

**Installing the quantum extras.** `pytket-qiskit` pulls in `ibm-platform-services`, which is published as a
source package whose build needs `pkg_resources`, removed in setuptools 81. If `pip install` fails with
`No module named 'pkg_resources'`, build it against an older setuptools:

```bash
pip install "setuptools<81"
pip install --no-build-isolation -r requirements-lock.txt
```

With uv: `uv pip install --build-constraint constraints.txt -r requirements-lock.txt`, where `constraints.txt`
contains `setuptools<81`. The core model, baselines, IQAE and the digital model need only NumPy, SciPy, Matplotlib and
PyYAML.

The repository is self-contained: the QeMCMC track and the Quantinuum helpers use four modules from the
team's earlier project `logistics_under_disruption`, vendored unchanged in `src/adbq/_vendor/` (set
`ADBQ_LOGISTICS_SRC` to use an external checkout instead).

## Environment, hardware and simulators

* **Machine:** {{env_cpu_line}}; no GPU. Measured run times of the heaviest steps are in
  [`docs/proposal.md`](docs/proposal.md#36-classical-compute-and-fit-to-the-timeline).
* **Packages:** NumPy {{env_numpy}}, SciPy {{env_scipy}}, Matplotlib {{env_matplotlib}}, PyYAML {{env_pyyaml}},
  pytest {{env_pytest}}; quantum extras Qiskit {{env_qiskit}}, Qiskit Aer {{env_qiskit_aer}}, pytket {{env_pytket}},
  pytket-qiskit {{env_pytket_qiskit}}, pytket-quantinuum {{env_pytket_quantinuum}} (exact list:
  `requirements-lock.txt`, written by `experiments/environment.py`).
* **Simulators:** adbq's own exact sparse state-vector simulator (`circuits.py`, no SDK) for every verified
  amplitude; Qiskit Aer for state-vector checks of the Qiskit circuits and for noisy shot simulation with
  published-spec noise models; pytket-quantinuum for offline compilation to the H-series gate set.
* **Quantum hardware:** none used. A small Quantinuum job is compiled and costed (`experiments/quantinuum_job.py`,
  dry run by default) and has not been submitted.
* **Sample outputs:** `results/` (JSON and Markdown summaries, figures, and one folder per pipeline run in
  `results/runs/` and `results/twin_runs/`).

## Baseline settings the brief leaves open

The brief mentions an open benchmark repository; the challenge site does not link one publicly, so the
baselines follow the brief's text (Section 3b, step 9). Two details are open, and
`experiments/baseline_sensitivity.py` ([`baseline_sensitivity.md`](results/baseline_sensitivity.md)) shows
they do not move the results:

* Stratified sampling: we ship proportional allocation. Equal and Neyman (variance-optimal) allocation never
  become the best classical sampler; Neyman bounds every possible allocation.
* Importance sampling: the brief's half-and-half mix of p and the loss-tilted law. Sweeping the tilt weight
  from {{bs_mix_lo}} to {{bs_mix_hi}} changes the best-classical-to-IQAE ratio by at most {{bs_max_shift}};
  targets where any variant replaces the best classical sampler by a margin above 1%: {{bs_changed}}.
* The n-country chain and the 2-D grid families are ours; the brief defines only n = 2.

When ADB's repository is available, the settings are configuration parameters and every result reruns.

## AI use

The code was assisted by Claude (Anthropic).
