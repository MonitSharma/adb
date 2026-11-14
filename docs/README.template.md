# Quantum amplitude estimation for joint catastrophe-bond losses

Phase 1 project for the ADB Quantum Innovation Challenge, *Joint-Loss Modelling for Disaster
Risk Financing* (Kyrgyz Republic / Tajikistan disaster relief bond). Proposal draft:
[`docs/proposal.md`](docs/proposal.md).

**Question.** Can quantum methods estimate the joint loss and payout risk of a multi-country
catastrophe-bond portfolio more accurately per query than strong classical samplers, and does
that hold as the portfolio grows?

![Accuracy per query](results/figures/accuracy_per_query.png)

## Results at a glance

Base scenario, queries to 1% relative RMSE (one query = one sampled term, or one application of A):

{{table_accuracy_base}}

* Across the five scenarios IQAE needs {{acc_mod_lo}}-{{acc_mod_hi}} fewer queries on the moderate
  targets and {{acc_rare_lo}}-{{acc_rare_hi}} fewer on the rare tail; Sobol wins on E[L].
* Exact VaR99 = {{var99_base}}, CVaR99 = {{cvar99_base}} (base). CVaR99 moves from {{cvar99_indep}}
  (independent) to {{cvar99_common}} (common shock); VaR95 stays at {{var95_base}}.
* On 2-D grids with shared perils, exact computation stops at {{grid_exact_rows}} rows. IQAE keeps
  an edge on a moderate tail ({{grid_t3_last}} at {{grid_last}}) and loses to importance sampling
  on rarer tails ({{grid_t5_last}}).
* Quantum-enhanced MCMC beats local MCMC under high contagion, but Wolff cluster moves beat every
  quantum sampler (the best quantum sampler needs {{qmcmc_k4_quantum_needs}} the evaluations).
* Certified IQAE intervals covered the exact value in at least {{cov_min}} of {{cov_runs}} runs per cell
  (nominal {{cov_nominal}}). Clopper-Pearson misses can be confidently wrong (worst {{cov_worst_miss}}), so the
  digital twin uses Chernoff-Hoeffding bounds (twin worst miss {{ti_ch_worst}}, at {{ti_cost_median}} the queries).
* **Digital twin.** Interval transport saves {{twin_save_lo}}-{{twin_save_hi}} of the queries on a fine
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

The digital twin (certified answers, a dependence sweep with interval transport, a structure
comparison, and a conditional tail-scenario library with drivers, all with a query ledger):

```bash
python -m adbq twin configs/twin-adb-base.yaml  # writes results/twin_runs/<name>/report.json, summary.md
```

Each run writes `results/runs/<name>/report.json`, `summary.md` and figures. Shipped configs:

{{table_runs}}

## What is in here

| module | contents |
|---|---|
| `model.py` | the brief's model; annual CTMC and independent-source routes; Table 5 |
| `baselines.py`, `risk_baselines.py` | MC, Sobol, stratified, importance and loss-tilted samplers with exact variances |
| `circuits.py`, `oracle.py` | gate-level A, exact sparse simulator |
| `qae.py`, `risk_qae.py` | IQAE (Clopper-Pearson or Chernoff-Hoeffding), MLAE; VaR bisection and CVaR with IQAE |
| `risk.py` | exact loss laws (two-country, chain), VaR and CVaR |
| `portfolio.py` | n-country chain and its exact transfer-matrix contraction |
| `grid.py`, `qmcmc.py` | calibrated Ising trigger grid; quantum-enhanced MCMC |
| `twin.py` | the certified portfolio digital twin (`adbq twin`) |
| `_vendor/supplychain_qmcmc` | four modules from the team's earlier project `logistics_under_disruption`, unchanged |
| `gridsources.py`, `grid_estimators.py` | 2-D shared-peril grid, exact sweep, MC / RQMC / tilt / cross-entropy, IQAE |
| `qiskit_circuits.py` | Qiskit A and Q, compiled counts, verification by composition |
| `resources.py` | logical resource model |
| `twin.py` | digital twin: versioned source laws, exact ratio bounds, interval transport, scenario library, paired comparisons, allocation, ledger |
| `pipeline.py`, `cli.py` | `adbq run config.yaml` |

Generated result summaries: [`risk_summary.md`](results/risk_summary.md),
[`grid_summary.md`](results/grid_summary.md), [`qmcmc_exact.md`](results/qmcmc_exact.md),
[`qmcmc_chains.md`](results/qmcmc_chains.md), [`grid_qae.md`](results/grid_qae.md),
[`real_circuits.md`](results/real_circuits.md), [`coverage.md`](results/coverage.md),
[`twin.md`](results/twin.md), [`ft_estimate.md`](results/ft_estimate.md).

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
python experiments/twin.py                            # digital-twin experiments
python experiments/twin_intervals.py                  # interval method for the twin (CP vs CH)
python experiments/ft_estimate.py                     # surface-code estimate
python experiments/quantinuum_job.py                  # dry run only: native counts and HQC
python -m adbq run configs/*.yaml                      # twin configs are skipped; use adbq twin
python -m adbq twin configs/twin-adb-base.yaml
python experiments/make_figures.py
python docs/build_docs.py                             # README.md and docs/proposal.md from results
python docs/build_pdf.py                              # docs/proposal.pdf (Windows with Microsoft Word)
python tools/package_submission.py                    # submission ZIP = the files git would publish
```

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

## Assumptions to confirm against ADB's benchmark repository

* Stratified sampling uses proportional allocation across the three strata.
* The brief's importance-sampling proposal tilts by annual principal loss, mixed half-and-half with p.
* The n-country chain and the 2-D grid families are ours; the brief defines only n = 2.

## AI use

Code and documentation were drafted with Claude (Anthropic) and checked against exact references,
independent implementations and the test suite. `README.md` and `docs/proposal.md` are generated by
`docs/build_docs.py`; every number in them comes from `results/`.
