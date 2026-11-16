# Application form: draft answers

Rendered by `docs/build_docs.py` into `submission/application_form.md` (git-ignored: it carries names and
contact details). Paste each field into the ADB Challenges form; word limits are checked by
`tests/test_docs.py`. Fields marked FILL IN need the team's own details.

## Team Name (max 10 words)

SMU

## Team Members

FILL IN, one block per member, in the form's format:
Complete Name (First Name, Last Name) / Country of Residence / Email Address / Organization

* LAU, Hoong Chuin / Singapore / FILL IN / Singapore Management University
* Sharma, Monit / Singapore / FILL IN / Singapore Management University

## Team Experience and Qualifications (max 300 words)

The team combines quantum algorithms for risk and decisions under uncertainty with AI and optimisation.
Prof. LAU Hoong Chuin (lead) is Professor of Computer Science at Singapore Management University and a
Changjiang Scholar, working on AI and quantum methods for optimisation under uncertainty; he founded
Horaion.AI, a venture built on SMU-licensed AI technology. Monit Sharma (SMU School of Computing and Information
Systems) built this prototype end to end: the gate-level quantum circuits and their verification, the classical
benchmarks, the digital twin and the pipeline.

Relevant prior work: a study of risk-averse newsvendor decisions with quantum amplitude estimation (Sharma and
Lau, manuscript for the European Journal of Operational Research), which estimates CVaR with QAE, introduces
certify-or-stop thresholds and in-circuit paired comparisons, and runs the complete adaptive procedure on IBM
quantum hardware; a study of adaptive quantum-assisted scenario exploration for pandemic digital twins (Lau et
al., manuscript for IEEE Transactions on Quantum Engineering), which supplies the certified calibration,
interval transport, exact cache thinning and budget allocation used here; earlier quantum amplitude estimation
pipelines for simulation-based inventory decisions; and quantum-enhanced Markov chain Monte Carlo for
correlated supply-chain disruptions, including runs on Quantinuum hardware.

The prototype for this challenge reproduces every reference value in the brief, benchmarks quantum amplitude
estimation against strong classical samplers (Monte Carlo, quasi-Monte Carlo, stratified, tuned and
cross-entropy importance sampling, cluster MCMC and parallel tempering), and reports its negative results.

## Solution Title (max 15 words)

Certified Quantum Estimation of Joint Catastrophe Tails: a Digital Twin for Disaster Risk Finance

## Executive Summary (max 300 words)

We propose a certified digital twin of ADB's disaster-risk portfolio: a versioned model of the bonds that
answers the questions ADB asks (how likely a joint tail event is, what drives it, how it moves when dependence
assumptions change, and which contract structure is better), each with a confidence interval and a recorded
query cost. Its estimator is iterative quantum amplitude estimation, implemented gate by gate, verified against
every reference value in the brief, compiled with Qiskit and pytket, and simulated with realistic noise.

On the brief's two-country model, quantum amplitude estimation needs {{acc_prob_lo}}-{{acc_prob_hi}} fewer
queries than the best tuned classical sampler on joint-payout and exceedance probabilities. On the rarest
target it is close to a tuned, rain-conditioned sampler ({{acc_rare_lo}}-{{acc_rare_hi}}). A tuned tilt beats
it on stop-loss, and Scrambled Sobol wins on expected loss. Exact CVaR99 rises from
{{cvar99_indep}} without dependence to {{cvar99_common}} under a common shock, while VaR95 stays at
{{var95_base}}. The twin re-uses certified answers across nearby dependence assumptions and saves
{{twin_save_lo}}-{{twin_save_hi}} of the queries on a fine sweep.

We test scaling honestly. Chains of countries are exactly solvable in linear time, so no quantum claim is made
there. On a 2-D grid of shared perils exact methods stop at {{grid_exact_rows}} rows; beyond it quantum
estimation keeps a lead on a moderately rare tail ({{grid_t3_last}} at {{grid_last}}) and loses to importance
sampling on rarer tails. Quantum-enhanced MCMC loses to classical cluster moves.

The full circuit needs fault tolerance; a 6-qubit sub-circuit is compiled and costed for current hardware
({{hqc_recommended}} HQC). The deliverable is classical-first and runs today from one portfolio specification;
the quantum route is benchmarked, costed and ready to scale where exact methods fail.

## Which Best Describes Your Proposed Solution?

Quantum Computing; Hybrid Quantum-Classical Approach; Statistical Modelling.

## Expected Advantage Over Classical Approaches (max 200 words)

Accuracy per query on joint-tail quantities. On the brief's model, quantum amplitude estimation needs
{{acc_prob_lo}}-{{acc_prob_hi}} fewer queries than the best tuned classical sampler for joint-payout and
exceedance probabilities at the same accuracy. On the probability that both bonds are exhausted, it is close to
a tuned sampler that also conditions on the independent rain triggers ({{acc_rare_lo}}-{{acc_rare_hi}}). For CVaR, the quantum route wins at {{cvar_tight}} accuracy and loses to a
tuned importance sampler at {{cvar_loose}}. Every answer carries an interval that holds with {{cov_nominal}}
probability, and the twin re-uses certified answers across nearby dependence assumptions, saving
{{twin_save_lo}}-{{twin_save_hi}} of the queries on a fine sweep. The state preparation grows linearly with the
number of independent loss sources, for any dependence written as a forward simulator.

We state the limits. A query is not a unit of time: one quantum query is a fault-tolerant circuit of
{{ft_t_per_query}} T gates, so there is no wall-clock advantage today. Good importance samplers beat quantum
estimation on rare tails of large light-tailed sums and on stop-loss, Sobol wins on expected loss, and classical paired Monte
Carlo wins on structure comparison. Phase 2 targets non-additive payouts and overlapping footprints, where
classical samplers are weakest.

## Provide GitHub Repository URL for Lightweight Prototype

FILL IN after publishing the repository (see the README's Reproduce section).

## Additional Notes (max 150 words)

Every number in the proposal and README is generated from the repository's results by a script
(`docs/build_docs.py`), and a test fails if a document is stale or contains a hand-typed result. Exact
references reproduce Table 5 of the brief for all five scenarios. No quantum hardware was used; IQAE is
emulated shot by shot at the exact amplitude of the verified circuit, and a small Quantinuum job is compiled
and costed but not submitted. Two baseline settings (stratified allocation and the form of the brief's
importance-sampling tilt) are our reading of the brief and will be aligned with ADB's benchmark material when
it is shared. The repository is self-contained; four modules from the team's earlier project are vendored
unchanged.
