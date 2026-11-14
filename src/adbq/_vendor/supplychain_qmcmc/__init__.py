"""Vendored from the team's earlier project ``logistics_under_disruption``
(src/supplychain_qmcmc, same authors), unchanged, so this repository runs on its own.

Only the four modules adbq uses are included: ``kernels`` (exact Metropolis-Hastings
matrices, asymptotic variances, spectral gaps), ``quantum`` (Layden et al. proposal and
Trotter layers), ``mrf`` (conventions both import) and ``quantinuum`` (offline H-series
compilation, gate counts, HQC cost model, published-spec noise models).
tests/test_vendor.py checks that the vendored copies are importable and, when the sibling
project is present, byte-identical to it.
"""
