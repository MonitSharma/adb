"""The vendored modules from logistics_under_disruption import, and match the sibling project if present."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "src" / "adbq" / "_vendor" / "supplychain_qmcmc"
SIBLING = ROOT.parent / "logistics_under_disruption" / "src" / "supplychain_qmcmc"
MODULES = ("kernels", "quantum", "mrf", "quantinuum")


def test_vendored_modules_import():
    from adbq._vendor.supplychain_qmcmc import kernels, mrf, quantinuum, quantum   # noqa: F401
    from adbq.qmcmc import logistics
    k, q = logistics()
    assert hasattr(k, "mh_transition") and hasattr(q, "trotter_unitary")


@pytest.mark.parametrize("name", MODULES)
def test_vendored_copy_matches_sibling(name):
    src = SIBLING / f"{name}.py"
    if not src.exists():
        pytest.skip("sibling project not present (expected on a clean clone)")
    assert (VENDOR / f"{name}.py").read_bytes() == src.read_bytes()
