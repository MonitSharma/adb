"""Exact VaR and CVaR of the three-year portfolio loss for the five brief scenarios.

Writes results/cvar.json. Run: python experiments/cvar.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adbq.model import SCENARIOS, loss_distribution, var_cvar  # noqa: E402

ALPHAS = (0.95, 0.99, 0.995)


def main():
    out = {}
    for name, s in SCENARIOS.items():
        lv, pr = loss_distribution(s)
        out[name] = {f"{a}": dict(zip(("var", "cvar"), var_cvar(lv, pr, a))) for a in ALPHAS}
        print(f"{name:18s}", "  ".join(f"VaR{a * 100:g}={out[name][f'{a}']['var'] / 1e6:6.1f}m "
                                     f"CVaR={out[name][f'{a}']['cvar'] / 1e6:6.1f}m" for a in ALPHAS))
    (ROOT / "results" / "cvar.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
