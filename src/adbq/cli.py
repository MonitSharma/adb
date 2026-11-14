"""Command line: ``adbq run config.yaml [...]``, ``adbq twin twin.yaml`` (or ``python -m adbq ...``)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="adbq", description="Joint-loss modelling for disaster risk financing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one or more portfolio configs")
    r.add_argument("configs", nargs="+", help="YAML config files")
    r.add_argument("--out", default="results/runs", help="output root (default results/runs)")
    v = sub.add_parser("validate", help="check configs without running")
    v.add_argument("configs", nargs="+")
    tw = sub.add_parser("twin", help="run the portfolio digital twin on a twin config")
    tw.add_argument("configs", nargs="+", help="twin YAML config files")
    tw.add_argument("--out", default="results/twin_runs", help="output root (default results/twin_runs)")
    a = ap.parse_args(argv)
    if a.cmd == "twin":
        from .twin import load_twin_config, run_twin
        for path in a.configs:
            t0 = time.time()
            out = run_twin(load_twin_config(path), a.out)
            print(f"{path} -> {out} ({time.time() - t0:.0f}s): report.json, summary.md")
        return 0
    import yaml
    from .pipeline import load_config, run
    for path in a.configs:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if "portfolio" in raw:                      # a twin config, not a portfolio config
            from .twin import load_twin_config
            load_twin_config(path)
            if a.cmd == "validate":
                print(f"ok  {path}: twin config (run with `adbq twin {path}`)")
            else:
                print(f"skip {path}: twin config, run it with `adbq twin {path}`")
            continue
        cfg = load_config(path)
        if a.cmd == "validate":
            print(f"ok  {path}: model {cfg['model']}, targets {cfg['targets']}")
            continue
        t0 = time.time()
        out = run(cfg, a.out)
        print(f"{path} -> {out} ({time.time() - t0:.0f}s): report.json, summary.md, figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
