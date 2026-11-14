"""Print the exact three-year targets for all five scenarios (Table 5 of the brief)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from adbq.model import SCENARIOS, SHORT, TABLE5, exact  # noqa: E402

print(f"{'scenario':18s}" + "".join(f"{s:>20s}" for s in SHORT))
for name in SCENARIOS:
    print(f"{name:18s}" + "".join(f"{v:20.6g}" for v in exact(name).values()))
    print(f"{'  brief Table 5':18s}" + "".join(f"{v:20.6g}" for v in TABLE5[name]))
