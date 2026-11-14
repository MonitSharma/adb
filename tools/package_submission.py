"""Build the prototype ZIP for the application form from exactly the files git would publish
(every file not ignored by .gitignore), so the ZIP matches the GitHub repository.

Writes submission/adbq-prototype.zip with a top-level folder adbq/ and prints a manifest
summary. Run from the repository root after `git init` (no commit is needed):

    python tools/package_submission.py
"""
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "adbq-prototype.zip"


def published_files() -> list[str]:
    r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                       capture_output=True, text=True, check=True)
    return sorted(f for f in r.stdout.splitlines() if f and (ROOT / f).is_file())


def main():
    files = published_files()
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ROOT / f, f"adbq/{f}")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(files)} files, {OUT.stat().st_size / 1e6:.1f} MB")
    for must in ("README.md", "requirements.txt", "requirements-lock.txt", "docs/proposal.pdf",
                 "results/environment.json", "src/adbq/_vendor/supplychain_qmcmc/kernels.py"):
        assert must in files, f"missing from the package: {must}"


if __name__ == "__main__":
    main()
