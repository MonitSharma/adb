"""Record the environment the results were produced in: Python and package versions, OS,
CPU and memory, and the simulators used. Writes results/environment.json and
requirements-lock.txt (exact versions of every package the prototype imports).

Run: python experiments/environment.py
"""
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = ["numpy", "scipy", "matplotlib", "PyYAML", "pytest",            # core
            "qiskit", "qiskit-aer", "pytket", "pytket-qiskit", "pytket-quantinuum"]   # quantum extras


def memory_gb():
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 2**30, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except Exception:
        return None


def main():
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    env = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "processor": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "memory_gb": memory_gb(),
        "gpu_used": False,
        "packages": versions,
        "simulators": [
            "adbq.circuits: exact sparse state-vector simulator (no SDK), used for every verified amplitude",
            f"Qiskit Aer {versions.get('qiskit-aer')}: state-vector and noisy shot simulation",
            f"pytket-quantinuum {versions.get('pytket-quantinuum')}: offline compilation to the H-series gate set (no account)",
        ],
        "quantum_hardware_used": False,
    }
    (ROOT / "results" / "environment.json").write_text(json.dumps(env, indent=1))
    lock = ["# Exact versions used to produce results/ (written by experiments/environment.py).",
            f"# Python {env['python']}, {env['os']}."]
    lock += [f"{p}=={v}" for p, v in versions.items() if v]
    (ROOT / "requirements-lock.txt").write_text("\n".join(lock) + "\n")
    print(json.dumps(env, indent=1))


if __name__ == "__main__":
    main()
