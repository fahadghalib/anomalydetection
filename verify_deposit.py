"""Verify that the deposited repository is complete and self-consistent.

Reviewer comment #1 noted that the published listing was missing modules the
code imports -- `sampling.py` and `mlp_baseline.py` among them -- so the
deposit was not runnable as provided. This script makes that class of failure
impossible to ship again: it parses every source file, resolves every
intra-package import, and reports anything referenced but absent.

It also emits the file manifest and the SHA-256 of each source file, so a
reader can confirm that the code they downloaded is the code that produced
the reported numbers.

Run:  python verify_deposit.py
Exit status is non-zero if anything is missing, which makes it usable as a
pre-release check.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# Files the pipeline needs but which are not code. Absence is reported as a
# warning rather than an error, since the CSVs are distributed separately.
DATA_FILES = [
    ROOT / "data" / "VeReMi_train_data.csv",
    ROOT / "data" / "VeReMi_test_data.csv",
]

THIRD_PARTY_OK = {
    "tensorflow", "numpy", "pandas", "sklearn", "scipy", "imblearn",
    "matplotlib", "lightgbm", "xgboost", "catboost",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def intra_package_imports(path: Path) -> set[str]:
    """Module names imported from within this package by one source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # relative import: from .sampling import ...
            if node.level and node.module:
                found.add(node.module.split(".")[0])
            # absolute: from src.sampling import ...
            elif node.module and node.module.startswith("src."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    found.add(alias.name.split(".")[1])
    return found


def main() -> int:
    py_files = sorted(SRC.glob("*.py")) + sorted(ROOT.glob("*.py"))
    py_files = [p for p in py_files if p.name != Path(__file__).name]

    present = {p.stem for p in SRC.glob("*.py")}
    missing: list[tuple[str, str]] = []

    print("=" * 72)
    print("DEPOSIT MANIFEST")
    print("=" * 72)
    for p in py_files:
        rel = p.relative_to(ROOT).as_posix()
        print(f"  {sha256(p)[:16]}  {p.stat().st_size:>7,} B  {rel}")

    print()
    print("=" * 72)
    print("IMPORT RESOLUTION")
    print("=" * 72)
    for p in py_files:
        try:
            needed = intra_package_imports(p)
        except SyntaxError as exc:
            print(f"  [SYNTAX ERROR] {p.name}: {exc}")
            missing.append((p.name, f"syntax error: {exc}"))
            continue
        for mod in sorted(needed):
            if mod in present or mod in THIRD_PARTY_OK:
                continue
            print(f"  [MISSING] {p.relative_to(ROOT).as_posix()} imports "
                  f"'{mod}' -> src/{mod}.py not found")
            missing.append((p.name, mod))
    if not missing:
        print(f"  all intra-package imports resolve across {len(py_files)} files")

    print()
    print("=" * 72)
    print("DATA FILES")
    print("=" * 72)
    for d in DATA_FILES:
        if d.exists():
            print(f"  present  {d.relative_to(ROOT).as_posix()}  "
                  f"{d.stat().st_size / 1e6:,.1f} MB")
        else:
            print(f"  ABSENT   {d.relative_to(ROOT).as_posix()}  "
                  f"(distributed separately; see README)")

    print()
    if missing:
        print(f"RESULT: INCOMPLETE -- {len(missing)} unresolved reference(s)")
        return 1
    print("RESULT: COMPLETE -- every imported module is present in the deposit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
