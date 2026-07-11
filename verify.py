#!/usr/bin/env python3
"""verify.py — the new tree's entire audit setup, one flat script.

Subcommands (each a small function; `all` is the CI / merge gate):

    acyclic            FAIL if any import points upward across layers
    tests --layer N    run only the tests at layer <= N (staged tests)
    debt               FAIL if suppressions - OPEN.md ng-ledger entries != 0
    provenance         FAIL if a landed module lacks its 4-line header
    signatures         FAIL if any function takes more than 5 arguments (CLAUDE.md 3b)
    version            FAIL if __version__ != top CHANGELOG.md entry
    all                everything above

No framework, no plugins — if a sixth check is ever needed it is one more
function then, not an abstraction now (CLAUDE.md 6b, redesign/09).
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src" / "pricebook_ng"
TESTS = ROOT / "tests_ng"
CHANGELOG = ROOT / "CHANGELOG.md"
OPEN = ROOT / "OPEN.md"

# The spine: package name -> layer rank. Dependencies may point DOWN (lower or
# equal rank) only, never up. This IS the law from CLAUDE.md 1.
LAYER = {
    "foundation": 0,
    "market": 1,
    "instruments": 2,
    "models": 3,
    "engine": 4,
    "risk": 5,
    "shell": 6,
}

# Suppression markers that must each be matched by an OPEN.md ledger entry.
_SUPPRESSIONS = re.compile(r"# *type: *ignore|# *noqa|# *pragma: *no cover|@pytest\.mark\.skip")


def _src_modules() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if p.name != "__init__.py")


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def acyclic() -> list[str]:
    """FAIL if any pricebook_ng import targets a higher layer than its importer.

    ponytail: layer-rank check, not full Tarjan SCC. It catches every upward
    edge (the load-bearing invariant); add SCC if an intra-layer cycle ever
    bites. Equal-rank imports (within a layer) are allowed.
    """
    errs: list[str] = []
    for mod in _src_modules():
        rel = mod.relative_to(SRC)
        importer_layer = LAYER.get(rel.parts[0])
        if importer_layer is None:
            continue
        tree = ast.parse(_read(mod), filename=str(mod))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.ImportFrom) and node.module:
                targets.append(node.module)
            elif isinstance(node, ast.Import):
                targets += [n.name for n in node.names]
            for t in targets:
                parts = t.split(".")
                if len(parts) >= 2 and parts[0] == "pricebook_ng":
                    dep_layer = LAYER.get(parts[1])
                    if dep_layer is not None and dep_layer > importer_layer:
                        errs.append(
                            f"{rel}: {rel.parts[0]}(L{importer_layer}) imports "
                            f"{parts[1]}(L{dep_layer}) — upward import"
                        )
    return errs


def run_tests(max_layer: int) -> int:
    """Run tests at layer <= max_layer. Directory-per-layer: tests_ng/L0..LN."""
    dirs = [str(TESTS / f"L{n}") for n in range(max_layer + 1) if (TESTS / f"L{n}").is_dir()]
    if not dirs:
        print(f"no test dirs at layer <= {max_layer}")
        return 0
    return subprocess.call([sys.executable, "-m", "pytest", *dirs])


def debt() -> list[str]:
    """suppressions - OPEN.md ng-ledger entries must be 0 (CLAUDE.md 5)."""
    suppressions = 0
    for p in list(_src_modules()) + sorted(TESTS.rglob("*.py")):
        for i, line in enumerate(_read(p).splitlines(), 1):
            if _SUPPRESSIONS.search(line):
                suppressions += 1
    ledger = 0
    if OPEN.exists():
        ledger = len(re.findall(r"^- \[NG-", _read(OPEN), flags=re.MULTILINE))
    if suppressions != ledger:
        return [f"suppressions ({suppressions}) != OPEN.md NG-ledger entries ({ledger})"]
    return []


_PROV_KEYS = ("quarry:", "source:", "oracle:", "slice:")


def provenance() -> list[str]:
    """Every landed module carries the 4-line provenance header (redesign/09)."""
    errs: list[str] = []
    for mod in _src_modules():
        text = _read(mod)
        missing = [k for k in _PROV_KEYS if k not in text]
        if missing:
            errs.append(f"{mod.relative_to(ROOT)}: missing provenance {missing}")
    return errs


_MAX_ARGS = 5  # CLAUDE.md 3b: over the ceiling means un-bundled vocabulary, not a bug


def signatures() -> list[str]:
    """FAIL if any function/method exceeds the 5-argument ceiling (CLAUDE.md 3b).

    The ruff `PLR0913`/`max-args=5` rule, enforced in the merge gate by our one
    tool. `self`/`cls` are not counted; `*args`/`**kwargs` are not counted.
    """
    errs: list[str] = []
    for mod in _src_modules():
        tree = ast.parse(_read(mod), filename=str(mod))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = [a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
            if names and names[0] in ("self", "cls"):
                names = names[1:]
            if len(names) > _MAX_ARGS:
                errs.append(
                    f"{mod.relative_to(ROOT)}:{node.lineno}: {node.name}() takes "
                    f"{len(names)} args (max {_MAX_ARGS})"
                )
    return errs


def version() -> list[str]:
    """__version__ must match the top CHANGELOG.md entry."""
    m = re.search(r'__version__\s*=\s*"([^"]+)"', _read(SRC / "__init__.py"))
    ver = m.group(1) if m else None
    m = re.search(r"^## \[([0-9]+\.[0-9]+\.[0-9]+)\]", _read(CHANGELOG), flags=re.MULTILINE)
    top = m.group(1) if m else None
    if ver != top:
        return [f"__version__ ({ver}) != top CHANGELOG entry ({top})"]
    return []


def _report(name: str, errs: list[str]) -> bool:
    if errs:
        print(f"FAIL {name}:")
        for e in errs:
            print(f"  - {e}")
        return False
    print(f"ok   {name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("acyclic")
    t = sub.add_parser("tests")
    t.add_argument("--layer", type=int, required=True)
    sub.add_parser("debt")
    sub.add_parser("provenance")
    sub.add_parser("signatures")
    sub.add_parser("version")
    sub.add_parser("all")
    args = ap.parse_args()

    if args.cmd == "tests":
        return run_tests(args.layer)
    checks = {
        "acyclic": acyclic, "debt": debt, "provenance": provenance,
        "signatures": signatures, "version": version,
    }
    if args.cmd == "all":
        return 0 if all(_report(n, f()) for n, f in checks.items()) else 1
    return 0 if _report(args.cmd, checks[args.cmd]()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
