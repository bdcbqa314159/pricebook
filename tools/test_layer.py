#!/usr/bin/env python3
"""Classify pricebook tests by max import layer.

For each test file under ``python/tests/``, computes
    layer(test) = max(layer(pkg) for pkg in pricebook.* imports)

using the live dependency graph of ``python/pricebook/``.

Tests with no pricebook imports → layer = -1 (always applicable).

Imports counted:
  * Top-level ``import pricebook.X`` and ``from pricebook.X import …``
  * Inside ``class`` bodies (still module-load time)
Imports NOT counted (look-ahead avoided):
  * Inside ``if TYPE_CHECKING:`` blocks (typing-only)
  * Inside function / method bodies (runtime-lazy)

CLI::

    tools/test_layer.py                    # show layer per test file
    tools/test_layer.py --max-layer 0      # print test paths at layer ≤ 0
    tools/test_layer.py --max-layer 1      # print test paths at layer ≤ 1
    tools/test_layer.py --json             # JSON dump

Pipe to pytest::

    .venv/bin/python -m pytest $(tools/test_layer.py --max-layer 0)
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "python" / "pricebook"
TEST_ROOT = REPO_ROOT / "python" / "tests"


# ──────────────────────────────────────────────────────────────────────────
# AST visitor — collect top-level pricebook imports, skipping TYPE_CHECKING
# and function-local imports.
# ──────────────────────────────────────────────────────────────────────────

class _ImportCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.depth = 0  # nested-function depth
        self.targets: list[str] = []

    def _is_type_checking(self, test: ast.AST) -> bool:
        if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
            return True
        if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
            return True
        return False

    def visit_If(self, node: ast.If) -> None:
        if self._is_type_checking(node.test):
            for stmt in node.orelse:
                self.visit(stmt)
            return
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Class-body imports are still module-load-time — visit children.
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        if self.depth:
            return
        for alias in node.names:
            if alias.name.startswith("pricebook"):
                self.targets.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self.depth:
            return
        mod = node.module or ""
        if mod.startswith("pricebook"):
            self.targets.append(mod)


def _collect_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    coll = _ImportCollector()
    coll.visit(tree)
    return coll.targets


def _pkg_of(mod: str) -> str | None:
    """Return the top-level package of ``pricebook.<pkg>.…`` or None."""
    parts = mod.split(".")
    if len(parts) < 2 or parts[0] != "pricebook":
        return None
    return parts[1]


# ──────────────────────────────────────────────────────────────────────────
# Compute layer(pkg) by walking the source-side dep graph.
# ──────────────────────────────────────────────────────────────────────────

def compute_layers() -> dict[str, int]:
    deps: dict[str, set[str]] = defaultdict(set)
    pkgs: set[str] = set()

    for sub in SRC_ROOT.iterdir():
        if sub.is_dir() and (sub / "__init__.py").exists():
            pkgs.add(sub.name)

    for py in SRC_ROOT.rglob("*.py"):
        if any(p.startswith("_legacy") for p in py.parts):
            continue
        rel = py.relative_to(SRC_ROOT)
        if len(rel.parts) <= 1:
            continue  # top-level pricebook/__init__.py
        this_pkg = rel.parts[0]
        pkgs.add(this_pkg)
        for imp in _collect_imports(py):
            tgt = _pkg_of(imp)
            if tgt and tgt != this_pkg and tgt in pkgs:
                deps[this_pkg].add(tgt)

    layer: dict[str, int] = {}

    def _layer(p: str) -> int:
        if p in layer:
            return layer[p]
        if not deps.get(p):
            layer[p] = 0
            return 0
        layer[p] = 1 + max(_layer(d) for d in deps[p] if d in pkgs)
        return layer[p]

    for p in pkgs:
        _layer(p)
    return layer


# ──────────────────────────────────────────────────────────────────────────
# Classify each test file.
# ──────────────────────────────────────────────────────────────────────────

def classify_tests(layers: dict[str, int]) -> list[tuple[Path, int, set[str]]]:
    """Returns list of (test_path, max_layer, set_of_pkgs).

    Tests with no pricebook imports get max_layer = -1.
    """
    out: list[tuple[Path, int, set[str]]] = []
    for py in sorted(TEST_ROOT.rglob("test_*.py")):
        imports = _collect_imports(py)
        pkgs = {p for p in (_pkg_of(m) for m in imports) if p is not None}
        if not pkgs:
            max_layer = -1
        else:
            known = {p for p in pkgs if p in layers}
            if not known:
                max_layer = -1
            else:
                max_layer = max(layers[p] for p in known)
        out.append((py, max_layer, pkgs))
    return out


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--max-layer",
        type=int,
        default=None,
        help="Print only test paths at layer ≤ this value (one per line).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON {test_path: {layer: int, pkgs: [..]}}.",
    )
    p.add_argument(
        "--show-layers",
        action="store_true",
        help="Print the package-layer mapping and exit.",
    )
    args = p.parse_args()

    layers = compute_layers()

    if args.show_layers:
        for pkg, L in sorted(layers.items(), key=lambda kv: (kv[1], kv[0])):
            print(f"  L{L}  {pkg}")
        return 0

    classification = classify_tests(layers)

    if args.json:
        out = {
            str(path.relative_to(REPO_ROOT)): {
                "layer": layer,
                "pkgs": sorted(pkgs),
            }
            for path, layer, pkgs in classification
        }
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.max_layer is not None:
        for path, L, _ in classification:
            if L <= args.max_layer:
                print(path.relative_to(REPO_ROOT))
        return 0

    # Default: human-readable, grouped by layer
    by_layer: dict[int, list[tuple[Path, set[str]]]] = defaultdict(list)
    for path, L, pkgs in classification:
        by_layer[L].append((path, pkgs))

    for L in sorted(by_layer):
        files = by_layer[L]
        tag = "always" if L < 0 else f"L{L}"
        print(f"\n[{tag}] — {len(files)} test files")
        for path, pkgs in files:
            print(f"  {path.relative_to(REPO_ROOT)}    pkgs={sorted(pkgs)}")

    print()
    print(f"Summary: {sum(len(v) for v in by_layer.values())} test files")
    for L in sorted(by_layer):
        tag = "always" if L < 0 else f"L{L}"
        print(f"  {tag}: {len(by_layer[L])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
