#!/usr/bin/env python3
"""Module-level dependency tree for L0 sub-packages.

For every module under L0 sub-packages (calibration, core, db, market_data,
numerical, pe, statistics, ts, viz), computes:

  * direct intra-L0 imports (this module's deps)
  * who imports it (this module's dependents)
  * topological order within its own sub-package

Same AST rules as ``test_layer.py``: top-level + class-body imports,
skip TYPE_CHECKING, skip function-local. Empirical, not from-docstrings.

CLI::

    tools/l0_deps.py                       # full tree, all 9 sub-packages
    tools/l0_deps.py --pkg core            # one sub-package
    tools/l0_deps.py --orphans             # list leaf modules (no pricebook deps)
    tools/l0_deps.py --reading-order       # flat reading order: leaves to roots
"""
from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "python" / "pricebook"

L0_PACKAGES = {
    "calibration", "core", "db", "market_data",
    "numerical", "pe", "statistics", "ts", "viz",
}


# ──────────────────────────────────────────────────────────────────────────
# Same AST visitor as test_layer.py — load-time imports only
# ──────────────────────────────────────────────────────────────────────────

class _ImportCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.depth = 0
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


# ──────────────────────────────────────────────────────────────────────────
# Build the module graph
# ──────────────────────────────────────────────────────────────────────────

def _modname(p: Path) -> str:
    """Convert a file path under SRC_ROOT to a dotted pricebook module name."""
    rel = p.relative_to(SRC_ROOT)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]  # strip .py
    return "pricebook." + ".".join(parts)


def _is_l0(modname: str) -> bool:
    parts = modname.split(".")
    return len(parts) >= 2 and parts[0] == "pricebook" and parts[1] in L0_PACKAGES


def _resolve(imp: str, all_modules: set[str]) -> str | None:
    """An import target like 'pricebook.core.day_count' resolves to a module
    name we know. An import target like 'pricebook.core' resolves to the
    package __init__ (pricebook.core)."""
    if imp in all_modules:
        return imp
    # Try as a sub-module: 'pricebook.core' itself or 'pricebook.X.Y'
    # where some prefix matches a module
    parts = imp.split(".")
    while len(parts) > 1:
        candidate = ".".join(parts)
        if candidate in all_modules:
            return candidate
        parts = parts[:-1]
    return None


def build_graph():
    """Returns (modules, deps, dependents):
      * modules: set of all module names in L0
      * deps[m]: set of L0 modules m imports
      * dependents[m]: set of L0 modules that import m
    """
    # Discover all L0 modules first
    modules: set[str] = set()
    paths: dict[str, Path] = {}
    for py in SRC_ROOT.rglob("*.py"):
        if any(p.startswith("_legacy") for p in py.parts):
            continue
        m = _modname(py)
        if _is_l0(m):
            modules.add(m)
            paths[m] = py

    deps: dict[str, set[str]] = defaultdict(set)
    dependents: dict[str, set[str]] = defaultdict(set)

    for m in sorted(modules):
        for imp in _collect_imports(paths[m]):
            tgt = _resolve(imp, modules)
            if tgt is None:
                continue
            if tgt == m:
                continue
            if not _is_l0(tgt):
                continue
            deps[m].add(tgt)
            dependents[tgt].add(m)

    return modules, deps, dependents, paths


def topo_order(modules: set[str], deps: dict[str, set[str]]) -> list[str]:
    """Topological order: a module appears AFTER all its deps."""
    visited: set[str] = set()
    out: list[str] = []
    def _visit(m: str, stack: list[str]):
        if m in visited:
            return
        if m in stack:
            # Cycle — just stop here; the caller (and this script's output)
            # will show the cycle.
            return
        stack.append(m)
        for d in sorted(deps.get(m, ())):
            _visit(d, stack)
        stack.pop()
        visited.add(m)
        out.append(m)
    for m in sorted(modules):
        _visit(m, [])
    return out


# ──────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────

def short(m: str) -> str:
    """Strip 'pricebook.' prefix for display."""
    return m[len("pricebook."):]


def print_tree(pkg_filter: str | None = None):
    modules, deps, dependents, _ = build_graph()
    order = topo_order(modules, deps)

    by_pkg: dict[str, list[str]] = defaultdict(list)
    for m in order:
        pkg = m.split(".")[1]
        if pkg_filter and pkg != pkg_filter:
            continue
        by_pkg[pkg].append(m)

    for pkg in sorted(by_pkg):
        mods = by_pkg[pkg]
        print(f"\n{'='*72}")
        print(f"  pricebook.{pkg}    ({len(mods)} modules, dependency order)")
        print(f"{'='*72}\n")
        for m in mods:
            d = sorted(deps.get(m, ()))
            r = sorted(dependents.get(m, ()))
            tag = ""
            if not d:
                tag = " [LEAF]"
            print(f"  {short(m)}{tag}")
            if d:
                for dd in d:
                    print(f"      → {short(dd)}")
            if r:
                # Just count dependents to avoid wall of text
                in_pkg = sum(1 for rr in r if rr.split(".")[1] == pkg)
                cross_pkg = len(r) - in_pkg
                print(f"      ← {len(r)} dependents ({in_pkg} intra, {cross_pkg} cross-L0)")


def print_reading_order(pkg_filter: str | None = None):
    modules, deps, dependents, _ = build_graph()
    order = topo_order(modules, deps)
    if pkg_filter:
        order = [m for m in order if m.split(".")[1] == pkg_filter]
    print(f"# L0 reading order — leaves to roots ({len(order)} modules)\n")
    for i, m in enumerate(order, 1):
        d = sorted(deps.get(m, ()))
        leaf = " [LEAF]" if not d else ""
        print(f"{i:3}.  {short(m)}{leaf}")


def print_orphans(pkg_filter: str | None = None):
    modules, deps, dependents, _ = build_graph()
    by_pkg: dict[str, list[str]] = defaultdict(list)
    for m in sorted(modules):
        if pkg_filter and m.split(".")[1] != pkg_filter:
            continue
        if not deps.get(m):
            by_pkg[m.split(".")[1]].append(m)
    for pkg in sorted(by_pkg):
        mods = by_pkg[pkg]
        print(f"\n[{pkg}]  {len(mods)} leaves:")
        for m in mods:
            r_count = len(dependents.get(m, ()))
            print(f"  {short(m)}    ({r_count} dependents)")


def print_summary():
    modules, deps, dependents, _ = build_graph()
    print(f"\nL0 module count: {len(modules)}")
    by_pkg = defaultdict(int)
    leaves_by_pkg = defaultdict(int)
    for m in modules:
        pkg = m.split(".")[1]
        by_pkg[pkg] += 1
        if not deps.get(m):
            leaves_by_pkg[pkg] += 1
    print(f"\n{'sub-package':<16} {'modules':>8} {'leaves':>8}")
    for pkg in sorted(by_pkg):
        print(f"{pkg:<16} {by_pkg[pkg]:>8} {leaves_by_pkg[pkg]:>8}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pkg", help="Limit to one L0 sub-package")
    p.add_argument("--orphans", action="store_true", help="Show leaf modules only")
    p.add_argument("--reading-order", action="store_true", help="Flat reading order")
    p.add_argument("--summary", action="store_true", help="One-line summary")
    args = p.parse_args()

    if args.pkg and args.pkg not in L0_PACKAGES:
        print(f"ERROR: {args.pkg} is not an L0 sub-package. Options: {sorted(L0_PACKAGES)}", file=sys.stderr)
        return 1

    if args.summary:
        print_summary()
    elif args.orphans:
        print_orphans(args.pkg)
    elif args.reading_order:
        print_reading_order(args.pkg)
    else:
        print_tree(args.pkg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
