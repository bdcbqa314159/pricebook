#!/usr/bin/env python3
"""Module-level dependency tree for any pricebook layer (L0..L6).

For every module under the chosen layer's sub-packages, computes:

  * direct intra-layer imports (this module's deps at load time)
  * who imports it (this module's dependents)
  * topological reading order within its own sub-package

Same AST rules as ``test_layer.py``: top-level + class-body imports,
skip TYPE_CHECKING, skip function-local. Empirical, not from-docstrings.

Sub-package membership of layers is computed live from the source tree
via ``test_layer.py``'s ``compute_layers()`` — no hard-coded list, so
adding a new sub-package automatically lands in the right place.

CLI::

    tools/layer_deps.py --layer 0                    # full tree for L0, to stdout
    tools/layer_deps.py --layer 0 --write            # full tree, write to L0_DEPS.md
    tools/layer_deps.py --layer 1 --pkg curves       # one sub-package only
    tools/layer_deps.py --layer 0 --orphans          # leaf modules only
    tools/layer_deps.py --layer 0 --reading-order    # flat reading order
    tools/layer_deps.py --layer 0 --summary          # one-line counts
"""
from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "python" / "pricebook"

# Reuse test_layer.compute_layers() so the layer assignment matches the
# rest of the audit/test tooling.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_layer import compute_layers  # noqa: E402


def packages_for_layer(target_layer: int) -> set[str]:
    """All sub-packages whose layer == target_layer (per compute_layers)."""
    return {pkg for pkg, lvl in compute_layers().items() if lvl == target_layer}


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


class _LocalImportCollector(ast.NodeVisitor):
    """Mirror of _ImportCollector but captures ONLY function-body imports."""
    def __init__(self) -> None:
        self.depth = 0
        self.targets: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Import(self, node: ast.Import) -> None:
        if not self.depth:
            return
        for alias in node.names:
            if alias.name.startswith("pricebook"):
                self.targets.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not self.depth:
            return
        mod = node.module or ""
        if mod.startswith("pricebook"):
            self.targets.append(mod)


def _collect_local_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    coll = _LocalImportCollector()
    coll.visit(tree)
    return coll.targets


# ──────────────────────────────────────────────────────────────────────────
# Build the module graph for a chosen layer
# ──────────────────────────────────────────────────────────────────────────

def _modname(p: Path) -> str:
    """Convert a file path under SRC_ROOT to a dotted pricebook module name."""
    rel = p.relative_to(SRC_ROOT)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return "pricebook." + ".".join(parts)


def _in_layer(modname: str, layer_pkgs: set[str]) -> bool:
    parts = modname.split(".")
    return len(parts) >= 2 and parts[0] == "pricebook" and parts[1] in layer_pkgs


def _resolve(imp: str, all_modules: set[str]) -> str | None:
    if imp in all_modules:
        return imp
    parts = imp.split(".")
    while len(parts) > 1:
        candidate = ".".join(parts)
        if candidate in all_modules:
            return candidate
        parts = parts[:-1]
    return None


def build_graph(layer_pkgs: set[str]):
    """Returns (modules, deps, dependents, paths) restricted to layer_pkgs."""
    modules: set[str] = set()
    paths: dict[str, Path] = {}
    for py in SRC_ROOT.rglob("*.py"):
        if any(p.startswith("_legacy") for p in py.parts):
            continue
        m = _modname(py)
        if _in_layer(m, layer_pkgs):
            modules.add(m)
            paths[m] = py

    deps: dict[str, set[str]] = defaultdict(set)
    dependents: dict[str, set[str]] = defaultdict(set)

    for m in sorted(modules):
        for imp in _collect_imports(paths[m]):
            tgt = _resolve(imp, modules)
            if tgt is None or tgt == m or not _in_layer(tgt, layer_pkgs):
                continue
            deps[m].add(tgt)
            dependents[tgt].add(m)

    return modules, deps, dependents, paths


def collect_cross_pkg_lazy(layer_pkgs: set[str]):
    """Lazy (function-body) imports that cross sub-package boundaries within
    the chosen layer. Returns sorted unique list of (source_module, target_module).
    """
    edges: set[tuple[str, str]] = set()
    for py in SRC_ROOT.rglob("*.py"):
        parts = py.relative_to(SRC_ROOT).parts
        if not parts or parts[0] not in layer_pkgs:
            continue
        src_pkg = parts[0]
        for imp in _collect_local_imports(py):
            pp = imp.split(".")
            if (len(pp) >= 2 and pp[0] == "pricebook"
                    and pp[1] in layer_pkgs and pp[1] != src_pkg):
                src_mod = _modname(py)
                edges.add((src_mod, imp))
    return sorted(edges)


def compute_depth(modules: set[str], deps: dict[str, set[str]]) -> dict[str, int]:
    """For each module: depth = longest path from it to a leaf.

    Leaves have depth 0. A module that imports a depth-d module has
    depth ≥ d+1. Cycles get treated as depth 0 at the cycle entry — the
    function is robust to (but does not warn on) cycles.
    """
    depth: dict[str, int] = {}

    def _depth(m: str, stack: set[str]) -> int:
        if m in depth:
            return depth[m]
        if m in stack:
            return 0  # cycle — treat as leaf for ordering purposes
        intra_deps = [d for d in deps.get(m, ()) if d in modules]
        if not intra_deps:
            depth[m] = 0
            return 0
        stack.add(m)
        d = 1 + max(_depth(dd, stack) for dd in intra_deps)
        stack.discard(m)
        depth[m] = d
        return d

    for m in modules:
        _depth(m, set())
    return depth


def depth_order(modules: set[str], deps: dict[str, set[str]]) -> list[str]:
    """Return modules sorted by (depth ascending, name).

    Depth 0 (leaves) come first; deeper-in-the-tree modules (those that
    sit on top of more layers) come last. This is the "most deep to most
    superficial" ordering — within a depth level, ties are broken
    alphabetically.
    """
    d = compute_depth(modules, deps)
    return sorted(modules, key=lambda m: (d[m], m))


# Kept for back-compat with callers that asked for "topo order". Same
# semantics as depth_order now (which is itself a valid topological
# order — every module appears after all of its deps).
topo_order = depth_order


# ──────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────

def short(m: str) -> str:
    return m[len("pricebook."):] if m.startswith("pricebook.") else m


def render_tree(layer: int, layer_pkgs: set[str], pkg_filter: str | None = None) -> str:
    modules, deps, dependents, _ = build_graph(layer_pkgs)
    order = depth_order(modules, deps)
    depths = compute_depth(modules, deps)

    by_pkg: dict[str, list[str]] = defaultdict(list)
    for m in order:
        pkg = m.split(".")[1]
        if pkg_filter and pkg != pkg_filter:
            continue
        by_pkg[pkg].append(m)

    lines: list[str] = []
    for pkg in sorted(by_pkg):
        mods = by_pkg[pkg]
        lines.append("")
        lines.append("=" * 72)
        lines.append(f"  pricebook.{pkg}    ({len(mods)} modules, deep → superficial)")
        lines.append("=" * 72)
        prev_depth = -1
        for m in mods:
            d = sorted(deps.get(m, ()))
            r = sorted(dependents.get(m, ()))
            depth_m = depths[m]
            # Visual band marker each time depth bumps up.
            if depth_m != prev_depth:
                lines.append("")
                lines.append(f"  -- depth {depth_m} "
                             f"({'leaves' if depth_m == 0 else f'sits on top of {depth_m} layer(s)'}) --")
                prev_depth = depth_m
            tag = "" if d else " [LEAF]"
            lines.append(f"  d{depth_m}  {short(m)}{tag}")
            if d:
                for dd in d:
                    lines.append(f"          → {short(dd)}  (d{depths[dd]})")
            if r:
                in_pkg = sum(1 for rr in r if rr.split(".")[1] == pkg)
                cross_pkg = len(r) - in_pkg
                lines.append(f"          ← {len(r)} dependents ({in_pkg} intra, "
                             f"{cross_pkg} cross-pkg)")
    return "\n".join(lines)


def render_reading_order(layer: int, layer_pkgs: set[str], pkg_filter: str | None = None) -> str:
    modules, deps, _, _ = build_graph(layer_pkgs)
    order = depth_order(modules, deps)
    depths = compute_depth(modules, deps)
    if pkg_filter:
        order = [m for m in order if m.split(".")[1] == pkg_filter]
    out = [f"# L{layer} reading order — deep → superficial ({len(order)} modules)\n"]
    prev_depth = -1
    for i, m in enumerate(order, 1):
        depth_m = depths[m]
        if depth_m != prev_depth:
            out.append(f"\n--- depth {depth_m} ---")
            prev_depth = depth_m
        d = sorted(deps.get(m, ()))
        leaf = " [LEAF]" if not d else ""
        out.append(f"{i:3}.  d{depth_m}  {short(m)}{leaf}")
    return "\n".join(out)


def render_orphans(layer: int, layer_pkgs: set[str], pkg_filter: str | None = None) -> str:
    modules, deps, dependents, _ = build_graph(layer_pkgs)
    by_pkg: dict[str, list[str]] = defaultdict(list)
    for m in sorted(modules):
        if pkg_filter and m.split(".")[1] != pkg_filter:
            continue
        if not deps.get(m):
            by_pkg[m.split(".")[1]].append(m)
    out: list[str] = []
    for pkg in sorted(by_pkg):
        mods = by_pkg[pkg]
        out.append(f"\n[{pkg}]  {len(mods)} leaves:")
        for m in mods:
            out.append(f"  {short(m)}    ({len(dependents.get(m, ()))} dependents)")
    return "\n".join(out)


def render_summary(layer: int, layer_pkgs: set[str]) -> str:
    modules, deps, _, _ = build_graph(layer_pkgs)
    by_pkg: dict[str, int] = defaultdict(int)
    leaves_by_pkg: dict[str, int] = defaultdict(int)
    for m in modules:
        pkg = m.split(".")[1]
        by_pkg[pkg] += 1
        if not deps.get(m):
            leaves_by_pkg[pkg] += 1
    lines = [f"\nL{layer} module count: {len(modules)}"]
    lines.append(f"L{layer} sub-packages: {sorted(layer_pkgs)}")
    lines.append("")
    lines.append(f"{'sub-package':<16} {'modules':>8} {'leaves':>8}")
    for pkg in sorted(by_pkg):
        lines.append(f"{pkg:<16} {by_pkg[pkg]:>8} {leaves_by_pkg[pkg]:>8}")
    return "\n".join(lines)


def render_markdown(layer: int, layer_pkgs: set[str]) -> str:
    """Self-contained markdown report — the canonical artefact for ``L<N>_DEPS.md``."""
    modules, deps, dependents, _ = build_graph(layer_pkgs)
    order = depth_order(modules, deps)
    depths = compute_depth(modules, deps)
    lazy = collect_cross_pkg_lazy(layer_pkgs)

    out: list[str] = []
    out.append(f"# L{layer} dependency tree — refactoring reading order")
    out.append("")
    out.append(f"**Generated:** by `tools/layer_deps.py --layer {layer} --write` (regenerate any time)")
    out.append(f"**Scope:** {len(modules)} modules across {len(layer_pkgs)} L{layer} sub-package(s): {sorted(layer_pkgs)}")
    out.append("**Method:** AST parse of every `.py`; top-level + class-body imports counted; `TYPE_CHECKING` blocks and function-local imports excluded from the load-time graph (function-local imports listed separately below as the 'lazy' picture).")
    out.append("")
    out.append("This document is for **systematic, layer-by-layer refactoring**. Read modules in the order listed within each sub-package: every module is positioned **after all its dependencies**, so by the time you reach a module you've already read everything it imports at load time.")
    out.append("")
    out.append("Auto-gitignored (`/*.md` rule); regenerate freely.")
    out.append("")
    out.append("---")
    out.append("")

    # Summary
    out.append("## Summary")
    out.append("")
    out.append(f"| Sub-package | Modules | Leaves (no L{layer} deps at load time) |")
    out.append("|---|---:|---:|")
    by_pkg: dict[str, int] = defaultdict(int)
    leaves_by_pkg: dict[str, int] = defaultdict(int)
    for m in modules:
        pkg = m.split(".")[1]
        by_pkg[pkg] += 1
        if not deps.get(m):
            leaves_by_pkg[pkg] += 1
    for pkg in sorted(by_pkg):
        out.append(f"| {pkg} | {by_pkg[pkg]} | {leaves_by_pkg[pkg]} |")
    total_leaves = sum(leaves_by_pkg.values())
    out.append(f"| **Total** | **{len(modules)}** | **{total_leaves}** |")
    out.append("")

    # Lazy cross-package edges
    out.append(f"## Lazy cross-sub-package edges (function-body imports within L{layer})")
    out.append("")
    if len(layer_pkgs) == 1:
        out.append(f"N/A — L{layer} has a single sub-package.")
    elif not lazy:
        out.append(f"None. Sub-packages within L{layer} are independent at both load-time AND call-time.")
    else:
        out.append(f"{len(lazy)} lazy edges. Sub-packages are independent at load-time but use these at call-time (cycle-breakers / optional dependencies):")
        out.append("")
        out.append("| Source module | → Target module |")
        out.append("|---|---|")
        for src, tgt in lazy:
            tgt_short = tgt[len("pricebook."):] if tgt.startswith("pricebook.") else tgt
            out.append(f"| `{short(src)}` | `{tgt_short}` |")
    out.append("")

    # Per-sub-package tree
    out.append("## Per-sub-package trees (deep → superficial)")
    out.append("")
    out.append(f"For each module: `d<N>` is its **depth** (longest path to a leaf within this layer; depth 0 = pure leaf, depth N = sits on top of N layers). `→` lines are load-time intra-L{layer} deps with their depths shown. `← N dependents (M intra, K cross-pkg)` shows fan-in. Within a sub-package, modules are sorted by (depth ascending, name) — so every leaf comes before every depth-1 module, which comes before every depth-2 module, etc.")
    out.append("")
    by_pkg_order: dict[str, list[str]] = defaultdict(list)
    for m in order:
        by_pkg_order[m.split(".")[1]].append(m)
    for pkg in sorted(by_pkg_order):
        mods = by_pkg_order[pkg]
        out.append(f"### {pkg} ({len(mods)} modules)")
        out.append("")
        out.append("```")
        prev_depth = -1
        for m in mods:
            d = sorted(deps.get(m, ()))
            r = sorted(dependents.get(m, ()))
            depth_m = depths[m]
            if depth_m != prev_depth:
                if prev_depth >= 0:
                    out.append("")
                out.append(f"  -- depth {depth_m} "
                           f"({'leaves' if depth_m == 0 else f'on top of {depth_m} layer(s)'}) --")
                prev_depth = depth_m
            tag = "" if d else " [LEAF]"
            out.append(f"  d{depth_m}  {short(m)}{tag}")
            if d:
                for dd in d:
                    out.append(f"          → {short(dd)}  (d{depths[dd]})")
            if r:
                in_pkg = sum(1 for rr in r if rr.split(".")[1] == pkg)
                cross_pkg = len(r) - in_pkg
                out.append(f"          ← {len(r)} dependents ({in_pkg} intra, {cross_pkg} cross-pkg)")
        out.append("```")
        out.append("")

    # Flat reading order
    out.append(f"## Flat reading order (all {len(modules)} modules, deep → superficial)")
    out.append("")
    out.append("```")
    prev_depth = -1
    for i, m in enumerate(order, 1):
        depth_m = depths[m]
        if depth_m != prev_depth:
            if prev_depth >= 0:
                out.append("")
            out.append(f"  --- depth {depth_m} ---")
            prev_depth = depth_m
        d = sorted(deps.get(m, ()))
        leaf = " [LEAF]" if not d else ""
        out.append(f"{i:3}.  d{depth_m}  {short(m)}{leaf}")
    out.append("```")
    out.append("")

    # Tool usage
    out.append("## Tool usage")
    out.append("")
    out.append("```bash")
    out.append("# Regenerate this file (overwrites L<N>_DEPS.md):")
    out.append(f".venv/bin/python tools/layer_deps.py --layer {layer} --write")
    out.append("")
    out.append("# Full tree to stdout:")
    out.append(f".venv/bin/python tools/layer_deps.py --layer {layer}")
    out.append("")
    out.append("# One sub-package only:")
    out.append(f".venv/bin/python tools/layer_deps.py --layer {layer} --pkg <name>")
    out.append("")
    out.append("# Leaves only:")
    out.append(f".venv/bin/python tools/layer_deps.py --layer {layer} --orphans")
    out.append("")
    out.append("# Flat reading order:")
    out.append(f".venv/bin/python tools/layer_deps.py --layer {layer} --reading-order")
    out.append("")
    out.append("# Summary stats:")
    out.append(f".venv/bin/python tools/layer_deps.py --layer {layer} --summary")
    out.append("```")
    out.append("")
    return "\n".join(out)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layer", type=int, required=True,
                   help="Layer number (0..6). Sub-packages chosen via test_layer.compute_layers.")
    p.add_argument("--pkg", help="Limit to one sub-package within the layer")
    p.add_argument("--orphans", action="store_true", help="Show leaf modules only")
    p.add_argument("--reading-order", action="store_true", help="Flat reading order")
    p.add_argument("--summary", action="store_true", help="One-line counts")
    p.add_argument("--write", action="store_true",
                   help="Write full markdown report to L<layer>_DEPS.md at repo root")
    args = p.parse_args()

    layer_pkgs = packages_for_layer(args.layer)
    if not layer_pkgs:
        print(f"ERROR: no sub-packages found at layer L{args.layer}.\n"
              f"Available layers: see `tools/test_layer.py --show-layers`.",
              file=sys.stderr)
        return 1

    if args.pkg and args.pkg not in layer_pkgs:
        print(f"ERROR: '{args.pkg}' is not in L{args.layer}. "
              f"Options: {sorted(layer_pkgs)}", file=sys.stderr)
        return 1

    if args.write:
        out_path = REPO_ROOT / f"L{args.layer}_DEPS.md"
        out_path.write_text(render_markdown(args.layer, layer_pkgs))
        print(f"Wrote {out_path}")
        return 0

    if args.summary:
        print(render_summary(args.layer, layer_pkgs))
    elif args.orphans:
        print(render_orphans(args.layer, layer_pkgs, args.pkg))
    elif args.reading_order:
        print(render_reading_order(args.layer, layer_pkgs, args.pkg))
    else:
        print(render_tree(args.layer, layer_pkgs, args.pkg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
