#!/usr/bin/env python3
"""Verification gates — carried from the migration tree, trimmed to what still applies.

**Trimmed on the reset (2026-10-30, `docs/handoffs/rulings_reset_to_fresh_repo.md`):** the
`quarry` gate (an import-floor on the old tree, which is now gitignored reference material) and
the drawdown/reconciliation reporting. Both measured *migration progress*, and there is no
migration to measure — the stock is mined by hand, never imported.

**Kept, because each exists for a defect that got through** and the reasons outlive the frame:
`acyclic` (layer direction), `layers` (semantic layer conformance), `fields` (≤5 dataclass
fields), `debt` (no silent suppressions), `provenance` (every module names its source and its
oracle), `version`, `slow` (the per-file wall-clock ceiling — three per-path-loop incidents),
and `tests` with a collection floor (CI once gated 177 of 366 tests while reporting green).

verify.py — the new tree's entire audit setup, one flat script.

Subcommands (each a small function; `all` is the CI / merge gate):

    acyclic            FAIL if any import points upward across layers
    tests --layer N    run only the tests at layer <= N (staged tests)
    fields             FAIL if a products/foundation value dataclass has >5 fields w/o exemption
    layers             FAIL if L0 (foundation/) contains finance vocabulary (semantic layer gate)
    debt               FAIL if suppressions - OPEN.md ng-ledger entries != 0
    slow               FAIL any test file over the per-file wall-clock ceiling (7c)
    provenance         FAIL if a landed module lacks its 4-line header
    version            FAIL if __version__ != top CHANGELOG.md entry
    all                everything above

The 5-argument ceiling (CLAUDE.md 3b) is enforced by the CI ruff step
(`ruff check src/pricebook`, rule PLR0913). Its dataclass-field analogue —
a product with 8 loose primitives is the same un-bundled smell PLR0913 can't
see — is the `fields` check here (docs/09).

No framework, no plugins — if a sixth check is ever needed it is one more
function then, not an abstraction now (CLAUDE.md 6b, docs/09).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src" / "pricebook"
TESTS = ROOT / "tests"
CHANGELOG = ROOT / "CHANGELOG.md"
OPEN = ROOT / "OPEN.md"

# The spine: package name -> layer rank. Dependencies may point DOWN (lower or
# equal rank) only, never up. This IS the law from CLAUDE.md 1.
LAYER = {
    "foundation": 0,
    "market": 1,
    "products": 2,
    "models": 3,
    "calibration": 3,
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
    """FAIL if any pricebook import targets a higher layer than its importer.

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
                if len(parts) >= 2 and parts[0] == "pricebook":
                    dep_layer = LAYER.get(parts[1])
                    if dep_layer is not None and dep_layer > importer_layer:
                        errs.append(
                            f"{rel}: {rel.parts[0]}(L{importer_layer}) imports "
                            f"{parts[1]}(L{dep_layer}) — upward import"
                        )
    return errs


#: Per-file wall-clock ceiling, seconds (CLAUDE.md 7c). Promoted from NG-WATCH-CLOCK on its
#: second instance. A scalar closed form called inside a path loop is O(paths x steps) Python
#: calls: it cost one file 73s (rates MC draft) and 59s (equity exposure draft), both under 1s
#: once vectorised. Measured headroom at the time of landing: the slowest file was 0.99s, and
#: most of that is interpreter startup -- so 10s is ~10x the slowest honest file and ~6x under
#: the failures it exists to catch, which runner variance cannot bridge in either direction.
SLOW_TEST_CEILING_S = 10.0
#: An honestly-heavy file says so, in the file, with a reason -- never silently (the
#: `fields-exempt` pattern). Exemptions are ledgered like any other suppression.
SLOW_EXEMPT = "# slow-test-exempt:"


def slow_tests() -> list[str]:
    """FAIL any test file over `SLOW_TEST_CEILING_S`. The floor rule's TIME analogue: the other
    faces bound what a check examined, this one bounds what it costs to examine it. Twice a
    per-path Python loop made a file 60-70x its vectorised cost, and nothing failed -- the suite
    was green and merely slow, which no gate could see."""
    out: list[str] = []
    for f in sorted(TESTS.rglob("test_*.py")):
        if SLOW_EXEMPT in f.read_text(encoding="utf-8"):
            continue
        t0 = time.time()
        subprocess.call(
            [sys.executable, "-m", "pytest", str(f), "-q", "--no-header"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        elapsed = time.time() - t0
        if elapsed > SLOW_TEST_CEILING_S:
            out.append(
                f"{f} took {elapsed:.1f}s > {SLOW_TEST_CEILING_S}s "
                f"(vectorise it, or mark it `{SLOW_EXEMPT} <reason>` and ledger the exemption)"
            )
    return out


def run_tests(max_layer: int | None) -> int:
    """Run tests at layer <= max_layer. Directory-per-layer: tests/L0..LN.

    `max_layer=None` runs EVERY layer dir present, and is what CI uses. A hardcoded
    ceiling is a filter with no floor (CLAUDE.md 6, second clause): `--layer 0` silently
    gated 177 of 366 tests for six slices while every report read "gates green", because
    the number has to be remembered and never was. Discovery cannot be forgotten.
    `--layer N` is kept for the per-tier local run CLAUDE.md 7b describes.
    """
    if max_layer is None:
        dirs = sorted(
            (str(d) for d in TESTS.glob("L*") if d.is_dir()),
            key=lambda d: int(d.rsplit("L", 1)[1]),
        )
    else:
        dirs = [str(TESTS / f"L{n}") for n in range(max_layer + 1) if (TESTS / f"L{n}").is_dir()]
    if not dirs and max_layer is None and TESTS.is_dir():
        # No `L*` tiers yet — a flat `tests/` is the scaffold's shape, and the layer tiers appear
        # as the layers do. Falling back here rather than failing keeps the gate meaningful on an
        # empty tree instead of making it something that must be remembered later.
        dirs = [str(TESTS)]
    if not dirs:
        print(f"no test dirs at layer <= {max_layer}")
        return 1
    rc = subprocess.call([sys.executable, "-m", "pytest", *dirs])
    if rc != 0:
        return rc
    # Floor rule, first face: floor-on-population. Green over a truncated collection is
    # not green (CLAUDE.md 6; ruling: docs/handoffs/rulings_floor_rule.md). Only the
    # un-truncated run is held to the floor; an explicit --layer N is a local tier run.
    if max_layer is None:
        n = _collected(dirs)
        print(f"collected {n} tests (floor {TEST_FLOOR})")
        if n < TEST_FLOOR:
            print(f"FAIL collected {n} < TEST_FLOOR {TEST_FLOOR} — truncated or shrunken suite")
            return 1
    return 0


def _collected(dirs: list[str]) -> int:
    """How many tests pytest actually collected, for the floor assertion."""
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", *dirs],
                         capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    return int(m.group(1)) if m else 0


_FIELD_LIMIT = 5


def _is_dataclass_decorator(dec: ast.expr) -> bool:
    if isinstance(dec, ast.Name):
        return dec.id == "dataclass"
    return isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "dataclass"


def _is_classvar(ann: ast.expr | None) -> bool:
    """A `ClassVar[...]` / `ClassVar` annotation is a class constant, not a dataclass field
    (dataclass excludes it from `__init__`), so it does not count toward the field limit."""
    node = ann.value if isinstance(ann, ast.Subscript) else ann  # unwrap ClassVar[...]
    if isinstance(node, ast.Attribute):
        return node.attr == "ClassVar"
    return isinstance(node, ast.Name) and node.id == "ClassVar"


def fields() -> list[str]:
    """A value dataclass has <= 5 fields — the dataclass analogue of PLR0913 (CLAUDE.md 3b):
    a product with 8 loose primitives is the same un-bundled smell PLR0913 can't see. Bundle
    primitives into ratified value objects (`Money`, `Accrual`, `ScheduleTerms`, …). A
    legitimately-wide aggregate / output-record / config type is allowed with an explicit
    `# fields-exempt: <reason>` marker in the class body."""
    errs: list[str] = []
    for mod in _src_modules():
        lines = _read(mod).splitlines()
        for node in ast.walk(ast.parse("\n".join(lines))):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(_is_dataclass_decorator(d) for d in node.decorator_list):
                continue
            n = sum(
                1
                for b in node.body
                if isinstance(b, ast.AnnAssign)
                and isinstance(b.target, ast.Name)
                and not _is_classvar(b.annotation)
            )
            span = "\n".join(lines[node.lineno - 1 : node.end_lineno or node.lineno])
            if n > _FIELD_LIMIT and "# fields-exempt:" not in span:
                errs.append(
                    f"{mod.relative_to(ROOT)}:{node.name} has {n} fields (>{_FIELD_LIMIT}) — "
                    "bundle into a value object or add '# fields-exempt: <reason>'"
                )
    return errs


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


#: The provenance header every landed module carries. `quarry:` became `stock:` on the
#: reset — same role (where the CONTENT came from) against the new reference tree.
_PROV_KEYS = ("stock:", "source:", "oracle:", "slice:")

#: Where mined content comes from. GITIGNORED, so it is present for local runs and absent
#: in CI — which the resolve-check below must handle WITHOUT quietly passing.
STOCK = ROOT / "_stock"


# The committed test floor (the floor rule, CLAUDE.md 6 second clause; ruling:
# docs/handoffs/rulings_floor_rule.md). CI is RED below it. A slice that adds a test
# tier raises this in the same commit. First value = the restart's measured 366.
TEST_FLOOR = 103

#: A `stock:` value naming no path — a module built fresh, with no stock ancestor. The
#: common case under the new model, so it must be expressible rather than fudged.
_NO_STOCK = {"—", "none", "(none", "(none)", "-"}


def provenance() -> list[str]:
    """Every landed module carries the 4-line provenance header, AND every `stock:` path RESOLVES.

    The floor rule's second face — resolve-don't-just-exist. Asserting the four lines are
    *present* is a proxy that passes while paths dangle: in the migration tree 17 of 45 did, two
    of them wrong since the day they were written. A presence check must dereference what it
    names, or it is not a gate.

    **`_stock/` is gitignored, so it is absent in CI**, and a resolve-check that silently skipped
    there would be a gate that cannot fail — this project's named defect class. So the skip is
    REPORTED, not silent: locally the paths are dereferenced, and in CI the gate says out loud
    that it checked headers only. `source:` and `oracle:` — the book or paper, and what proves the
    number — are verified in both, and under the new model they are the ones that matter: content
    is mined by hand, but the *oracle* is the contract.
    """
    errs: list[str] = []
    checked = 0
    have_stock = STOCK.is_dir()
    for mod in _src_modules():
        text = _read(mod)
        missing = [k for k in _PROV_KEYS if k not in text]
        if missing:
            errs.append(f"{mod.relative_to(ROOT)}: missing provenance {missing}")
        for key in ("source:", "oracle:"):
            if re.search(rf"{key}\s*(\S+)", text) is None:
                errs.append(f"{mod.relative_to(ROOT)}: {key} is empty")
        if not have_stock:
            continue
        for m in re.finditer(r"stock:\s*(\S+)", text):
            raw = m.group(1).rstrip(";,.")
            if raw in _NO_STOCK or not raw.startswith("_stock/"):
                continue
            checked += 1
            # a header may name a directory ("drew on several modules in curves/"), which is
            # legitimate provenance — it must still dereference to something real.
            if not (ROOT / raw).exists():
                errs.append(f"{mod.relative_to(ROOT)}: stock path does not resolve: {raw}")
    if not have_stock:
        print("    note: _stock/ absent (gitignored) — headers checked, stock paths NOT resolved")
        return errs
    landed = len(_src_modules())
    if landed == 0:
        # An empty library is the scaffold's honest state, not a violation. Say so out loud:
        # a gate reporting "ok" over zero modules would read as evidence of something.
        print("    note: no landed modules yet — nothing to check")
    elif checked == 0:
        # Floor-on-population: with modules present, SOME stock path must have been dereferenced,
        # or this gate examined nothing and its green means nothing. All-fresh-built modules
        # (every header `stock: —`) are legitimate — they are counted, so the floor sees them.
        fresh = sum(1 for m in _src_modules() if re.search(r"stock:\s*(—|none|-)", _read(m)))
        if fresh < landed:
            errs.append(
                f"{landed} modules, {fresh} declare no stock ancestor, yet NO stock path "
                "resolved — the check examined an empty population"
            )
    return errs


# The quarry must stay IMPORTABLE. Floor rule (CLAUDE.md §6) applied to the quarry gate:
# a committed floor on the importable-module count, RED on any drop. Rulings:
# docs/handoffs/rulings_quarry_park_model.md §3 — which also replaced an earlier proposal to assert
# `import pricebook` succeeds: that is a PROXY and it passed while 265 of 583 modules were broken
# (the top package's __init__ does not eagerly import its sub-packages).
def version() -> list[str]:
    """__version__ must match the top CHANGELOG.md entry."""
    m = re.search(r'__version__\s*=\s*"([^"]+)"', _read(SRC / "__init__.py"))
    ver = m.group(1) if m else None
    m = re.search(r"^## \[([0-9]+\.[0-9]+\.[0-9]+)\]", _read(CHANGELOG), flags=re.MULTILINE)
    top = m.group(1) if m else None
    if ver != top:
        return [f"__version__ ({ver}) != top CHANGELOG entry ({top})"]
    return []


# Finance vocabulary that must never appear in L0 (`foundation/` is finance-free). A
# safety net for the semantic-layer rule (CLAUDE.md 1): `acyclic` proves dependency
# DIRECTION but cannot see a module sitting on the WRONG layer. The `foundation/black.py`
# drift (Black-76 at L0) is exactly the case this catches. Curated to be near-zero
# false-positive in legitimate conventions/time/money code; the full semantic audit is
# the human 5th review input at each checkpoint.
_L0_FORBIDDEN = re.compile(
    r"\bstrike\b|\bpayoff\b|\bmoneyness\b|volatilit|implied_vol|black_scholes|"
    r"black_76|garman|kohlhagen|\bswaption\b|\bbarrier\b|discount_factor",
    re.IGNORECASE,
)


def layers() -> list[str]:
    """Semantic layer conformance: L0 (`foundation/`) is finance-free — no pricing or
    option vocabulary (strikes, vols, payoffs, discounting). `acyclic` enforces the
    dependency law; this enforces the layer *definition* (CLAUDE.md 1, the black.py
    precedent). Extend per-layer as later layers grow their own definitions."""
    errs: list[str] = []
    for mod in _src_modules():
        if mod.relative_to(SRC).parts[0] != "foundation":
            continue
        for i, line in enumerate(_read(mod).splitlines(), 1):
            m = _L0_FORBIDDEN.search(line)
            if m:
                errs.append(
                    f"{mod.relative_to(ROOT)}:{i}: L0 finance-free violation — "
                    f"'{m.group()}' (finance belongs at L2+/L3; see CLAUDE.md 1)"
                )
    return errs


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
    t.add_argument("--layer", type=int, default=None)  # omitted = every layer present
    sub.add_parser("fields")
    sub.add_parser("layers")
    sub.add_parser("debt")
    sub.add_parser("provenance")
    sub.add_parser("slow")
    sub.add_parser("version")
    sub.add_parser("all")
    args = ap.parse_args()

    if args.cmd == "tests":
        return run_tests(args.layer)
    checks = {
        "acyclic": acyclic, "fields": fields, "layers": layers, "debt": debt,
        "provenance": provenance, "version": version, "slow": slow_tests,
    }
    if args.cmd == "all":
        # `slow` re-runs every file separately, so it is NOT in `all` — CI calls it once,
        # explicitly, rather than making every local `all` pay for it.
        quick = {n: f for n, f in checks.items() if n != "slow"}
        return 0 if all(_report(n, f()) for n, f in quick.items()) else 1
    return 0 if _report(args.cmd, checks[args.cmd]()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
