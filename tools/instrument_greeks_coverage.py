"""Scan pricebook for instruments and report Greek-method coverage.

An "instrument" = any class in pricebook/ (excluding test/ and notebooks/)
that defines a `pv(...)` or `pv_ctx(...)` method. For each, check whether
the class also defines `dv01(...)` and/or `greeks(...)`.

Output: a markdown table grouped by sub-package. Two purposes:
  1. After T-PRC-PT2, this report quantifies what the RPC layer's narrowed
     except will silently skip (NotImplementedError + AttributeError paths
     will fire on every instrument missing the method).
  2. Roadmap for incremental Greek coverage: each row is a follow-up.

Usage:
    .venv/bin/python tools/instrument_greeks_coverage.py
    .venv/bin/python tools/instrument_greeks_coverage.py --missing-only
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

PRICEBOOK_ROOT = Path(__file__).resolve().parent.parent / "python" / "pricebook"

# We treat these method names as the "instrument contract" surface.
PV_METHODS = {"pv", "pv_ctx"}
GREEK_METHODS = {"dv01", "greeks"}


def scan_file(path: Path) -> list[dict]:
    """Return one entry per class that defines a pv-ish method.

    Each entry: {sub_pkg, module, class, methods}.
    """
    try:
        src = path.read_text()
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {
            n.name
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if PV_METHODS & methods:
            rel = path.relative_to(PRICEBOOK_ROOT)
            sub_pkg = rel.parts[0] if len(rel.parts) > 1 else "(root)"
            out.append({
                "sub_pkg": sub_pkg,
                "module": rel.as_posix(),
                "class": node.name,
                "methods": methods,
            })
    return out


def scan_all() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(PRICEBOOK_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rows.extend(scan_file(path))
    return rows


def format_markdown(rows: list[dict], missing_only: bool) -> str:
    by_pkg: dict[str, list[dict]] = {}
    for r in rows:
        by_pkg.setdefault(r["sub_pkg"], []).append(r)

    total = len(rows)
    have_dv01 = sum(1 for r in rows if "dv01" in r["methods"])
    have_greeks = sum(1 for r in rows if "greeks" in r["methods"])
    have_either = sum(
        1 for r in rows if GREEK_METHODS & r["methods"]
    )

    lines: list[str] = []
    lines.append("# Instrument Greek-coverage report")
    lines.append("")
    lines.append(
        f"**Scope:** {total} instrument classes (defined `pv` or `pv_ctx`) "
        f"across {len(by_pkg)} sub-packages."
    )
    lines.append("")
    lines.append("## Headline coverage")
    lines.append("")
    lines.append(f"- `dv01(...)`: {have_dv01} / {total} = {have_dv01 / total:.0%}")
    lines.append(f"- `greeks(...)`: {have_greeks} / {total} = {have_greeks / total:.0%}")
    lines.append(
        f"- Either method: {have_either} / {total} = {have_either / total:.0%}"
    )
    lines.append(f"- **Neither: {total - have_either} / {total} = "
                 f"{(total - have_either) / total:.0%}**")
    lines.append("")

    for pkg in sorted(by_pkg):
        pkg_rows = by_pkg[pkg]
        if missing_only:
            pkg_rows = [r for r in pkg_rows if not (GREEK_METHODS & r["methods"])]
            if not pkg_rows:
                continue
        n = len(pkg_rows)
        n_dv01 = sum(1 for r in pkg_rows if "dv01" in r["methods"])
        n_greeks = sum(1 for r in pkg_rows if "greeks" in r["methods"])
        n_neither = sum(1 for r in pkg_rows if not (GREEK_METHODS & r["methods"]))
        lines.append(
            f"## `{pkg}/` — {n} instruments "
            f"(dv01: {n_dv01}, greeks: {n_greeks}, neither: {n_neither})"
        )
        lines.append("")
        lines.append("| Module | Class | dv01 | greeks |")
        lines.append("|--------|-------|:----:|:------:|")
        for r in sorted(pkg_rows, key=lambda x: (x["module"], x["class"])):
            dv01 = "✅" if "dv01" in r["methods"] else "❌"
            greeks = "✅" if "greeks" in r["methods"] else "❌"
            lines.append(f"| `{r['module']}` | `{r['class']}` | {dv01} | {greeks} |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--missing-only", action="store_true",
                        help="Only list instruments missing BOTH dv01 and greeks")
    parser.add_argument("--out", type=str, default=None,
                        help="Write report to this file (default: stdout)")
    args = parser.parse_args()

    rows = scan_all()
    report = format_markdown(rows, args.missing_only)
    if args.out:
        Path(args.out).write_text(report)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
