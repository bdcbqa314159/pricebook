"""The live status instrument (T81) — generates `docs/STATUS.md`'s mechanical rows from the tree.

**Why this is a tool and not a page.** A hand-drafted `STATUS.md` was written at re-audit #9 and went
stale **within hours** — it said "T78 in review" after T78 had merged — and every mechanical row in it
duplicated `CHANGELOG.md`, `references.md`, the arc plans or `to_go_deeper.md`. §7 draws the line
between a *guarded* convention and *folklore* by whether a test compares the thing; a status page is
subject to the same rule, so the counts are derived here and a drift test compares the committed page
against a fresh generation. A stale page fails CI, which is the only reason it can be trusted.

**The split.** Everything between the `<!-- curated -->` markers is **judgement** — asset-class
readiness, the bibliography, the 1.0 scorecard — and is preserved verbatim; the tool regenerates the
rows around it. Refreshing the curated half is a re-audit job (§9's (4) pass).

**Structure and status only, never content.** `_stock/` is gitignored because the sources are
copyrighted. This page is tracked and may later render publicly, so it carries counts and module
names and no formula, source line or transcription — asserted by its own test, not merely intended.

Run `python tests/tooling/status.py --write` to regenerate.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

CURATED_START = "<!-- curated:start -->"
CURATED_END = "<!-- curated:end -->"
GENERATED_START = "<!-- generated:start -->"
GENERATED_END = "<!-- generated:end -->"

#: Engine module stem → asset class. **An explicit map, not a guess** — a classifier that inferred
#: the class from a name would quietly misfile, and a misfiled engine is a wrong row that looks
#: right. Anything absent is reported in `unclassified` rather than dropped, which is the
#: applied-count discipline (§9) in a counting tool: a tree that cannot be fully classified must
#: *say so*, because the alternative is a smaller, cleaner, wrong page.
ASSET_CLASSES: dict[str, str] = {
    "capfloor": "rates", "cap_strip": "rates", "nominal_cap": "rates", "hw_swaption": "rates",
    "hw_bermudan": "rates", "bermudan_swaption_mc": "rates", "swaption_mc": "rates",
    "interest_rate_swap": "rates", "smile_caplet": "rates", "cev_caplet": "rates",
    "hw_cap_calibration": "rates", "cash": "rates",
    "credit_default_swap": "credit", "forward_cds": "credit", "first_to_default_basket": "credit",
    "synthetic_cdo_tranche": "credit", "base_correlation": "credit", "loss_recursion": "credit",
    "defaultable_equity": "credit", "defaultable_equity_mc": "credit",
    "variance_replication": "volatility", "volatility_swap": "volatility",
    "realized_variance_option": "volatility", "vix": "volatility",
    "three_halves": "volatility", "time_changed_levy": "volatility", "timer_option": "volatility",
    "equity_vanilla": "equity", "heston_fourier": "equity", "heston_mc": "equity",
    "bates_fourier": "equity", "local_vol_mc": "equity", "equity_rate_hybrid": "equity",
    "hybrid_mc": "equity",
    "inflation": "inflation", "yoy": "inflation", "jy_calibration": "inflation",
    "jy_full_calibration": "inflation", "mmii_calibration": "inflation",
    "commodity_vanilla": "commodity", "commodity_asian": "commodity",
    "commodity_spread": "commodity", "schwartz": "commodity",
    "fourier_inversion": "shared", "mc_path_engine": "shared", "registry": "shared",
}

#: Risk-layer function prefixes that indicate a greek, mapped to the asset class they cover.
#: **Two entries, and finding the second one is why this is a tool.** The hand-drafted page said
#: "Greeks are near-empty (Black MC only)", and re-audit #9's F-9.1 note said `sensitivities.py`
#: holds "`black_mc_delta`/`gamma`/`vega` and nothing else" — both wrong: `digital_mc_delta` is
#: there too. The generator contradicted a claim both surfaces had accepted, on its first run
#: against the real tree. The column is still near-empty; it is near-empty by **four** functions
#: over two payoffs, not three over one.
GREEK_PREFIXES: dict[str, str] = {"black_mc_": "equity", "digital_mc_": "equity"}


@dataclass(frozen=True)
class Inventory:
    """What the tree actually contains. Every field derived, none typed in by hand."""

    models: int
    engines: int
    products: int
    registry_pairs: int
    engines_by_class: dict[str, int] = field(default_factory=dict)
    greeks_by_class: frozenset[str] = frozenset()
    unclassified: tuple[str, ...] = ()


def _modules(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.py") if p.name != "__init__.py")


def survey(
    root: Path,
    classes: dict[str, str] | None = None,
    greek_prefixes: dict[str, str] | None = None,
) -> Inventory:
    """Walk `src/pricebook` (or a fixture tree) and count what is there.

    The maps are **arguments with the real tree's as defaults**, not a hard-coded table with a
    name-prefix fallback for fixtures. The first draft had both paths live at once, and on the real
    tree the fallback fired alongside the map and invented two asset classes that do not exist
    (`black`, `digital`). A heuristic kept only for the tests had corrupted production's answer —
    so there is one path, and the fixture supplies its own map.
    """
    engines = _modules(root / "engine")
    by_class: Counter[str] = Counter()
    unclassified: list[str] = []
    registry_pairs = 0

    for module in engines:
        text = module.read_text()
        registry_pairs += len(re.findall(r"^@register\(", text, re.M))
        stem = module.stem
        asset_class = (classes if classes is not None else ASSET_CLASSES).get(stem)
        if asset_class is None:
            unclassified.append(stem)
        elif asset_class != "shared":
            by_class[asset_class] += 1

    prefixes = greek_prefixes if greek_prefixes is not None else GREEK_PREFIXES
    greeks: set[str] = set()
    for module in _modules(root / "risk"):
        text = module.read_text()
        for prefix, asset_class in prefixes.items():
            if re.search(rf"^def {re.escape(prefix)}", text, re.M):
                greeks.add(asset_class)

    return Inventory(
        models=len(_modules(root / "models")),
        engines=len(engines),
        products=len(_modules(root / "products")),
        registry_pairs=registry_pairs,
        engines_by_class=dict(sorted(by_class.items())),
        greeks_by_class=frozenset(greeks),
        unclassified=tuple(sorted(unclassified)),
    )


def mechanical_rows(
    root: Path,
    classes: dict[str, str] | None = None,
    greek_prefixes: dict[str, str] | None = None,
) -> list[str]:
    """The generated block: counts and inventories, no judgement and no source content."""
    inventory = survey(root, classes, greek_prefixes)
    lines = [
        "### Inventory — derived from the tree, regenerated and drift-gated",
        "",
        f"- **Models:** {inventory.models}",
        f"- **Engines:** {inventory.engines}, dispatching **{inventory.registry_pairs}** "
        "`(product, model)` pairs",
        f"- **Products:** {inventory.products}",
        "",
        "| Asset class | Engines | Greeks |",
        "|---|---|---|",
    ]
    classes = sorted(set(inventory.engines_by_class) | inventory.greeks_by_class)
    for asset_class in classes:
        count = inventory.engines_by_class.get(asset_class, 0)
        has_greeks = "yes" if asset_class in inventory.greeks_by_class else "—"
        lines.append(f"| {asset_class} | {count} | {has_greeks} |")
    if inventory.unclassified:
        lines += [
            "",
            f"**Unclassified engines ({len(inventory.unclassified)}):** "
            + ", ".join(f"`{name}`" for name in inventory.unclassified)
            + " — reported rather than dropped, so the table cannot look complete when it is not.",
        ]
    return lines


def render(
    root: Path,
    template: str,
    classes: dict[str, str] | None = None,
    greek_prefixes: dict[str, str] | None = None,
) -> str:
    """The page: the template's curated region verbatim, the generated region regenerated.

    Idempotent by construction — there is no timestamp in the generated region, because a page that
    changed on every run would make the drift gate assert nothing.
    """
    generated = "\n".join(mechanical_rows(root, classes, greek_prefixes))
    block = f"{GENERATED_START}\n{generated}\n{GENERATED_END}"
    if GENERATED_START in template and GENERATED_END in template:
        head, rest = template.split(GENERATED_START, 1)
        _, tail = rest.split(GENERATED_END, 1)
        return f"{head}{block}{tail}"
    return f"{template.rstrip()}\n\n{block}\n"


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[2]
    page = root / "docs" / "STATUS.md"
    current = page.read_text() if page.exists() else ""
    fresh = render(root / "src" / "pricebook", current)
    if "--write" in argv:
        page.write_text(fresh)
        print(f"wrote {page.relative_to(root)}")
        return 0
    if current != fresh:
        print("docs/STATUS.md is STALE — run with --write")
        return 1
    print("docs/STATUS.md is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
