"""Tests for the auto-discovery path in `pricebook.core.serialization`.

Fix A.12 B1 — `_ensure_loaded` walks every submodule of `pricebook.*` rather
than relying on a curated 24-module whitelist. These tests assert:
1. Auto-discovery completes with no import failures.
2. The registry contains classes from across the codebase (not just the
   subset the old whitelist covered).
3. `from_dict` dispatches correctly to a type that was NOT in the old
   whitelist (regression-prevention for the silent-broken-on-new-modules
   failure mode).
"""

from __future__ import annotations


def test_auto_discovery_succeeds_with_no_import_failures():
    """No module in the pricebook tree should fail to import.

    A failure here means a new module under pricebook has an import-time
    error — a real bug to fix at the source, not silenced.
    """
    # Fresh import to avoid leakage from other tests.
    import importlib
    import pricebook.core.serialization
    importlib.reload(pricebook.core.serialization)
    from pricebook.core.serialization import _ensure_loaded, _failed_imports
    _ensure_loaded()
    assert _failed_imports == [], (
        f"Auto-discovery had {len(_failed_imports)} import failure(s): "
        f"{_failed_imports[:5]}"
    )


def test_registry_covers_many_types():
    """The registry should contain a substantial number of types — the
    pre-fix curated whitelist registered ~50; with auto-discovery we expect
    100+. Lower bound chosen to be robust against incremental additions
    without locking in a brittle exact count.
    """
    from pricebook.core.serialization import registered_types
    types = registered_types()
    assert len(types) >= 100, (
        f"Expected at least 100 registered types after auto-discovery; "
        f"got {len(types)}. The whitelist may have regressed."
    )


def test_registry_covers_each_subpackage():
    """Every major pricebook subpackage that uses @serialisable should
    have at least one of its types reachable via the registry."""
    from pricebook.core.serialization import _ensure_loaded, registered_types
    _ensure_loaded()
    types = set(registered_types())

    # One known type per subpackage (chosen to be stable wire identifiers).
    expected_examples = {
        "core/curves":       ("discount_curve", "survival_curve"),
        "core/trade":        ("trade", "portfolio"),
        "fixed_income":      ("swap", "bond"),
        "credit":            ("cds", "loan"),
        "fx":                ("fx_forward",),
        "equity":            ("trs",),
        "options":           ("swaption", "capfloor"),
    }
    missing: list[str] = []
    for subpkg, examples in expected_examples.items():
        if not any(e in types for e in examples):
            missing.append(f"{subpkg}: none of {examples} registered")
    assert missing == [], "subpackages missing from registry:\n  " + "\n  ".join(missing)


def test_registry_picks_up_module_not_in_old_whitelist():
    """Regression: a class from a module NOT in the pre-fix curated whitelist
    is now in the registry.

    `core.market_conventions` was NOT in the old `_ensure_loaded` curated
    list. Auto-discovery should pick its `@serialisable_convention` types
    (`equity_index_spec`, `commodity_contract_spec`, `linker_convention`).

    Convention objects use flat dicts (no `{"type": ..., "params": ...}`
    envelope) and round-trip via their own class-level `from_dict`, not via
    the global registry-dispatch `from_dict`. The fix is that the *class*
    is now registered (so it could be looked up if nested inside another
    object via `_deserialise_atom`).
    """
    from pricebook.core.market_conventions import EquityIndexSpec
    from pricebook.core.serialisable import _REGISTRY
    from pricebook.core.serialization import _ensure_loaded
    _ensure_loaded()
    assert "equity_index_spec" in _REGISTRY
    assert _REGISTRY["equity_index_spec"] is EquityIndexSpec
    # Round-trip the convention object via its own from_dict.
    spec = EquityIndexSpec(
        ticker="SPX",
        name="S&P 500",
        exchange="CBOE",
        currency="USD",
        settlement_lag=2,
        option_style="european",
        option_multiplier=100.0,
        dividend_frequency="quarterly",
        ex_date_rule="T-1",
    )
    d = spec.to_dict()
    rebuilt = EquityIndexSpec.from_dict(d)
    assert rebuilt == spec
