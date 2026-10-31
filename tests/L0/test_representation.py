"""The serialised representation — the scope line's durable public contract.

**Every test here is about the version actually being READ.** The first cut of this tranche
stamped `schema_version`, asserted in a test that it was *present*, and dereferenced it nowhere —
so a payload from a future build would have loaded silently under today's code, which is the exact
failure the field exists to prevent. Cowork's review caught it. Presence is not the property that
matters; refusal is.
"""

from datetime import date

import pytest

from pricebook.foundation import Cashflow, Currency, Money, check_schema_version
from pricebook.products import FixedCashflow

USD = Currency("USD")
WHEN = date(2026, 6, 30)


def _flow() -> FixedCashflow:
    return FixedCashflow(Cashflow(WHEN, Money(1_234.56, USD)))


def test_a_product_round_trips() -> None:
    assert FixedCashflow.from_dict(_flow().to_dict()) == _flow()


def test_a_FUTURE_schema_version_is_REFUSED_not_best_effort_parsed() -> None:
    """The property the whole field exists for. A reader that guesses at a format it does not know
    produces a plausible wrong object, which is worse than a failure."""
    payload = _flow().to_dict()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="reads up to"):
        FixedCashflow.from_dict(payload)


def test_a_MISSING_schema_version_is_REFUSED() -> None:
    payload = _flow().to_dict()
    del payload["schema_version"]
    with pytest.raises(ValueError, match="no schema_version"):
        FixedCashflow.from_dict(payload)


def test_a_MALFORMED_schema_version_is_REFUSED() -> None:
    for bad in ("1", 0, -3, 1.5, True, None):
        payload = _flow().to_dict()
        payload["schema_version"] = bad
        with pytest.raises(ValueError):
            FixedCashflow.from_dict(payload)


def test_the_NESTED_cashflow_checks_its_version_too() -> None:
    """Both `from_dict`s call the same check, so a nested payload cannot slip past the outer one."""
    inner = Cashflow(WHEN, Money(1.0, USD)).to_dict()
    inner["schema_version"] = 99
    with pytest.raises(ValueError, match="reads up to"):
        Cashflow.from_dict(inner)


def test_the_version_is_checked_BEFORE_the_kind() -> None:
    """Order matters: in an unknown format, the `kind` field's meaning is itself unknown, so
    reporting 'not a fixed_cashflow' would be a guess about a payload we cannot read."""
    payload = _flow().to_dict()
    payload["schema_version"] = 99
    payload["kind"] = "something_else"
    with pytest.raises(ValueError, match="reads up to"):  # version, not kind
        FixedCashflow.from_dict(payload)


def test_the_check_returns_the_version_it_validated() -> None:
    """So a future migration can branch on it rather than re-reading the raw field."""
    assert check_schema_version(_flow().to_dict()) == 1
