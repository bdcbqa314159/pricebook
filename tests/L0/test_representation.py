"""The serialised representation — the scope line's durable public contract.

**Every test here is about the version actually being READ.** The first cut of this tranche
stamped `schema_version`, asserted in a test that it was *present*, and dereferenced it nowhere —
so a payload from a future build would have loaded silently under today's code, which is the exact
failure the field exists to prevent. Cowork's review caught it. Presence is not the property that
matters; refusal is.
"""

from datetime import date

import pytest

from pricebook.foundation import (
    SCHEMA_VERSION,
    Cashflow,
    Currency,
    IndexInterpolation,
    InflationIndex,
    Money,
    check_schema_version,
)
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
    """So a future migration can branch on it rather than re-reading the raw field. Pinned to the
    module's own constant rather than a literal: the point is that the value round-trips, and a
    literal here would have to be edited at every bump for no gain."""
    assert check_schema_version(_flow().to_dict()) == SCHEMA_VERSION


# ---- tranche 7: the index became a nested object, and a v1 payload must FAIL LOUDLY ----

def test_a_PRE_T7_bare_index_name_is_REFUSED_not_defaulted() -> None:
    """The v1 -> v2 break, asserted. A v1 payload wrote `"index": "HICPxt"` — a name, which cannot
    say which monthly print `I(t)` is. Reading one by assuming "3 months, linear" would price a UK
    gilt on the TIPS convention: a silent wrong number, which is exactly what tranche 7 removed
    from the engine. So the absent conventions are the caller's problem to resolve, not a gap
    `from_dict` fills in.

    Note `check_schema_version` alone CANNOT catch this — it only rejects payloads NEWER than the
    build, so a v1 payload passes the version gate and the refusal has to come from the reader."""
    with pytest.raises(ValueError, match="bare name"):
        InflationIndex.from_dict("HICPxt")


def test_the_index_round_trips_its_CONVENTIONS_not_only_its_name() -> None:
    """Two indices differing only in convention must survive the round trip as different objects —
    otherwise the representation silently merges a gilt and a TIPS."""
    gilt = InflationIndex("UKRPI", publication_lag=8, interpolation=IndexInterpolation.FLAT)
    tips = InflationIndex("UKRPI", publication_lag=3, interpolation=IndexInterpolation.LINEAR)
    assert InflationIndex.from_dict(gilt.to_dict()) == gilt
    assert InflationIndex.from_dict(tips.to_dict()) == tips
    assert gilt.to_dict() != tips.to_dict()
