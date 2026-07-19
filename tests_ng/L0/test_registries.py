"""Registry-hardening oracles (L0) — Topic 0 final cleanup (A2).

The quarry's bug was *registry mutation* (a JSON load rebound `_REGISTRY` at import, dropping
27 of 28 indices). "Frozen everywhere" is this tree's ethos, so the registries are read-only:
the **closed** registries (calendars, rate indices) are `MappingProxyType`; the **open** ones
(currency, unit — S1/S4) keep a mutable backing behind the sanctioned `register_*` path but
expose a read-only view. Direct mutation raises either way; open registration still works.
"""

import pytest


def test_closed_calendar_registry_is_frozen():
    from pricebook_ng.foundation import market_calendars as mc
    with pytest.raises(TypeError):
        mc._CALENDARS["ZZZ"] = None


def test_closed_rate_index_registry_is_frozen():
    from pricebook_ng.foundation import rate_index as ri
    with pytest.raises(TypeError):
        ri._REGISTRY["ZZZ"] = None


def test_open_currency_registry_view_is_frozen():
    from pricebook_ng.foundation.money import CURRENCIES
    with pytest.raises(TypeError):
        CURRENCIES["ZZZ"] = None


def test_open_unit_registry_view_is_frozen():
    from pricebook_ng.foundation.money import UNITS
    with pytest.raises(TypeError):
        UNITS["ZZZ"] = None


def test_open_registration_still_works_through_the_sanctioned_path():
    # S1: a new market is a *declaration* — register_currency must keep working at runtime
    from pricebook_ng.foundation.money import CURRENCIES, currency, register_currency
    register_currency("QAR", 2)                            # Qatari riyal
    assert currency("QAR").code == "QAR"
    assert "QAR" in CURRENCIES                             # the frozen view reflects the addition
