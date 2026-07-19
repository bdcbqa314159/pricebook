"""Underlying / sibling-identity oracles (L0) — Topic 0 gate rework (F4/S10).

The general index/underlying concept: each sibling reports its `asset_class`, so a new
asset class slots in as one L0 identity (§3c). Defined now, populated later.
"""

from pricebook_ng.foundation.money import Currency, CurrencyPair, Unit
from pricebook_ng.foundation.tenor import Tenor, TenorUnit
from pricebook_ng.foundation.underlying import (
    AssetClass,
    CommodityUnderlying,
    EquityUnderlying,
    FxFixing,
    InflationIndex,
    InflationInterp,
    ReferenceEntity,
    Underlying,
)


def test_each_sibling_reports_its_asset_class():
    acme = ReferenceEntity("ACME")
    cpi = InflationIndex("USCPI", Tenor(3, TenorUnit.MONTH), InflationInterp.DAILY_LINEAR, 300.0)
    wmr = FxFixing("EURUSD_WMR", CurrencyPair(Currency.EUR, Currency.USD), "WMR", "16:00 London")
    spx = EquityUnderlying("SPX", "CBOE", Currency.USD)
    brent = CommodityUnderlying("BRENT", Unit.BARREL, "Sullom Voe", "Brent Blend")
    for u, ac in [(acme, AssetClass.CREDIT), (cpi, AssetClass.INFLATION), (wmr, AssetClass.FX),
                  (spx, AssetClass.EQUITY), (brent, AssetClass.COMMODITY)]:
        assert isinstance(u, Underlying)
        assert u.asset_class is ac


def test_commodity_identity_carries_location_and_grade():
    # Brent ≠ WTI is a matter of delivery location + grade (S10)
    brent = CommodityUnderlying("BRENT", Unit.BARREL, "Sullom Voe", "Brent Blend")
    wti = CommodityUnderlying("WTI", Unit.BARREL, "Cushing", "WTI")
    assert brent != wti
    assert brent.delivery_location == "Sullom Voe"
