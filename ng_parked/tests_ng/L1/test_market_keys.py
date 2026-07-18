"""Amendment A5 oracle — MarketKey namespacing (L1).

The keyed registry namespaces market data by asset class, so an FX currency "EUR"
and an equity ticker "EUR" are distinct keys that never collide in the snapshot's
`curves`/`spots`/`vols` maps.
"""

from datetime import date

from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot

D0 = date(2026, 1, 5)


def test_same_id_different_asset_do_not_collide():
    fx_eur = MarketKey(AssetClass.FX, "EUR")
    eq_eur = MarketKey(AssetClass.EQUITY, "EUR")
    assert fx_eur != eq_eur

    snap = MarketSnapshot(
        valuation_date=D0,
        discount_curve=FlatDiscountCurve(0.03, D0, DC.ACT_365_FIXED),
        spots={fx_eur: 1.10, eq_eur: 42.0},
    )
    assert snap.spots[fx_eur] == 1.10   # the FX spot
    assert snap.spots[eq_eur] == 42.0   # the equity spot — no collision


def test_marketkey_is_frozen_and_hashable():
    k = MarketKey(AssetClass.CREDIT, "ACME_CO")
    assert k in {k}                     # hashable
    assert k == MarketKey(AssetClass.CREDIT, "ACME_CO")   # value equality
