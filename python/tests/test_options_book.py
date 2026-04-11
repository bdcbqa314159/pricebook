"""Tests for cross-asset options book."""

import pytest
from datetime import date

from pricebook.options_book import (
    AssetClassExposure,
    ExpiryBucket,
    OptionEntry,
    OptionsBook,
    VolPnLAttribution,
)


def _entry(ac="equity", underlying="AAPL", vega=100, gamma=5, theta=-10,
           delta=50, expiry=date(2024, 6, 15), trade_id="t1"):
    return OptionEntry(trade_id, ac, underlying, expiry, 1_000_000,
                       delta, gamma, vega, theta)


# ---- Step 1: unified vol book ----

class TestOptionsBook:
    def test_empty(self):
        book = OptionsBook("test")
        assert len(book) == 0
        assert book.total_vega() == 0.0

    def test_add_and_count(self):
        book = OptionsBook("test")
        book.add(_entry())
        assert len(book) == 1

    def test_aggregate_vega_equals_sum(self):
        """Step 1 test: aggregate vega = sum of per-asset vegas."""
        book = OptionsBook("test")
        book.add(_entry(ac="equity", vega=100))
        book.add(_entry(ac="fx", underlying="EUR/USD", vega=200, trade_id="t2"))
        book.add(_entry(ac="ir", underlying="5Y", vega=150, trade_id="t3"))
        book.add(_entry(ac="commodity", underlying="WTI", vega=80, trade_id="t4"))

        by_ac = book.by_asset_class()
        sum_ac = sum(a.net_vega for a in by_ac)
        assert sum_ac == pytest.approx(book.total_vega())
        assert book.total_vega() == pytest.approx(530)

    def test_by_asset_class(self):
        book = OptionsBook("test")
        book.add(_entry(ac="equity", vega=100, gamma=5, theta=-10))
        book.add(_entry(ac="equity", vega=50, gamma=3, theta=-5, trade_id="t2"))
        book.add(_entry(ac="fx", underlying="EUR/USD", vega=200, trade_id="t3"))
        by_ac = book.by_asset_class()
        assert len(by_ac) == 2
        eq = next(a for a in by_ac if a.asset_class == "equity")
        assert eq.net_vega == pytest.approx(150)
        assert eq.net_gamma == pytest.approx(8)
        assert eq.n_positions == 2

    def test_by_expiry(self):
        book = OptionsBook("test")
        book.add(_entry(expiry=date(2024, 3, 15), vega=100))
        book.add(_entry(expiry=date(2024, 6, 15), vega=200, trade_id="t2"))
        book.add(_entry(expiry=date(2024, 6, 15), vega=50, trade_id="t3"))
        by_exp = book.by_expiry()
        assert len(by_exp) == 2
        jun = next(b for b in by_exp if "2024-06-15" in b.expiry_label)
        assert jun.net_vega == pytest.approx(250)

    def test_n_asset_classes(self):
        book = OptionsBook("test")
        book.add(_entry(ac="equity"))
        book.add(_entry(ac="fx", trade_id="t2"))
        assert book.n_asset_classes == 2

    def test_total_gamma_theta(self):
        book = OptionsBook("test")
        book.add(_entry(gamma=5, theta=-10))
        book.add(_entry(gamma=3, theta=-8, trade_id="t2"))
        assert book.total_gamma() == pytest.approx(8)
        assert book.total_theta() == pytest.approx(-18)


# ---- Step 2: vol P&L attribution ----

class TestVolPnLAttribution:
    def test_vega_pnl(self):
        book = OptionsBook("test")
        book.add(_entry(ac="equity", underlying="AAPL", vega=1000))
        book.add(_entry(ac="fx", underlying="EUR/USD", vega=500, trade_id="t2"))
        attribs = book.vol_pnl_attribution(
            vol_changes={"AAPL": 0.01, "EUR/USD": 0.02},
        )
        eq = next(a for a in attribs if a.asset_class == "equity")
        fx = next(a for a in attribs if a.asset_class == "fx")
        assert eq.vega_pnl == pytest.approx(10.0)
        assert fx.vega_pnl == pytest.approx(10.0)

    def test_total_equals_sum_of_assets(self):
        """Step 2 test: total = sum of per-asset contributions."""
        book = OptionsBook("test")
        book.add(_entry(ac="equity", underlying="AAPL", vega=1000, gamma=50, theta=-20))
        book.add(_entry(ac="fx", underlying="EUR/USD", vega=500, gamma=30, theta=-15,
                        trade_id="t2"))
        attribs = book.vol_pnl_attribution(
            vol_changes={"AAPL": 0.01, "EUR/USD": 0.02},
            spot_changes={"AAPL": 2.0, "EUR/USD": 0.005},
            dt_days=1.0,
        )
        total = sum(a.total_pnl for a in attribs)
        # Verify each component sums
        total_vega = sum(a.vega_pnl for a in attribs)
        total_gamma = sum(a.gamma_pnl for a in attribs)
        total_theta = sum(a.theta_pnl for a in attribs)
        assert total == pytest.approx(total_vega + total_gamma + total_theta)

    def test_gamma_pnl_formula(self):
        book = OptionsBook("test")
        book.add(_entry(ac="equity", underlying="AAPL", gamma=100, vega=0, theta=0))
        attribs = book.vol_pnl_attribution(
            vol_changes={}, spot_changes={"AAPL": 4.0},
        )
        eq = attribs[0]
        # 0.5 × 100 × 16 = 800
        assert eq.gamma_pnl == pytest.approx(800.0)

    def test_theta_pnl(self):
        book = OptionsBook("test")
        book.add(_entry(ac="equity", underlying="AAPL", theta=-25, vega=0, gamma=0))
        attribs = book.vol_pnl_attribution(vol_changes={}, dt_days=1.0)
        assert attribs[0].theta_pnl == pytest.approx(-25.0)

    def test_missing_underlying_zero(self):
        book = OptionsBook("test")
        book.add(_entry(ac="equity", underlying="AAPL", vega=1000))
        attribs = book.vol_pnl_attribution(vol_changes={})
        assert attribs[0].vega_pnl == pytest.approx(0.0)
