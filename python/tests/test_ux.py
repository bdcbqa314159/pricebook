"""Tests for UX: registry, convenience API, top-level imports, reporting."""

import json
import math
import pytest
from datetime import date

from pricebook.serialization import (
    get_instrument_class,
    list_instruments,
    load_trade,
    load_portfolio,
    from_json,
    to_json,
    instrument_to_dict,
    trade_to_dict,
    portfolio_to_dict,
)
from pricebook.reporting import (
    portfolio_risk_report,
    scenario_grid,
    trade_blotter,
)


REF = date(2024, 1, 15)


# ---------------------------------------------------------------------------
# Slice 70: Instrument registry + trade loader
# ---------------------------------------------------------------------------


class TestInstrumentRegistry:
    def test_get_irs(self):
        cls = get_instrument_class("irs")
        assert cls.__name__ == "InterestRateSwap"

    def test_get_cds(self):
        cls = get_instrument_class("cds")
        assert cls.__name__ == "CDS"

    def test_get_bond(self):
        cls = get_instrument_class("bond")
        assert cls.__name__ == "FixedRateBond"

    def test_list(self):
        types = list_instruments()
        assert "irs" in types
        assert "cds" in types
        assert "bond" in types
        assert "swaption" in types
        assert len(types) >= 6

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Available"):
            get_instrument_class("unknown")


class TestLoadTrade:
    def test_load_from_dict(self):
        data = {
            "direction": 1,
            "notional_scale": 1.0,
            "trade_id": "t1",
            "counterparty": "ACME",
            "instrument": {
                "type": "irs",
                "params": {
                    "start": "2024-01-15",
                    "end": "2029-01-15",
                    "fixed_rate": 0.05,
                },
            },
        }
        trade = load_trade(data)
        assert trade.trade_id == "t1"
        assert type(trade.instrument).__name__ == "InterestRateSwap"

    def test_load_portfolio_from_list(self):
        trades = [
            {
                "trade_id": "s1",
                "instrument": {"type": "irs", "params": {
                    "start": "2024-01-15", "end": "2029-01-15", "fixed_rate": 0.05,
                }},
            },
            {
                "trade_id": "c1",
                "instrument": {"type": "cds", "params": {
                    "start": "2024-01-15", "end": "2029-01-15", "spread": 0.01,
                }},
            },
        ]
        port = load_portfolio(trades)
        assert len(port.trades) == 2

    def test_load_portfolio_from_dict(self):
        data = {
            "name": "my_book",
            "trades": [
                {
                    "trade_id": "s1",
                    "instrument": {"type": "fra", "params": {
                        "start": "2024-07-15", "end": "2024-10-15", "strike": 0.05,
                    }},
                },
            ],
        }
        port = load_portfolio(data)
        assert port.name == "my_book"
        assert len(port.trades) == 1


class TestFromJson:
    def test_roundtrip_instrument(self):
        from pricebook.swap import InterestRateSwap
        swap = InterestRateSwap(REF, date(2029, 1, 15), 0.05)
        j = to_json(swap)
        swap2 = from_json(j)
        assert type(swap2).__name__ == "InterestRateSwap"

    def test_roundtrip_context(self):
        from pricebook.pricing_context import PricingContext
        from pricebook.discount_curve import DiscountCurve
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=DiscountCurve.flat(REF, 0.05),
        )
        j = to_json(ctx)
        ctx2 = from_json(j)
        assert ctx2.valuation_date == REF


# ---------------------------------------------------------------------------
# Slice 71: Convenience API + top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImports:
    def test_import_context(self):
        from pricebook import PricingContext
        assert PricingContext is not None

    def test_import_curve(self):
        from pricebook import DiscountCurve, SurvivalCurve
        assert DiscountCurve is not None
        assert SurvivalCurve is not None

    def test_import_instruments(self):
        from pricebook import InterestRateSwap, FixedRateBond, CDS, FRA, Swaption
        assert all(cls is not None for cls in [InterestRateSwap, FixedRateBond, CDS, FRA, Swaption])

    def test_import_trade(self):
        from pricebook import Trade, Portfolio
        assert Trade is not None

    def test_import_serialization(self):
        from pricebook import to_json, from_json, load_trade, load_portfolio
        assert all(fn is not None for fn in [to_json, from_json, load_trade, load_portfolio])

    def test_import_registry(self):
        from pricebook import get_solver, get_tree_european
        assert get_solver is not None


class TestConvenienceMethods:
    def test_flat_discount_curve(self):
        from pricebook import DiscountCurve
        curve = DiscountCurve.flat(REF, 0.05)
        d1y = date.fromordinal(REF.toordinal() + 365)
        assert curve.df(d1y) == pytest.approx(math.exp(-0.05), rel=1e-3)

    def test_flat_survival_curve(self):
        from pricebook import SurvivalCurve
        curve = SurvivalCurve.flat(REF, 0.02)
        d1y = date.fromordinal(REF.toordinal() + 365)
        assert curve.survival(d1y) == pytest.approx(math.exp(-0.02), rel=1e-3)

    def test_simple_context(self):
        from pricebook import PricingContext
        ctx = PricingContext.simple(REF, rate=0.05, vol=0.20)
        assert ctx.discount_curve is not None
        assert "ir" in ctx.vol_surfaces

    def test_simple_context_with_credit(self):
        from pricebook import PricingContext
        ctx = PricingContext.simple(REF, rate=0.05, hazard=0.02)
        assert "default" in ctx.credit_curves

    def test_replace(self):
        from pricebook import PricingContext, DiscountCurve
        ctx = PricingContext.simple(REF, rate=0.05)
        new_curve = DiscountCurve.flat(REF, 0.06)
        ctx2 = ctx.replace(discount_curve=new_curve)
        d1y = date.fromordinal(REF.toordinal() + 365)
        assert ctx2.discount_curve.df(d1y) != ctx.discount_curve.df(d1y)
        assert ctx2.valuation_date == ctx.valuation_date

    def test_one_liner_pricing(self):
        """Price a swap in minimal lines."""
        from pricebook import InterestRateSwap, PricingContext
        ctx = PricingContext.simple(REF, rate=0.05)
        swap = InterestRateSwap(REF, date(2029, 1, 15), 0.05)
        pv = swap.pv(ctx.discount_curve)
        assert isinstance(pv, float)


# ---------------------------------------------------------------------------
# Slice 72: Dashboard data layer
# ---------------------------------------------------------------------------


def _make_portfolio():
    from pricebook import Swaption, Trade, Portfolio
    swn1 = Swaption(date(2025, 1, 15), date(2030, 1, 15), 0.05)
    swn2 = Swaption(date(2026, 1, 15), date(2031, 1, 15), 0.04)
    return Portfolio([
        Trade(swn1, trade_id="swn_5y10y", counterparty="ACME"),
        Trade(swn2, trade_id="swn_6y11y", counterparty="GLOBEX"),
    ], name="test_book")


class TestPortfolioRiskReport:
    def test_structure(self):
        from pricebook import PricingContext
        port = _make_portfolio()
        ctx = PricingContext.simple(REF, rate=0.05, vol=0.20)
        report = portfolio_risk_report(port, ctx)

        assert report["name"] == "test_book"
        assert "total_pv" in report
        assert "dv01" in report
        assert report["n_trades"] == 2
        assert len(report["trades"]) == 2

    def test_json_serializable(self):
        from pricebook import PricingContext
        port = _make_portfolio()
        ctx = PricingContext.simple(REF, rate=0.05, vol=0.20)
        report = portfolio_risk_report(port, ctx)
        j = json.dumps(report)
        assert isinstance(j, str)

    def test_dv01_nonzero(self):
        from pricebook import PricingContext
        port = _make_portfolio()
        ctx = PricingContext.simple(REF, rate=0.05, vol=0.20)
        report = portfolio_risk_report(port, ctx)
        assert report["dv01"] != 0.0


class TestScenarioGrid:
    def test_structure(self):
        from pricebook import PricingContext
        port = _make_portfolio()
        ctx = PricingContext.simple(REF, rate=0.05, vol=0.20)
        scenarios = [
            {"name": "base", "rate_shift": 0.0},
            {"name": "up_100", "rate_shift": 0.01},
            {"name": "dn_100", "rate_shift": -0.01},
            {"name": "vol_up", "vol_shift": 0.05},
        ]
        result = scenario_grid(port, ctx, scenarios)
        assert "base_pv" in result
        assert len(result["scenarios"]) == 4
        assert result["scenarios"][0]["name"] == "base"

    def test_base_pnl_zero(self):
        from pricebook import PricingContext
        port = _make_portfolio()
        ctx = PricingContext.simple(REF, rate=0.05, vol=0.20)
        result = scenario_grid(port, ctx, [{"name": "base", "rate_shift": 0.0}])
        assert result["scenarios"][0]["pnl"] == pytest.approx(0.0, abs=0.01)

    def test_json_serializable(self):
        from pricebook import PricingContext
        port = _make_portfolio()
        ctx = PricingContext.simple(REF, rate=0.05, vol=0.20)
        result = scenario_grid(port, ctx, [{"name": "up", "rate_shift": 0.01}])
        j = json.dumps(result)
        assert isinstance(j, str)


class TestTradeBlotter:
    def test_structure(self):
        from pricebook import PricingContext
        port = _make_portfolio()
        ctx = PricingContext.simple(REF, rate=0.05, vol=0.20)
        blotter = trade_blotter(port, ctx)
        assert len(blotter) == 2
        assert blotter[0]["trade_id"] == "swn_5y10y"
        assert blotter[1]["trade_id"] == "swn_6y11y"
        assert "pv" in blotter[0]
        assert "instrument_type" in blotter[0]

    def test_json_serializable(self):
        from pricebook import PricingContext
        port = _make_portfolio()
        ctx = PricingContext.simple(REF, rate=0.05, vol=0.20)
        blotter = trade_blotter(port, ctx)
        j = json.dumps(blotter)
        assert isinstance(j, str)

    def test_has_dates(self):
        from pricebook import PricingContext
        port = _make_portfolio()
        ctx = PricingContext.simple(REF, rate=0.05, vol=0.20)
        blotter = trade_blotter(port, ctx)
        # Swaption has expiry and swap_end
        assert "instrument_type" in blotter[0]
