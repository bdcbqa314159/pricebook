"""Tests for CSA and funding framework."""

import pytest
from datetime import date

from pricebook.csa import (
    CSA,
    CollateralType,
    MarginFrequency,
    FundingModel,
    required_collateral,
    uncollateralised_exposure,
    collateral_adjusted_pv,
    funding_benefit_analysis,
)
from pricebook.pricing_context import PricingContext
from pricebook.vol_surface import FlatVol
from pricebook.swaption import Swaption
from pricebook.trade import Trade
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


def _make_ctx():
    curve = make_flat_curve(REF, 0.05)
    return PricingContext(
        valuation_date=REF,
        discount_curve=curve,
        vol_surfaces={"ir": FlatVol(0.20)},
    )


class FakeInstrument:
    def pv_ctx(self, ctx):
        return 1_000_000.0


class NegativeInstrument:
    def pv_ctx(self, ctx):
        return -500_000.0


# ---------------------------------------------------------------------------
# CSA terms
# ---------------------------------------------------------------------------


class TestCSA:
    def test_fully_collateralised(self):
        csa = CSA(threshold=0, mta=0)
        assert csa.is_fully_collateralised

    def test_not_fully_collateralised(self):
        csa = CSA(threshold=1_000_000)
        assert not csa.is_fully_collateralised

    def test_margin_period_days(self):
        assert CSA(margin_frequency=MarginFrequency.DAILY).margin_period_days == 1
        assert CSA(margin_frequency=MarginFrequency.WEEKLY).margin_period_days == 7
        assert CSA(margin_frequency=MarginFrequency.MONTHLY).margin_period_days == 30


class TestRequiredCollateral:
    def test_zero_threshold(self):
        csa = CSA(threshold=0, mta=0, rounding=1)
        assert required_collateral(1_000_000, csa) == 1_000_000

    def test_threshold(self):
        csa = CSA(threshold=500_000, mta=0, rounding=1)
        assert required_collateral(1_000_000, csa) == 500_000

    def test_below_threshold(self):
        csa = CSA(threshold=2_000_000)
        assert required_collateral(1_000_000, csa) == 0.0

    def test_mta(self):
        csa = CSA(threshold=0, mta=100_000, rounding=1)
        assert required_collateral(50_000, csa) == 0.0  # below MTA
        assert required_collateral(200_000, csa) == 200_000

    def test_rounding(self):
        csa = CSA(threshold=0, mta=0, rounding=10_000)
        # 1,234,567 rounds down to 1,230,000
        assert required_collateral(1_234_567, csa) == 1_230_000

    def test_initial_margin(self):
        csa = CSA(threshold=0, mta=0, rounding=1, initial_margin=100_000)
        coll = required_collateral(500_000, csa)
        assert coll == 600_000  # exposure + IM

    def test_negative_exposure(self):
        csa = CSA(threshold=0)
        assert required_collateral(-100_000, csa) == 0.0

    def test_zero_exposure(self):
        csa = CSA()
        assert required_collateral(0, csa) == 0.0


class TestUncollateralisedExposure:
    def test_fully_collateralised(self):
        csa = CSA(threshold=0, mta=0, rounding=1)
        assert uncollateralised_exposure(1_000_000, csa) == 0.0

    def test_with_threshold(self):
        csa = CSA(threshold=500_000, mta=0, rounding=1)
        uncoll = uncollateralised_exposure(1_000_000, csa)
        assert uncoll == 500_000  # threshold portion uncollateralised

    def test_below_threshold(self):
        csa = CSA(threshold=2_000_000)
        assert uncollateralised_exposure(1_000_000, csa) == 1_000_000


# ---------------------------------------------------------------------------
# Funding model
# ---------------------------------------------------------------------------


class TestFundingModel:
    def test_spread(self):
        fm = FundingModel(secured_rate=0.05, unsecured_rate=0.055)
        assert fm.funding_spread == pytest.approx(0.005)

    def test_zero_spread(self):
        fm = FundingModel(secured_rate=0.05, unsecured_rate=0.05)
        assert fm.funding_spread == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Generic PV adjustment — works with any trade
# ---------------------------------------------------------------------------


class TestCollateralAdjustedPV:
    def test_fully_collateralised_no_funding_cost(self):
        ctx = _make_ctx()
        csa = CSA(threshold=0, mta=0, rounding=1)
        funding = FundingModel(secured_rate=0.05, unsecured_rate=0.055)
        inst = FakeInstrument()

        result = collateral_adjusted_pv(inst, ctx, csa, funding)
        assert result["base_pv"] == pytest.approx(1_000_000)
        assert result["uncollateralised"] == pytest.approx(0.0)
        assert result["funding_cost"] == pytest.approx(0.0)

    def test_uncollateralised_has_funding_cost(self):
        ctx = _make_ctx()
        csa = CSA(threshold=2_000_000)  # no collateral required
        funding = FundingModel(secured_rate=0.05, unsecured_rate=0.055)
        inst = FakeInstrument()

        result = collateral_adjusted_pv(inst, ctx, csa, funding)
        assert result["uncollateralised"] == pytest.approx(1_000_000)
        assert result["funding_cost"] > 0

    def test_adjusted_pv_less_than_base(self):
        """Funding cost reduces PV."""
        ctx = _make_ctx()
        csa = CSA(threshold=2_000_000)
        funding = FundingModel(secured_rate=0.05, unsecured_rate=0.06)
        inst = FakeInstrument()

        result = collateral_adjusted_pv(inst, ctx, csa, funding)
        assert result["adjusted_pv"] < result["base_pv"]

    def test_with_trade_wrapper(self):
        """Works with Trade objects too."""
        ctx = _make_ctx()
        csa = CSA(threshold=0, mta=0, rounding=1)
        funding = FundingModel()

        swn = Swaption(date(2025, 1, 15), date(2030, 1, 15), strike=0.05)
        trade = Trade(swn, direction=1, notional_scale=1.0)

        result = collateral_adjusted_pv(trade, ctx, csa, funding)
        assert result["base_pv"] > 0  # swaption has positive value

    def test_with_raw_instrument(self):
        """Works with raw instruments that have pv_ctx."""
        ctx = _make_ctx()
        csa = CSA()
        funding = FundingModel()

        swn = Swaption(date(2025, 1, 15), date(2030, 1, 15), strike=0.05)
        result = collateral_adjusted_pv(swn, ctx, csa, funding)
        assert result["base_pv"] > 0

    def test_negative_pv(self):
        """Negative PV → we post collateral."""
        ctx = _make_ctx()
        csa = CSA(threshold=0, mta=0, rounding=1)
        funding = FundingModel(secured_rate=0.05, unsecured_rate=0.06)
        inst = NegativeInstrument()

        result = collateral_adjusted_pv(inst, ctx, csa, funding)
        assert result["base_pv"] < 0
        assert result["collateral"] == 0.0  # negative exposure → no collateral received

    def test_funding_cost_scales_with_horizon(self):
        ctx = _make_ctx()
        csa = CSA(threshold=2_000_000)
        funding = FundingModel(secured_rate=0.05, unsecured_rate=0.06)
        inst = FakeInstrument()

        r1 = collateral_adjusted_pv(inst, ctx, csa, funding, horizon=1.0)
        r2 = collateral_adjusted_pv(inst, ctx, csa, funding, horizon=2.0)
        assert r2["funding_cost"] == pytest.approx(2.0 * r1["funding_cost"])

    def test_zero_spread_no_cost(self):
        ctx = _make_ctx()
        csa = CSA(threshold=2_000_000)
        funding = FundingModel(secured_rate=0.05, unsecured_rate=0.05)
        inst = FakeInstrument()

        result = collateral_adjusted_pv(inst, ctx, csa, funding)
        assert result["funding_cost"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Comparative analysis
# ---------------------------------------------------------------------------


class TestFundingBenefitAnalysis:
    def test_compare_csa_options(self):
        ctx = _make_ctx()
        funding = FundingModel(secured_rate=0.05, unsecured_rate=0.06)
        inst = FakeInstrument()

        options = [
            ("uncollateralised", CSA(threshold=10_000_000)),
            ("partial", CSA(threshold=500_000)),
            ("fully_collateralised", CSA(threshold=0, mta=0, rounding=1)),
        ]

        results = funding_benefit_analysis(inst, ctx, options, funding)
        assert len(results) == 3
        assert results[0]["name"] == "uncollateralised"
        assert results[2]["name"] == "fully_collateralised"

        # Tighter collateral → lower funding cost
        assert results[0]["funding_cost"] >= results[1]["funding_cost"]
        assert results[1]["funding_cost"] >= results[2]["funding_cost"]

    def test_all_results_have_name(self):
        ctx = _make_ctx()
        funding = FundingModel()
        inst = FakeInstrument()

        results = funding_benefit_analysis(
            inst, ctx, [("a", CSA()), ("b", CSA(threshold=100))], funding
        )
        assert all("name" in r for r in results)
