"""Tests for equity, commodity, and inflation conventions."""
import pytest
from pricebook.market_conventions import (
    get_equity_index, get_commodity_contract, get_linker_convention,
    index_ratio, EQUITY_INDICES, COMMODITY_CONTRACTS, LME_METALS, LINKER_CONVENTIONS,
)


# ---- Equity ----

class TestEquityIndices:
    def test_spx(self):
        spec = get_equity_index("SPX")
        assert spec.currency == "USD"
        assert spec.option_style == "european"
        assert spec.option_multiplier == 100.0
        assert spec.settlement_lag == 2

    def test_sx5e(self):
        spec = get_equity_index("SX5E")
        assert spec.currency == "EUR"
        assert spec.exchange == "Eurex"

    def test_nky(self):
        spec = get_equity_index("NKY")
        assert spec.currency == "JPY"
        assert spec.option_multiplier == 1000.0

    def test_ukx(self):
        spec = get_equity_index("UKX")
        assert spec.currency == "GBP"

    def test_all_indices_have_currency(self):
        for ticker, spec in EQUITY_INDICES.items():
            assert spec.currency, f"{ticker} missing currency"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_equity_index("NONEXISTENT")


# ---- Commodity ----

class TestCommodityContracts:
    def test_wti(self):
        spec = get_commodity_contract("CL")
        assert spec.name == "WTI Crude Oil"
        assert spec.contract_size == 1000
        assert spec.unit == "barrels"
        assert spec.settlement_type == "physical"

    def test_brent_cash_settled(self):
        spec = get_commodity_contract("BRN")
        assert spec.settlement_type == "cash"

    def test_gold(self):
        spec = get_commodity_contract("GC")
        assert spec.contract_size == 100
        assert spec.unit == "troy oz"

    def test_nat_gas(self):
        spec = get_commodity_contract("NG")
        assert spec.contract_size == 10000

    def test_lme_copper(self):
        spec = get_commodity_contract("LCU")
        assert spec.exchange == "LME"
        assert spec.contract_months == "prompt"  # LME prompt date system

    def test_corn(self):
        spec = get_commodity_contract("ZC")
        assert spec.unit == "bushels"

    def test_all_have_tick(self):
        for sym, spec in {**COMMODITY_CONTRACTS, **LME_METALS}.items():
            assert spec.tick_value > 0, f"{sym} missing tick_value"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_commodity_contract("NONEXISTENT")


# ---- Inflation linkers ----

class TestLinkerConventions:
    def test_us_tips(self):
        conv = get_linker_convention("US")
        assert conv.index_name == "CPI-U (All Urban)"
        assert conv.lag_months == 3
        assert conv.deflation_floor is True
        assert conv.coupon_frequency == "semi-annual"

    def test_uk_ilg(self):
        conv = get_linker_convention("UK")
        assert conv.index_name == "RPI"
        assert conv.lag_months == 8
        assert conv.deflation_floor is False

    def test_france_oati(self):
        conv = get_linker_convention("FR")
        assert conv.index_name == "HICP ex-tobacco"
        assert conv.coupon_frequency == "annual"

    def test_italy_btpei(self):
        conv = get_linker_convention("IT")
        assert conv.index_name == "HICP ex-tobacco"

    def test_australia_lag(self):
        conv = get_linker_convention("AU")
        assert conv.lag_months == 6  # AUS has quarterly CPI with 6M lag
        assert conv.index_ratio_method == "quarterly"

    def test_japan_deflation_floor(self):
        conv = get_linker_convention("JP")
        assert conv.deflation_floor is True

    def test_all_countries(self):
        for country in ["US", "UK", "FR", "IT", "DE", "CA", "AU", "JP"]:
            conv = get_linker_convention(country)
            assert conv.lag_months > 0

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_linker_convention("ZZ")


class TestIndexRatio:
    def test_simple_ratio(self):
        """No daily interpolation: just ref/base."""
        assert index_ratio(250, 260) == pytest.approx(1.04)

    def test_daily_linear(self):
        """Daily interpolation: CPI_start + (d-1)/D × (CPI_end − CPI_start)."""
        # Day 15 of 30-day month, CPI from 260 to 262
        ratio = index_ratio(250, 0, daily_cpi_start=260, daily_cpi_end=262,
                             day_of_month=15, days_in_month=30)
        # interp = 260 + 14/30 × 2 = 260.9333
        assert ratio == pytest.approx(260.9333 / 250, rel=1e-4)

    def test_day_1_uses_start(self):
        ratio = index_ratio(250, 0, daily_cpi_start=260, daily_cpi_end=262,
                             day_of_month=1, days_in_month=30)
        assert ratio == pytest.approx(260 / 250)

    def test_last_day_near_end(self):
        ratio = index_ratio(250, 0, daily_cpi_start=260, daily_cpi_end=262,
                             day_of_month=30, days_in_month=30)
        # interp = 260 + 29/30 × 2 ≈ 261.933
        assert ratio == pytest.approx(261.9333 / 250, rel=1e-3)
