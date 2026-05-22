"""Tests for repo specialness analytics and repo rate Greeks."""

import pytest
from datetime import date

from pricebook.fixed_income.repo_specialness import (
    forecast_specialness, specialness_term_structure, supply_demand_indicator,
    get_specialness_conventions, list_specialness_markets,
    SpecialnessForecast, SpecialnessConventions,
)
from pricebook.fixed_income.repo_greeks import (
    repo_dv01, carry_sensitivity_ladder, repo_portfolio_greeks,
    RepoGreeksResult, RepoPortfolioGreeks,
)


# ═══════════════════════════════════════════════════════════════
# 1.3: Specialness Analytics
# ═══════════════════════════════════════════════════════════════

class TestSpecialnessConventions:
    def test_ust(self):
        c = get_specialness_conventions("UST")
        assert "10Y" in c.on_the_run_tenors
        assert c.settlement_days == 1

    def test_bund(self):
        c = get_specialness_conventions("BUND")
        assert "10Y" in c.on_the_run_tenors

    def test_jgb(self):
        c = get_specialness_conventions("JGB")
        assert "40Y" in c.on_the_run_tenors

    def test_list_markets(self):
        markets = list_specialness_markets()
        assert len(markets) == 6
        assert "UST" in markets
        assert "BTP" in markets

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_specialness_conventions("FAKE")

    def test_to_dict(self):
        d = get_specialness_conventions("UST").to_dict()
        assert "market" in d


class TestSpecialnessForecast:
    def test_mean_reversion(self):
        """Very special bond should forecast lower (mean reversion)."""
        f = forecast_specialness("UST_10Y", "UST", current_special_bp=100,
                                  historical_mean_bp=20)
        assert f.forecast_special_bp < 100

    def test_cheap_special_forecast_up(self):
        """Below-average special should forecast up."""
        f = forecast_specialness("UST_10Y", "UST", current_special_bp=5,
                                  historical_mean_bp=20)
        assert f.forecast_special_bp > 5

    def test_pre_auction_squeeze(self):
        """Specialness should increase before auction."""
        f_no_auction = forecast_specialness("UST_10Y", "UST", 30, 20)
        f_pre_auction = forecast_specialness("UST_10Y", "UST", 30, 20,
                                              days_to_auction=2)
        assert f_pre_auction.forecast_special_bp > f_no_auction.forecast_special_bp

    def test_signal_rich(self):
        f = forecast_specialness("X", "UST", 80, 20, 15)
        assert f.signal == "RICH_SPECIAL"

    def test_signal_cheap(self):
        f = forecast_specialness("X", "UST", 5, 20, 15)
        assert f.signal == "CHEAP_SPECIAL"

    def test_to_dict(self):
        d = forecast_specialness("X", "UST", 30, 20).to_dict()
        assert "forecast_special_bp" in d
        assert "z_score" in d


class TestSpecialnessTermStructure:
    def test_basic(self):
        gc = {1: 0.053, 7: 0.052, 30: 0.051}
        special = {1: 0.050, 7: 0.048, 30: 0.047}
        ts = specialness_term_structure(gc, special)
        assert len(ts) == 3
        assert all(t["spread_bp"] > 0 for t in ts)


class TestSupplyDemand:
    def test_high_risk(self):
        r = supply_demand_indicator(5.0, True, 3, 20, 15)
        assert r["signal"] == "HIGH_SPECIAL_RISK"

    def test_low_risk(self):
        r = supply_demand_indicator(0.5, False, 30, 100, 0)
        assert r["signal"] == "LOW"


# ═══════════════════════════════════════════════════════════════
# 1.4: Repo Rate Greeks
# ═══════════════════════════════════════════════════════════════

class TestRepoDV01:
    def test_basic(self):
        g = repo_dv01(1e6, 0.05, 90)
        assert isinstance(g, RepoGreeksResult)
        assert g.repo_dv01 > 0

    def test_longer_tenor_higher_dv01(self):
        g30 = repo_dv01(1e6, 0.05, 30)
        g180 = repo_dv01(1e6, 0.05, 180)
        assert g180.repo_dv01 > g30.repo_dv01

    def test_larger_notional_higher_dv01(self):
        g1 = repo_dv01(1e6, 0.05, 90)
        g10 = repo_dv01(10e6, 0.05, 90)
        assert abs(g10.repo_dv01 / g1.repo_dv01 - 10) < 0.01

    def test_roll_theta_positive(self):
        g = repo_dv01(1e6, 0.05, 90)
        assert g.roll_theta > 0

    def test_to_dict(self):
        d = repo_dv01(1e6, 0.05, 90).to_dict()
        assert "repo_dv01" in d


class TestCarryLadder:
    def test_basic(self):
        trades = [
            {"notional": 1e6, "repo_rate": 0.05, "repo_days": 1},
            {"notional": 2e6, "repo_rate": 0.04, "repo_days": 30},
            {"notional": 3e6, "repo_rate": 0.045, "repo_days": 90},
        ]
        ladder = carry_sensitivity_ladder(trades)
        assert len(ladder) == 6  # default 6 buckets
        # O/N bucket should have the 1-day trade
        assert ladder[0].bucket == "O/N"
        assert ladder[0].dv01 > 0

    def test_pct_sums_to_100(self):
        trades = [
            {"notional": 1e6, "repo_rate": 0.05, "repo_days": 7},
            {"notional": 1e6, "repo_rate": 0.05, "repo_days": 90},
        ]
        ladder = carry_sensitivity_ladder(trades)
        total_pct = sum(b.pct_of_total for b in ladder)
        assert abs(total_pct - 100) < 0.1


class TestPortfolioGreeks:
    def test_basic(self):
        trades = [
            {"notional": 1e6, "repo_rate": 0.05, "repo_days": 30},
            {"notional": 2e6, "repo_rate": 0.04, "repo_days": 90},
        ]
        pg = repo_portfolio_greeks(trades)
        assert isinstance(pg, RepoPortfolioGreeks)
        assert pg.n_trades == 2
        assert pg.total_repo_dv01 > 0
        assert pg.weighted_avg_tenor_days > 30

    def test_to_dict(self):
        trades = [{"notional": 1e6, "repo_rate": 0.05, "repo_days": 30}]
        d = repo_portfolio_greeks(trades).to_dict()
        assert "total_repo_dv01" in d
