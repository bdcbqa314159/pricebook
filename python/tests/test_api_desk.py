"""Tests for trader API (api_desk.py)."""

from __future__ import annotations

import math
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.api_desk import (
    analyse, cln, trs, repo,
    vol_surface, swap_book, cds_book,
    multicurve, recovery_analysis, dashboard,
)
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)


def _curve():
    return make_flat_curve(REF, 0.04)


# ── analyse() ──

class TestAnalyse:

    def test_irs(self):
        result = analyse("irs", curve=_curve(), tenor="5Y", rate=0.04, notional=50_000_000)
        assert "pv" in result
        assert "par_rate" in result
        assert "dv01" in result
        assert "carry" in result

    def test_cds(self):
        result = analyse("cds", curve=_curve(), tenor="5Y", spread=0.01, hazard=0.02, notional=10_000_000)
        assert "pv" in result
        assert "cs01" in result
        assert "jtd" in result
        assert "par_spread" in result

    def test_cln(self):
        result = analyse("cln", curve=_curve(), tenor="5Y", coupon=0.05, hazard=0.02, notional=10_000_000)
        assert "pv" in result
        assert "cs01" in result
        assert "jtd" in result

    def test_bond(self):
        result = analyse("bond", curve=_curve(), tenor="10Y", coupon=0.04, repo_rate=0.04)
        assert "pv" in result
        assert "ytm" in result
        assert "mod_duration" in result
        assert "key_rate_dv01" in result

    def test_amortising_irs_with_schedule(self):
        """Explicit per-period notional schedule."""
        schedule = [50e6, 40e6, 30e6, 20e6, 10e6]
        result = analyse("irs", curve=_curve(), tenor="5Y", rate=0.04,
                         notional=schedule)
        assert result["type"] == "amortising_irs"
        assert len(result["notional_schedule"]) >= 5
        assert result["average_notional"] > 0
        assert result["weighted_average_life"] > 0
        assert math.isfinite(result["pv"])
        assert math.isfinite(result["dv01"])

    def test_amortising_irs_with_profile(self):
        """Profile-driven amortisation."""
        result = analyse("irs", curve=_curve(), tenor="5Y", rate=0.04,
                         notional=50_000_000, notional_profile="amortising")
        assert result["type"] == "amortising_irs"
        # Notional should decrease
        sched = result["notional_schedule"]
        assert sched[0] > sched[-1]
        assert result["n_periods"] > 1

    def test_accreting_irs(self):
        """Accreting notional profile."""
        result = analyse("irs", curve=_curve(), tenor="5Y", rate=0.04,
                         notional=10_000_000, notional_profile="accreting",
                         final_notional=50_000_000)
        assert result["type"] == "amortising_irs"
        sched = result["notional_schedule"]
        assert sched[-1] > sched[0]

    def test_unknown_raises(self):
        import pytest
        with pytest.raises(ValueError):
            analyse("unknown", curve=_curve())


# ── One-liners ──

class TestOneLiners:

    def test_cln_oneliner(self):
        result = cln("5Y", 0.05, _curve(), notional=10_000_000)
        assert result["type"] == "cln"
        assert math.isfinite(result["pv"])

    def test_trs_oneliner(self):
        result = trs("6M", 100.0, _curve(), notional=10_000_000, sigma=0.20)
        assert result["type"] == "trs"
        assert "delta" in result
        assert "carry" in result

    def test_repo_oneliner(self):
        result = repo(30, 10_000_000, 0.04)
        assert result["type"] == "repo"
        assert result["cash_lent"] == 10_000_000 * 0.95
        assert result["interest"] > 0


# ── Vol surface ──

class TestVolSurface:

    def test_fx_surface(self):
        surface = vol_surface("fx", [
            {"expiry": "1M", "atm": 0.08, "rr25": -0.01, "bf25": 0.003},
            {"expiry": "3M", "atm": 0.09, "rr25": -0.012, "bf25": 0.004},
        ], spot=1.08, ref=REF)
        assert surface is not None
        assert len(surface.expiries) == 2

    def test_equity_surface(self):
        surface = vol_surface("equity", [
            {"expiry": "3M", "strikes": [90, 95, 100, 105, 110],
             "vols": [0.22, 0.20, 0.18, 0.17, 0.165]},
        ], spot=100, ref=REF)
        assert surface is not None


# ── Books ──

class TestBooks:

    def test_swap_book(self):
        result = swap_book([
            {"tenor": "5Y", "rate": 0.038, "direction": "payer", "notional": 50_000_000},
            {"tenor": "2Y", "rate": 0.039, "direction": "receiver", "notional": 20_000_000},
        ], curve=_curve())
        assert result["n_positions"] == 2
        assert "total_dv01" in result
        assert "net_dv01" in result
        assert "stress" in result
        assert len(result["stress"]) >= 3

    def test_cds_book(self):
        result = cds_book([
            {"name": "AAPL", "tenor": "5Y", "spread": 0.005, "sector": "tech", "notional": 10_000_000},
            {"name": "JPM", "tenor": "5Y", "spread": 0.008, "sector": "financials", "notional": 10_000_000},
        ], curve=_curve())
        assert result["n_positions"] == 2
        assert "total_cs01" in result
        assert "by_sector" in result


# ── Multi-curve ──

class TestMulticurve:

    def test_build_usd_eur(self):
        curves = multicurve(
            ref=REF,
            usd={"swaps": {"1Y": 0.047, "5Y": 0.038, "10Y": 0.036}},
            eur={"swaps": {"1Y": 0.034, "5Y": 0.028, "10Y": 0.026}},
        )
        assert "USD" in curves
        assert "EUR" in curves
        assert curves["USD"].df(REF + relativedelta(years=5)) < 1.0


# ── Recovery ──

class TestRecovery:

    def test_recovery_analysis(self):
        result = recovery_analysis(
            cds_spreads={1: 0.005, 5: 0.01, 10: 0.012},
            curve=_curve(), tenor="5Y", coupon=0.05,
        )
        assert "greeks" in result
        assert "surface" in result
        assert result["direct_effect"] > 0
        assert result["indirect_effect"] < 0


# ── Dashboard ──

class TestDashboard:

    def test_swap_dashboard(self):
        result = dashboard("swap", [
            {"tenor": "5Y", "rate": 0.038, "direction": "payer", "notional": 50_000_000},
        ], curve=_curve())
        assert "total_dv01" in result
        assert "stress" in result

    def test_cds_dashboard(self):
        result = dashboard("cds", [
            {"name": "MSFT", "tenor": "5Y", "spread": 0.006, "sector": "tech", "notional": 25_000_000},
        ], curve=_curve())
        assert "total_cs01" in result
        assert "by_sector" in result
