"""Tests for vol desk, arbitrage scanner, and calibration pipeline."""

from __future__ import annotations

import math
from datetime import date

from dateutil.relativedelta import relativedelta

from pricebook.desks.vol_desk import (
    VolBook, VolPosition, vol_dashboard, VolDashboard,
    vol_stress_report, vol_correlation_monitor,
)
from pricebook.options.vol_arbitrage_scanner import (
    scan_surface, ArbitrageScanResult, enforce_no_arb,
)
from pricebook.options.vol_calibration import (
    calibrate_fx_surface, calibrate_equity_surface,
    calibrate_ir_surface, CalibratedVolSurface,
)
from pricebook.options.vol_surface import FlatVol


REF = date(2024, 7, 15)


# ── Vol book + dashboard ──

class TestVolBook:

    def test_add_positions(self):
        book = VolBook()
        book.add(VolPosition("fx", "OPT1", "EURUSD", REF + relativedelta(months=3),
                             vega=50_000, vanna=1_000, volga=500))
        book.add(VolPosition("equity", "OPT2", "SPX", REF + relativedelta(months=6),
                             vega=80_000, vanna=2_000))
        assert len(book) == 2

    def test_total_vega(self):
        book = VolBook()
        book.add(VolPosition("fx", "O1", "EURUSD", REF + relativedelta(months=3), vega=50_000))
        book.add(VolPosition("equity", "O2", "SPX", REF + relativedelta(months=6), vega=-30_000))
        assert book.total_vega() == 20_000

    def test_vega_by_asset_class(self):
        book = VolBook()
        book.add(VolPosition("fx", "O1", "EURUSD", REF + relativedelta(months=3), vega=50_000))
        book.add(VolPosition("equity", "O2", "SPX", REF + relativedelta(months=6), vega=80_000))
        by_ac = book.vega_by_asset_class()
        assert "fx" in by_ac
        assert "equity" in by_ac
        assert by_ac["fx"] == 50_000

    def test_vega_ladder(self):
        book = VolBook()
        book.add(VolPosition("fx", "O1", "EURUSD", REF + relativedelta(months=1), vega=10_000))
        book.add(VolPosition("fx", "O2", "EURUSD", REF + relativedelta(years=2), vega=30_000))
        ladder = book.vega_ladder(REF)
        assert len(ladder) >= 2

    def test_dashboard(self):
        book = VolBook()
        book.add(VolPosition("fx", "O1", "EURUSD", REF + relativedelta(months=3),
                             vega=50_000, implied_vol=0.10, realised_vol=0.08))
        db = vol_dashboard(book, REF)
        assert db.n_positions == 1
        assert db.total_vega == 50_000
        assert "EURUSD" in db.vol_premium

    def test_dashboard_to_dict(self):
        book = VolBook()
        book.add(VolPosition("fx", "O1", "EURUSD", REF + relativedelta(months=3), vega=50_000))
        d = vol_dashboard(book, REF).to_dict()
        assert "vega_by_asset" in d
        assert "vega_ladder" in d


# ── Stress ──

class TestVolStress:

    def test_five_scenarios(self):
        book = VolBook()
        book.add(VolPosition("fx", "O1", "EURUSD", REF + relativedelta(months=3), vega=50_000))
        results = vol_stress_report(book, REF)
        assert len(results) == 5

    def test_vol_up_positive_for_long_vega(self):
        book = VolBook()
        book.add(VolPosition("fx", "O1", "EURUSD", REF + relativedelta(months=3), vega=50_000))
        results = vol_stress_report(book, REF)
        up = [r for r in results if r.scenario == "vol_up_2"][0]
        assert up.pnl > 0


# ── Arbitrage scanner ──

class TestArbitrageScanner:

    def test_flat_vol_no_arb(self):
        surface = FlatVol(0.20)
        strikes = [90, 95, 100, 105, 110]
        expiries = [REF + relativedelta(months=m) for m in [1, 3, 6, 12]]
        result = scan_surface("test", surface, strikes, expiries, "equity", REF)
        assert result.is_clean

    def test_scan_result_to_dict(self):
        surface = FlatVol(0.20)
        result = scan_surface("test", surface, [90, 100, 110],
                              [REF + relativedelta(months=3)], "fx", REF)
        d = result.to_dict()
        assert "is_clean" in d
        assert "calendar" in d

    def test_enforce_no_arb(self):
        surface = FlatVol(0.20)
        strikes = [90, 95, 100, 105, 110]
        expiries = [REF + relativedelta(months=m) for m in [1, 3, 6]]
        adjusted, n_fixes = enforce_no_arb(surface, strikes, expiries, REF)
        assert isinstance(adjusted, dict)


# ── Calibration pipeline ──

class TestCalibration:

    def test_fx_calibration(self):
        quotes = [
            {"expiry": REF + relativedelta(months=1), "atm": 0.08, "rr25": -0.01, "bf25": 0.003},
            {"expiry": REF + relativedelta(months=3), "atm": 0.09, "rr25": -0.012, "bf25": 0.004},
            {"expiry": REF + relativedelta(months=6), "atm": 0.095, "rr25": -0.015, "bf25": 0.005},
            {"expiry": REF + relativedelta(years=1), "atm": 0.10, "rr25": -0.018, "bf25": 0.006},
        ]
        surface = calibrate_fx_surface(REF, quotes, spot=1.08)
        assert len(surface.expiries) == 4
        vol = surface.vol(REF + relativedelta(months=3))
        assert 0.05 < vol < 0.20

    def test_equity_calibration(self):
        quotes = [
            {"expiry": REF + relativedelta(months=3),
             "strikes": [4800, 5000, 5200, 5400, 5600],
             "vols": [0.22, 0.19, 0.17, 0.16, 0.165]},
            {"expiry": REF + relativedelta(months=6),
             "strikes": [4800, 5000, 5200, 5400, 5600],
             "vols": [0.23, 0.20, 0.18, 0.17, 0.175]},
        ]
        surface = calibrate_equity_surface(REF, quotes, spot=5400)
        vol = surface.vol(REF + relativedelta(months=3), 5400)
        assert 0.10 < vol < 0.30

    def test_ir_calibration(self):
        quotes = [
            {"expiry": REF + relativedelta(years=1), "tenor": "5Y", "atm": 0.50},
            {"expiry": REF + relativedelta(years=2), "tenor": "5Y", "atm": 0.55},
            {"expiry": REF + relativedelta(years=5), "tenor": "5Y", "atm": 0.45},
        ]
        surface = calibrate_ir_surface(REF, quotes)
        vol = surface.vol(REF + relativedelta(years=1))
        assert vol == 0.50  # ATM only → exact match

    def test_surface_to_dict(self):
        quotes = [{"expiry": REF + relativedelta(months=3), "atm": 0.09, "rr25": -0.01, "bf25": 0.003}]
        surface = calibrate_fx_surface(REF, quotes, spot=1.08)
        d = surface.to_dict()
        assert "atm_vols" in d
        assert "n_tenors" in d

    def test_surface_bumped(self):
        quotes = [{"expiry": REF + relativedelta(months=3), "atm": 0.10, "rr25": 0, "bf25": 0}]
        surface = calibrate_fx_surface(REF, quotes, spot=1.08)
        bumped = surface.bumped(0.02)
        assert bumped.vol(REF + relativedelta(months=3)) > surface.vol(REF + relativedelta(months=3))

    def test_arb_report(self):
        quotes = [
            {"expiry": REF + relativedelta(months=3), "atm": 0.10, "rr25": 0, "bf25": 0},
            {"expiry": REF + relativedelta(months=6), "atm": 0.11, "rr25": 0, "bf25": 0},
        ]
        surface = calibrate_fx_surface(REF, quotes, spot=1.08)
        report = surface.arb_report()
        assert hasattr(report, 'is_clean')


# ── Correlation monitor ──

class TestCorrelationMonitor:

    def test_monitor(self):
        book = VolBook()
        book.add(VolPosition("fx", "O1", "EURUSD", REF + relativedelta(months=3),
                             vega=50_000, implied_vol=0.10, realised_vol=0.08))
        book.add(VolPosition("equity", "O2", "SPX", REF + relativedelta(months=6),
                             vega=80_000, implied_vol=0.18, realised_vol=0.20))
        monitor = vol_correlation_monitor(book)
        assert "EURUSD" in monitor
        assert monitor["EURUSD"]["signal"] == "rich"  # implied > realised
        assert monitor["SPX"]["signal"] == "cheap"   # implied < realised
