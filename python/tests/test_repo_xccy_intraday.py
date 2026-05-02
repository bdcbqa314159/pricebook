"""Tests for cross-currency repo and intraday position keeping."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.repo_desk import RepoTrade
from pricebook.serialisable import from_dict
from pricebook.serialization import to_json, from_json
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)


# ── Cross-currency repo ──

class TestCrossCurrencyRepo:

    def test_single_currency_default(self):
        """Default is same currency — no FX impact."""
        t = RepoTrade(
            counterparty="JPM", collateral_issuer="UST10Y",
            face_amount=50_000_000, bond_price=102.0, repo_rate=0.045,
            term_days=30, direction="repo", start_date=REF,
        )
        assert not t.is_cross_currency
        assert t.fx_rate == 1.0
        assert t.fx_haircut == 0.0

    def test_eur_bond_usd_cash(self):
        """EUR Bund, USD cash. FX rate = 1.08 (EUR/USD)."""
        t = RepoTrade(
            counterparty="DB", collateral_issuer="DBR_2.5_2034",
            face_amount=50_000_000, bond_price=105.0,
            repo_rate=0.035, term_days=30, coupon_rate=0.025,
            direction="repo", start_date=REF,
            haircut=0.02, fx_haircut=0.05,  # extra 5% for FX risk
            bond_currency="EUR", cash_currency="USD",
            fx_rate=1.08,  # 1 EUR = 1.08 USD
        )
        assert t.is_cross_currency
        assert t.bond_currency == "EUR"
        assert t.cash_currency == "USD"

        # Market value in EUR
        mv_eur = 50_000_000 * 105.0 / 100.0
        assert t.market_value == pytest.approx(mv_eur)

        # Cash in USD: EUR value × FX rate × (1 - haircut - fx_haircut)
        expected_cash = mv_eur * 1.08 * (1 - 0.02 - 0.05)
        assert t.cash_amount == pytest.approx(expected_cash)

    def test_jpy_bond_usd_cash(self):
        """JGB in JPY, cash in USD."""
        t = RepoTrade(
            counterparty="Nomura", collateral_issuer="JGB_0.5_2034",
            face_amount=5_000_000_000, bond_price=98.0,  # JPY
            repo_rate=0.001, term_days=30, coupon_rate=0.005,
            direction="repo", start_date=REF,
            haircut=0.03, fx_haircut=0.08,  # higher FX haircut for JPY
            bond_currency="JPY", cash_currency="USD",
            fx_rate=0.0064,  # 1 JPY = 0.0064 USD
        )
        assert t.is_cross_currency
        mv_jpy = 5_000_000_000 * 98.0 / 100.0
        cash_usd = mv_jpy * 0.0064 * (1 - 0.03 - 0.08)
        assert t.cash_amount == pytest.approx(cash_usd)

    def test_xccy_margin_call(self):
        """FX move + price move → combined margin call."""
        t = RepoTrade(
            counterparty="DB", collateral_issuer="DBR",
            face_amount=50_000_000, bond_price=105.0,
            repo_rate=0.035, term_days=30, direction="repo",
            start_date=REF, haircut=0.02, fx_haircut=0.05,
            bond_currency="EUR", cash_currency="USD", fx_rate=1.08,
        )
        # EUR weakens: 1.08 → 1.05, bond drops 105 → 103
        mc = t.xccy_margin_call(current_bond_price=103.0, current_fx_rate=1.05)
        assert math.isfinite(mc)

    def test_xccy_serialisation(self):
        t = RepoTrade(
            counterparty="DB", collateral_issuer="DBR",
            face_amount=50_000_000, bond_price=105.0,
            repo_rate=0.035, term_days=30, direction="repo",
            start_date=REF, bond_currency="EUR", cash_currency="USD",
            fx_rate=1.08, fx_haircut=0.05,
        )
        j = to_json(t)
        t2 = from_json(j)
        assert t2.bond_currency == "EUR"
        assert t2.cash_currency == "USD"
        assert t2.fx_rate == 1.08
        assert t2.fx_haircut == 0.05
        assert t2.cash_amount == pytest.approx(t.cash_amount)

    def test_gbp_gilt_eur_cash(self):
        """Any pair works: GBP Gilt, EUR cash."""
        t = RepoTrade(
            counterparty="Barclays", collateral_issuer="UKT_4.0_2034",
            face_amount=30_000_000, bond_price=101.0,
            repo_rate=0.04, term_days=30, coupon_rate=0.04,
            direction="repo", start_date=REF,
            haircut=0.02, fx_haircut=0.04,
            bond_currency="GBP", cash_currency="EUR", fx_rate=1.17,
        )
        assert t.is_cross_currency
        mv_gbp = 30_000_000 * 101.0 / 100.0
        cash_eur = mv_gbp * 1.17 * (1 - 0.02 - 0.04)
        assert t.cash_amount == pytest.approx(cash_eur)


# ── Intraday position keeping ──

class TestIntradaySnapshot:

    def test_snapshot_at_inception(self):
        t = RepoTrade(
            counterparty="JPM", collateral_issuer="UST10Y",
            face_amount=50_000_000, bond_price=102.0,
            repo_rate=0.045, term_days=30, direction="repo",
            start_date=REF, settlement_days=0, trade_id="T001",
        )
        snap = t.snapshot(REF)
        assert snap["trade_id"] == "T001"
        assert snap["remaining_days"] == 30
        assert snap["accrued_interest"] == pytest.approx(0.0)
        assert snap["mark_to_market"] == pytest.approx(0.0)
        assert snap["variation_margin"] == pytest.approx(0.0)

    def test_snapshot_midway_unchanged(self):
        """Day 15, nothing moved → only accrued changes."""
        t = RepoTrade(
            counterparty="JPM", collateral_issuer="UST10Y",
            face_amount=50_000_000, bond_price=102.0,
            repo_rate=0.045, term_days=30, direction="repo",
            start_date=REF, settlement_days=0, trade_id="T001",
        )
        snap = t.snapshot(REF + timedelta(days=15))
        assert snap["remaining_days"] == 15
        assert snap["accrued_interest"] > 0
        assert snap["mark_to_market"] == pytest.approx(0.0)  # same rate
        assert snap["variation_margin"] == pytest.approx(0.0)  # same price

    def test_snapshot_with_moves(self):
        """Day 15, bond drops 1pt, repo rate up 50bp."""
        t = RepoTrade(
            counterparty="JPM", collateral_issuer="UST10Y",
            face_amount=50_000_000, bond_price=102.0,
            repo_rate=0.045, term_days=30, direction="repo",
            start_date=REF, settlement_days=0, trade_id="T001",
        )
        snap = t.snapshot(
            REF + timedelta(days=15),
            current_bond_price=101.0,
            current_repo_rate=0.050,
        )
        assert snap["mark_to_market"] != 0  # rate moved
        assert snap["variation_margin"] != 0  # price moved
        assert math.isfinite(snap["total_unrealised"])

    def test_snapshot_xccy_fx_pnl(self):
        """Cross-currency: FX moves → FX P&L in snapshot."""
        t = RepoTrade(
            counterparty="DB", collateral_issuer="DBR",
            face_amount=50_000_000, bond_price=105.0,
            repo_rate=0.035, term_days=30, direction="repo",
            start_date=REF, settlement_days=0,
            bond_currency="EUR", cash_currency="USD",
            fx_rate=1.08,
        )
        snap = t.snapshot(REF + timedelta(days=10), current_fx_rate=1.05)
        assert snap["fx_pnl"] != 0  # FX moved

    def test_multiple_snapshots_timeline(self):
        """Snapshots at different times show progression."""
        t = RepoTrade(
            counterparty="JPM", collateral_issuer="UST10Y",
            face_amount=50_000_000, bond_price=102.0,
            repo_rate=0.045, term_days=30, direction="repo",
            start_date=REF, settlement_days=0,
        )
        snaps = [t.snapshot(REF + timedelta(days=d)) for d in [0, 7, 15, 30]]
        # Accrued should increase monotonically
        accruals = [s["accrued_interest"] for s in snaps]
        for i in range(1, len(accruals)):
            assert accruals[i] >= accruals[i-1]
        # Remaining should decrease
        remaining = [s["remaining_days"] for s in snaps]
        for i in range(1, len(remaining)):
            assert remaining[i] <= remaining[i-1]


# ── Curves needed for xccy pricing ──

class TestXccyCurves:

    def test_pv_with_usd_curve_on_eur_repo(self):
        """EUR bond repo priced on USD OIS curve (cash currency curve)."""
        t = RepoTrade(
            counterparty="DB", collateral_issuer="DBR",
            face_amount=50_000_000, bond_price=105.0,
            repo_rate=0.035, term_days=30, direction="repo",
            start_date=REF, settlement_days=0,
            bond_currency="EUR", cash_currency="USD", fx_rate=1.08,
        )
        # Discount on USD curve (cash currency)
        usd_curve = make_flat_curve(REF, 0.05)
        pv = t.pv(usd_curve, REF)
        assert math.isfinite(pv)

        # Different curves give different PV
        eur_curve = make_flat_curve(REF, 0.03)
        pv_eur = t.pv(eur_curve, REF)
        assert pv != pytest.approx(pv_eur, rel=0.01)
