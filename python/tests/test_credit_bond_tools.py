"""Tests for credit bond tools."""

import pytest

from pricebook.credit_bond_tools import (
    ConcentrationResult,
    CrossSectorRV,
    MigrationImpact,
    SectorSpreadSignal,
    SectorWeight,
    TrackingErrorResult,
    concentration_risk,
    cross_sector_rv,
    index_tracking_error,
    rating_migration_impact,
    sector_allocation,
    sector_spread_monitor,
)


# ---- Step 1: IG/HY allocation ----

class TestSectorAllocation:
    def test_single_sector(self):
        positions = [("financials", 10_000_000)]
        result = sector_allocation(positions)
        assert len(result) == 1
        assert result[0].weight == pytest.approx(1.0)

    def test_multiple_sectors(self):
        positions = [
            ("financials", 10_000_000),
            ("industrials", 10_000_000),
        ]
        result = sector_allocation(positions)
        assert len(result) == 2
        assert result[0].weight == pytest.approx(0.5)
        assert result[1].weight == pytest.approx(0.5)

    def test_aggregates_within_sector(self):
        positions = [
            ("financials", 5_000_000),
            ("financials", 5_000_000),
            ("industrials", 10_000_000),
        ]
        result = sector_allocation(positions)
        fin = next(s for s in result if s.sector == "financials")
        assert fin.market_value == pytest.approx(10_000_000)
        assert fin.weight == pytest.approx(0.5)

    def test_empty(self):
        assert sector_allocation([]) == []


class TestIndexTrackingError:
    def test_zero_when_matching(self):
        """Step 1 test: tracking error is zero when book matches index."""
        portfolio = {"financials": 0.30, "industrials": 0.40, "utilities": 0.30}
        index = {"financials": 0.30, "industrials": 0.40, "utilities": 0.30}
        result = index_tracking_error(portfolio, index)
        assert result.tracking_error == pytest.approx(0.0)

    def test_nonzero_when_different(self):
        portfolio = {"financials": 0.40, "industrials": 0.40, "utilities": 0.20}
        index = {"financials": 0.30, "industrials": 0.40, "utilities": 0.30}
        result = index_tracking_error(portfolio, index)
        # |+0.10| + |0| + |-0.10| = 0.20
        assert result.tracking_error == pytest.approx(0.20)
        assert result.max_overweight == "financials"
        assert result.max_underweight == "utilities"

    def test_missing_sector_in_portfolio(self):
        portfolio = {"financials": 1.0}
        index = {"financials": 0.5, "industrials": 0.5}
        result = index_tracking_error(portfolio, index)
        assert result.active_weights["industrials"] == pytest.approx(-0.5)

    def test_missing_sector_in_index(self):
        portfolio = {"financials": 0.5, "EM": 0.5}
        index = {"financials": 1.0}
        result = index_tracking_error(portfolio, index)
        assert result.active_weights["EM"] == pytest.approx(0.5)


class TestConcentrationRisk:
    def test_single_name_hhi_one(self):
        result = concentration_risk(
            name_weights={"AAPL": 1.0},
            sector_weights={"tech": 1.0},
            rating_weights={"AA": 1.0},
        )
        assert result.herfindahl_name == pytest.approx(1.0)
        assert result.top_name_weight == pytest.approx(1.0)

    def test_diversified_low_hhi(self):
        names = {f"name_{i}": 0.01 for i in range(100)}
        sectors = {f"sector_{i}": 0.1 for i in range(10)}
        ratings = {"AAA": 0.2, "AA": 0.3, "A": 0.3, "BBB": 0.2}
        result = concentration_risk(names, sectors, ratings)
        # 100 names × 0.01² = 0.01
        assert result.herfindahl_name == pytest.approx(0.01)
        # 10 sectors × 0.1² = 0.10
        assert result.herfindahl_sector == pytest.approx(0.10)

    def test_empty_weights(self):
        result = concentration_risk({}, {}, {})
        assert result.herfindahl_name == 0.0
        assert result.top_name_weight == 0.0


# ---- Step 2: sector rotation ----

class TestSectorSpreadMonitor:
    def test_wide_signal(self):
        history = [100, 105, 110, 95, 100] * 4
        sig = sector_spread_monitor("financials", 200.0, history, threshold=2.0)
        assert sig.signal == "wide"
        assert sig.z_score is not None
        assert sig.z_score > 2.0

    def test_tight_signal(self):
        history = [100, 105, 110, 95, 100] * 4
        sig = sector_spread_monitor("financials", 50.0, history, threshold=2.0)
        assert sig.signal == "tight"

    def test_fair_signal(self):
        history = [100, 105, 110, 95, 100] * 4
        sig = sector_spread_monitor("financials", 102.0, history)
        assert sig.signal == "fair"

    def test_no_history(self):
        sig = sector_spread_monitor("financials", 100.0, [])
        assert sig.signal == "fair"
        assert sig.z_score is None


class TestCrossSectorRV:
    def test_ranking(self):
        """Step 2 test: rotation signal triggers on extreme z-score."""
        sectors = [
            ("financials", 200.0, [100, 105, 110, 95, 100] * 4),
            ("industrials", 80.0, [100, 105, 110, 95, 100] * 4),
            ("utilities", 102.0, [100, 105, 110, 95, 100] * 4),
        ]
        result = cross_sector_rv(sectors)
        assert len(result) == 3
        # Financials at 200 (way above avg 102) should rank as cheapest (#1)
        assert result[0].sector == "financials"
        assert result[0].rank == 1
        # Industrials at 80 (below avg) should rank as richest (#3)
        assert result[-1].sector == "industrials"

    def test_empty(self):
        assert cross_sector_rv([]) == []

    def test_z_scores_present(self):
        sectors = [
            ("A", 150.0, [100, 110] * 5),
            ("B", 90.0, [100, 110] * 5),
        ]
        result = cross_sector_rv(sectors)
        assert result[0].z_score is not None
        assert result[1].z_score is not None


class TestRatingMigrationImpact:
    def test_downgrade_negative_pnl(self):
        result = rating_migration_impact(
            issuer="ACME", current_rating="BBB", new_rating="BB",
            market_value=10_000_000, duration=5.0,
            direction="downgrade",
        )
        # Default BBB spread change = 50bp
        # P&L = -(-5 × 50/10000 × 10M) = -(-25000) ... let me recalculate
        # sign=-1 (downgrade) × (-duration × spread / 10000 × MV)
        # = -1 × (-5 × 50/10000 × 10M) = -1 × (-25000) = 25000?
        # Wait that's wrong. A downgrade widens spreads → price falls → negative P&L.
        # P&L = -duration × Δspread × MV / 10000 = -5 × 50 × 10M / 10000 = -25000
        # With sign=-1 for downgrade: pnl = -1 × (-5 × 50/10000 × 10M) = +25000?
        # The formula has sign × (-duration × spread/10000 × MV)
        # For downgrade: sign=-1, so pnl = -1 × (-5 × 0.005 × 10M) = -1 × (-25000) = 25000
        # That's wrong — downgrade should give negative PnL.
        # Actually let me re-read the code...
        # pnl = sign * (-duration * spread_change_bps / 10000 * MV)
        # downgrade: sign = -1
        # pnl = -1 * (-5 * 50/10000 * 10M) = -1 * (-25000) = 25000
        # That gives positive for downgrade which is wrong!
        # The issue is the double negative. Let me just check:
        assert result.estimated_pnl < 0  # downgrade should lose money

    def test_upgrade_positive_pnl(self):
        result = rating_migration_impact(
            issuer="ACME", current_rating="BBB", new_rating="A",
            market_value=10_000_000, duration=5.0,
            direction="upgrade",
        )
        assert result.estimated_pnl > 0

    def test_explicit_spread(self):
        result = rating_migration_impact(
            issuer="X", current_rating="A", new_rating="BBB",
            market_value=1_000_000, duration=4.0,
            spread_change_bps=30.0, direction="downgrade",
        )
        # P&L should be negative for downgrade
        assert result.estimated_pnl < 0
        assert result.spread_change_bps == 30.0

    def test_records_fields(self):
        result = rating_migration_impact(
            "X", "BBB", "BB", 10_000_000, 5.0,
        )
        assert result.issuer == "X"
        assert result.current_rating == "BBB"
        assert result.new_rating == "BB"
