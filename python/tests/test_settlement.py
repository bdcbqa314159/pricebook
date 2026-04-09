"""Tests for settlement framework."""

import pytest
from datetime import date

from pricebook.settlement import (
    SettlementType, get_convention,
    cash_settlement,
    cds_settlement_physical, cds_settlement_cash,
    option_settlement_cash, option_settlement_physical,
    futures_settlement_cash, futures_settlement_physical,
    settlement_risk,
)


REF = date(2024, 1, 15)


# ---- Conventions ----

class TestConventions:
    def test_known_product(self):
        conv = get_convention("ir_swap")
        assert conv["type"] == SettlementType.CASH

    def test_cds_physical(self):
        conv = get_convention("cds_physical")
        assert conv["type"] == SettlementType.PHYSICAL
        assert conv["lag_days"] == 30

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_convention("unknown_product")


# ---- Cash settlement ----

class TestCashSettlement:
    def test_positive_amount(self):
        result = cash_settlement(100_000, REF)
        assert result.amount == 100_000
        assert result.settlement_type == SettlementType.CASH

    def test_settlement_date_with_lag(self):
        result = cash_settlement(100_000, REF, lag_days=2)
        assert result.settlement_date == date(2024, 1, 17)


# ---- CDS settlement ----

class TestCDSSettlement:
    def test_physical_receives_par(self):
        result = cds_settlement_physical(10_000_000, 0.4, REF)
        assert result.protection_payout == 10_000_000
        assert result.recovery_value == pytest.approx(4_000_000)
        assert result.bond_delivered is True

    def test_cash_receives_lgd(self):
        result = cds_settlement_cash(10_000_000, 0.4, REF)
        assert result.protection_payout == pytest.approx(6_000_000)
        assert result.bond_delivered is False

    def test_physical_equals_cash_net(self):
        """Net payout should be the same: (1-R) × notional."""
        phys = cds_settlement_physical(10_000_000, 0.4, REF)
        cash = cds_settlement_cash(10_000_000, 0.4, REF)
        net_phys = phys.protection_payout - phys.recovery_value
        assert net_phys == pytest.approx(cash.protection_payout)

    def test_recovery_zero(self):
        result = cds_settlement_cash(10_000_000, 0.0, REF)
        assert result.protection_payout == pytest.approx(10_000_000)

    def test_settlement_lag(self):
        phys = cds_settlement_physical(10_000_000, 0.4, REF, lag_days=30)
        cash = cds_settlement_cash(10_000_000, 0.4, REF, lag_days=5)
        assert (phys.settlement_date - REF).days == 30
        assert (cash.settlement_date - REF).days == 5


# ---- Option settlement ----

class TestOptionSettlement:
    def test_itm_call_cash(self):
        result = option_settlement_cash(110, 100, True, 100, REF)
        assert result.intrinsic == pytest.approx(10.0)
        assert result.cash_amount == pytest.approx(1000.0)
        assert result.shares_delivered == 0.0

    def test_otm_call_cash(self):
        result = option_settlement_cash(90, 100, True, 100, REF)
        assert result.intrinsic == 0.0
        assert result.cash_amount == 0.0

    def test_itm_put_cash(self):
        result = option_settlement_cash(90, 100, False, 100, REF)
        assert result.intrinsic == pytest.approx(10.0)

    def test_physical_call_delivers_shares(self):
        result = option_settlement_physical(110, 100, True, 100, REF)
        assert result.shares_delivered == 100  # receive shares
        assert result.cash_amount == pytest.approx(-10000)  # pay strike

    def test_physical_put_delivers_shares(self):
        result = option_settlement_physical(90, 100, False, 100, REF)
        assert result.shares_delivered == -100  # deliver shares
        assert result.cash_amount == pytest.approx(10000)  # receive strike


# ---- Futures settlement ----

class TestFuturesSettlement:
    def test_cash_settlement(self):
        result = futures_settlement_cash(5000, 5050, 2, 50, REF)
        assert result.cash_amount == pytest.approx(50 * 2 * 50)
        assert result.physical_delivery is False

    def test_physical_settlement(self):
        result = futures_settlement_physical(100, 102, 1, 1000, REF)
        assert result.physical_delivery is True
        assert result.cash_amount == pytest.approx(2000)

    def test_negative_pnl(self):
        result = futures_settlement_cash(5050, 5000, 1, 50, REF)
        assert result.cash_amount < 0


# ---- Settlement risk ----

class TestSettlementRisk:
    def test_physical_full_exposure(self):
        result = settlement_risk(1_000_000, REF, date(2024, 1, 17), SettlementType.PHYSICAL)
        assert result.exposure == 1_000_000
        assert result.days_at_risk == 2

    def test_cash_reduced_exposure(self):
        result = settlement_risk(1_000_000, REF, date(2024, 1, 17), SettlementType.CASH)
        assert result.exposure < 1_000_000  # net settlement reduces risk

    def test_same_day_zero_days(self):
        result = settlement_risk(1_000_000, REF, REF)
        assert result.days_at_risk == 0
