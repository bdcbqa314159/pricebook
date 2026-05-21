"""Tests for RFR futures instruments."""

import pytest
from datetime import date

from pricebook.fixed_income.rfr_futures import (
    RFRFutureSpec, generate_rfr_contracts, rfr_futures_convexity,
    rfr_futures_to_forwards, list_futures_currencies,
)


REF = date(2024, 1, 15)


class TestContractGeneration:
    def test_usd_contracts(self):
        contracts = generate_rfr_contracts("USD", REF, n_1m=6, n_3m=8)
        assert len(contracts["1M"]) == 6
        assert len(contracts["3M"]) == 8
        assert all(c.rfr_name == "SOFR" for c in contracts["1M"])
        assert all(c.rfr_name == "SOFR" for c in contracts["3M"])

    def test_gbp_contracts(self):
        contracts = generate_rfr_contracts("GBP", REF)
        assert all(c.rfr_name == "SONIA" for c in contracts["1M"])

    def test_eur_contracts(self):
        contracts = generate_rfr_contracts("EUR", REF)
        assert all(c.rfr_name == "ESTR" for c in contracts["1M"])

    def test_1m_monthly(self):
        """1M contracts should be consecutive months."""
        contracts = generate_rfr_contracts("USD", REF, n_1m=4)["1M"]
        months = [c.contract_month.month for c in contracts]
        assert months == [2, 3, 4, 5]  # Feb, Mar, Apr, May

    def test_3m_quarterly(self):
        """3M contracts should be IMM quarters."""
        contracts = generate_rfr_contracts("USD", REF, n_3m=4)["3M"]
        months = [c.contract_month.month for c in contracts]
        assert all(m in [3, 6, 9, 12] for m in months)

    def test_accrual_dates(self):
        """Accrual start = first of month, end = last of month for 1M."""
        contracts = generate_rfr_contracts("USD", REF, n_1m=1)["1M"]
        c = contracts[0]
        assert c.accrual_start.day == 1
        assert c.accrual_end.month == c.accrual_start.month

    def test_unknown_currency_raises(self):
        with pytest.raises(ValueError, match="No RFR futures"):
            generate_rfr_contracts("ZAR", REF)

    def test_list_currencies(self):
        ccys = list_futures_currencies()
        assert "USD" in ccys
        assert "GBP" in ccys
        assert "EUR" in ccys
        assert len(ccys) == 5


class TestImpliedRate:
    def test_price_to_rate(self):
        spec = RFRFutureSpec("USD", "SOFR", "3M", date(2024, 3, 1),
                              date(2024, 3, 1), date(2024, 5, 31), price=94.75)
        assert abs(spec.implied_rate - 0.0525) < 1e-10

    def test_set_rate(self):
        spec = RFRFutureSpec("USD", "SOFR", "3M", date(2024, 3, 1),
                              date(2024, 3, 1), date(2024, 5, 31))
        spec.implied_rate = 0.05
        assert abs(spec.price - 95.0) < 1e-10


class TestConvexity:
    def test_positive_adjustment(self):
        """Convexity adjustment should be positive (futures rate > forward rate)."""
        spec = RFRFutureSpec("USD", "SOFR", "3M", date(2025, 3, 1),
                              date(2025, 3, 1), date(2025, 5, 31), price=95.0)
        ca = rfr_futures_convexity(spec, REF)
        assert ca > 0

    def test_increases_with_maturity(self):
        """Longer-dated futures have larger convexity adjustments."""
        s1 = RFRFutureSpec("USD", "SOFR", "3M", date(2024, 6, 1),
                            date(2024, 6, 1), date(2024, 8, 31), price=95.0)
        s2 = RFRFutureSpec("USD", "SOFR", "3M", date(2026, 6, 1),
                            date(2026, 6, 1), date(2026, 8, 31), price=95.0)
        ca1 = rfr_futures_convexity(s1, REF)
        ca2 = rfr_futures_convexity(s2, REF)
        assert ca2 > ca1

    def test_near_contract_small(self):
        """Very near-term contract → small adjustment."""
        spec = RFRFutureSpec("USD", "SOFR", "3M", date(2024, 2, 1),
                              date(2024, 2, 1), date(2024, 4, 30), price=95.0)
        ca = rfr_futures_convexity(spec, REF)
        assert ca < 0.0001  # very small for near-term


class TestToForwards:
    def test_basic(self):
        specs = [
            RFRFutureSpec("USD", "SOFR", "3M", date(2024, 3, 1),
                           date(2024, 3, 1), date(2024, 5, 31), price=95.0),
            RFRFutureSpec("USD", "SOFR", "3M", date(2024, 6, 1),
                           date(2024, 6, 1), date(2024, 8, 31), price=94.75),
        ]
        fwds = rfr_futures_to_forwards(specs, REF)
        assert len(fwds) == 2
        # Forward = implied - convexity < implied
        for start, end, fwd in fwds:
            assert fwd > 0
            assert fwd < 0.06  # reasonable range

    def test_skips_zero_price(self):
        specs = [
            RFRFutureSpec("USD", "SOFR", "3M", date(2024, 3, 1),
                           date(2024, 3, 1), date(2024, 5, 31), price=0.0),
        ]
        fwds = rfr_futures_to_forwards(specs, REF)
        assert len(fwds) == 0


class TestSerialization:
    def test_to_dict(self):
        spec = RFRFutureSpec("USD", "SOFR", "3M", date(2024, 3, 1),
                              date(2024, 3, 1), date(2024, 5, 31), price=95.0)
        d = spec.to_dict()
        assert d["rfr_name"] == "SOFR"
        assert d["implied_rate"] == 0.05
