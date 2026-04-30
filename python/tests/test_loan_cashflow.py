"""Tests for loan cashflow features: floor, sweep, PIK, grid, CSA."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from pricebook.discount_curve import DiscountCurve
from pricebook.loan_cashflow import (
    FlooredTermLoan, PIKTermLoan, PricingGrid,
    ExcessCashFlowSweep, SOFRCSAAdjustment, SOFR_CSA_BPS,
)
from pricebook.serialisable import from_dict

REF = date(2026, 4, 28)
END = REF + timedelta(days=1825)


def _disc(rate=0.03):
    return DiscountCurve.flat(REF, rate)


# ---- SOFR Floor ----

class TestFlooredTermLoan:

    def test_floor_above_rate(self):
        """When SOFR < floor, coupon = floor + spread (not SOFR + spread)."""
        loan = FlooredTermLoan(REF, END, spread=0.03, floor_rate=0.05,
                                notional=10_000_000)
        # With 1% SOFR and 5% floor, floor is binding
        disc = _disc(0.01)
        flows = loan.cashflows(disc)
        # First coupon should use floor (5%) + spread (3%) = 8%
        _, interest, _ = flows[0]
        yf = interest / (10_000_000 * 0.08)  # approximate year fraction
        assert interest > 0
        # Compare with unfloored
        from pricebook.loan import TermLoan
        unfloored = TermLoan(REF, END, spread=0.03, notional=10_000_000)
        _, int_unfl, _ = unfloored.cashflows(disc)[0]
        assert interest > int_unfl  # floored coupon is higher

    def test_floor_below_rate(self):
        """When SOFR > floor, floor doesn't bind."""
        loan = FlooredTermLoan(REF, END, spread=0.03, floor_rate=0.01,
                                notional=10_000_000)
        disc = _disc(0.05)  # SOFR at 5%, floor at 1%
        from pricebook.loan import TermLoan
        unfloored = TermLoan(REF, END, spread=0.03, notional=10_000_000)
        pv_fl = loan.pv(disc)
        pv_unfl = unfloored.pv(disc)
        assert pv_fl == pytest.approx(pv_unfl, rel=0.01)

    def test_floor_value_positive(self):
        """Embedded floor has positive value when SOFR near floor."""
        loan = FlooredTermLoan(REF, END, spread=0.03, floor_rate=0.03,
                                notional=10_000_000)
        fv = loan.floor_value(_disc(0.03), vol=0.50)
        assert fv > 0

    def test_floor_value_zero_when_deep_otm(self):
        """Floor value ≈ 0 when SOFR >> floor."""
        loan = FlooredTermLoan(REF, END, spread=0.03, floor_rate=0.001,
                                notional=10_000_000)
        fv = loan.floor_value(_disc(0.10), vol=0.50)
        assert fv < 1000  # very small for 10M notional


class TestFlooredSerialisation:

    def test_round_trip(self):
        loan = FlooredTermLoan(REF, END, spread=0.03, floor_rate=0.02,
                                notional=5_000_000)
        d = loan.to_dict()
        assert d["type"] == "floored_term_loan"
        loan2 = from_dict(d)
        assert loan2.floor_rate == 0.02
        assert loan2.spread == 0.03

    def test_with_csa_and_grid(self):
        grid = PricingGrid([3.0, 4.0, 5.0], [0.020, 0.025, 0.030, 0.035])
        csa = SOFRCSAAdjustment(tenor_months=3)
        loan = FlooredTermLoan(REF, END, spread=0.03, floor_rate=0.01,
                                csa=csa, pricing_grid=grid)
        d = loan.to_dict()
        loan2 = from_dict(d)
        assert loan2.csa.csa_bps == pytest.approx(26.161)
        assert loan2.pricing_grid.grid_spread(3.5) == 0.025


# ---- Excess Cash Flow Sweep ----

class TestSweep:

    def test_positive_excess(self):
        sweep = ExcessCashFlowSweep(sweep_pct=0.50, capex_budget=10_000_000,
                                     debt_service=5_000_000)
        prepay = sweep.mandatory_prepayment(ebitda=25_000_000)
        assert prepay == pytest.approx(5_000_000)  # (25-10-5) × 50%

    def test_no_excess(self):
        sweep = ExcessCashFlowSweep(sweep_pct=0.50, capex_budget=10_000_000,
                                     debt_service=5_000_000)
        prepay = sweep.mandatory_prepayment(ebitda=10_000_000)
        assert prepay == 0.0

    def test_higher_ebitda_more_sweep(self):
        sweep = ExcessCashFlowSweep(sweep_pct=0.75)
        p1 = sweep.mandatory_prepayment(ebitda=10_000_000)
        p2 = sweep.mandatory_prepayment(ebitda=20_000_000)
        assert p2 > p1


# ---- Pricing Grid ----

class TestPricingGrid:

    def test_lookup(self):
        grid = PricingGrid([3.0, 4.0, 5.0], [0.020, 0.025, 0.030, 0.035])
        assert grid.grid_spread(2.5) == 0.020
        assert grid.grid_spread(3.5) == 0.025
        assert grid.grid_spread(4.5) == 0.030
        assert grid.grid_spread(6.0) == 0.035

    def test_boundary(self):
        grid = PricingGrid([3.0], [0.020, 0.030])
        assert grid.grid_spread(3.0) == 0.020  # at boundary → lower bucket
        assert grid.grid_spread(3.1) == 0.030

    def test_round_trip(self):
        grid = PricingGrid([3.0, 5.0], [0.02, 0.03, 0.04])
        d = grid.to_dict()
        grid2 = PricingGrid.from_dict(d)
        assert grid2.grid_spread(4.0) == 0.03

    def test_invalid_lengths(self):
        with pytest.raises(ValueError, match="spreads length"):
            PricingGrid([3.0, 4.0], [0.02, 0.03])


# ---- PIK ----

class TestPIKTermLoan:

    def test_pik_grows_outstanding(self):
        """During PIK, outstanding should increase."""
        pik_end = REF + timedelta(days=365)
        loan = PIKTermLoan(REF, END, spread=0.03, pik_rate=0.05,
                            pik_end=pik_end, notional=10_000_000)
        flows = loan.cashflows(_disc())
        # First 4 periods (1 year quarterly) are PIK: cash_interest = 0
        for _, interest, _ in flows[:4]:
            assert interest == 0.0
        # After PIK, interest should be on higher outstanding
        _, interest_after, _ = flows[4]
        assert interest_after > 0

    def test_pik_serialisation(self):
        loan = PIKTermLoan(REF, END, spread=0.03, pik_rate=0.05,
                            pik_end=REF + timedelta(days=365))
        d = loan.to_dict()
        assert d["type"] == "pik_term_loan"
        loan2 = from_dict(d)
        assert loan2.pik_rate == 0.05


# ---- SOFR CSA ----

class TestSOFRCSA:

    def test_3m_csa(self):
        csa = SOFRCSAAdjustment(tenor_months=3)
        assert csa.csa_bps == pytest.approx(26.161)
        assert csa.csa_rate == pytest.approx(0.0026161)

    def test_1m_csa(self):
        csa = SOFRCSAAdjustment(tenor_months=1)
        assert csa.csa_bps == pytest.approx(11.448)

    def test_values_match_isda(self):
        assert SOFR_CSA_BPS[1] == pytest.approx(11.448)
        assert SOFR_CSA_BPS[3] == pytest.approx(26.161)
        assert SOFR_CSA_BPS[6] == pytest.approx(42.826)
        assert SOFR_CSA_BPS[12] == pytest.approx(71.513)
