"""Tests for TRS regulatory capital: SA-CCR, SIMM, KVA, leverage ratio."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.bootstrap import bootstrap
from pricebook.discount_curve import DiscountCurve
from pricebook.trs import TotalReturnSwap, FundingLegSpec
from pricebook.loan import TermLoan
from pricebook.regulatory.trs_capital import (
    trs_sa_ccr_add_on,
    trs_simm_sensitivities,
    trs_kva,
    trs_leverage_exposure,
)


REF = date(2026, 4, 27)


def _curve():
    deposits = [(REF + timedelta(days=91), 0.04)]
    swaps = [(REF + timedelta(days=365), 0.038),
             (REF + timedelta(days=1825), 0.035)]
    return bootstrap(REF, deposits, swaps)


# ---- SA-CCR ----

class TestTRSSACCR:

    def test_equity_trs_ead(self):
        """Equity TRS should map to EQ_SINGLE asset class."""
        curve = _curve()
        trs = TotalReturnSwap(
            underlying=100.0, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365))
        result = trs_sa_ccr_add_on(trs, curve)
        assert result.asset_class == "EQ_SINGLE"
        assert result.ead > 0
        assert result.supervisory_factor == pytest.approx(0.32)

    def test_bond_trs_ead(self):
        """Bond TRS defaults to CR_BBB."""
        curve = _curve()
        from pricebook.bond import FixedRateBond
        bond = FixedRateBond(
            face_value=100, coupon_rate=0.05,
            maturity=REF + timedelta(days=3650),
            issue_date=REF - timedelta(days=365))
        trs = TotalReturnSwap(
            underlying=bond, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365))
        result = trs_sa_ccr_add_on(trs, curve, rating="BBB")
        assert result.asset_class == "CR_BBB"
        assert result.ead > 0

    def test_loan_trs_ead(self):
        curve = _curve()
        loan = TermLoan(REF, REF + timedelta(days=1825),
                        spread=0.03, notional=10_000_000)
        trs = TotalReturnSwap(
            underlying=loan, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365))
        result = trs_sa_ccr_add_on(trs, curve)
        assert result.ead > 0

    def test_rating_affects_sf(self):
        """Higher-risk rating → higher SF."""
        curve = _curve()
        trs = TotalReturnSwap(
            underlying=100.0, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365))
        # Equity doesn't depend on rating, test with loan
        loan = TermLoan(REF, REF + timedelta(days=1825),
                        spread=0.03, notional=10_000_000)
        trs_loan = TotalReturnSwap(
            underlying=loan, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365))
        r_bbb = trs_sa_ccr_add_on(trs_loan, curve, rating="BBB")
        r_b = trs_sa_ccr_add_on(trs_loan, curve, rating="B")
        assert r_b.supervisory_factor > r_bbb.supervisory_factor

    def test_ead_components(self):
        """EAD = alpha × (RC + PFE)."""
        curve = _curve()
        trs = TotalReturnSwap(
            underlying=100.0, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365))
        result = trs_sa_ccr_add_on(trs, curve, alpha=1.4)
        assert result.ead == pytest.approx(1.4 * (result.replacement_cost + result.pfe))


# ---- SIMM ----

class TestTRSSIMM:

    def test_equity_trs_simm(self):
        """Equity TRS should produce EQ delta sensitivity."""
        curve = _curve()
        trs = TotalReturnSwap(
            underlying=100.0, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365))
        simm = trs_simm_sensitivities(trs, curve)
        assert simm.risk_class == "EQ"
        assert simm.total_delta > 0
        assert len(simm.delta_sensitivities) >= 1

    def test_bond_trs_simm(self):
        """Bond TRS should produce GIRR + CSR sensitivities."""
        curve = _curve()
        from pricebook.bond import FixedRateBond
        bond = FixedRateBond(
            face_value=100, coupon_rate=0.05,
            maturity=REF + timedelta(days=3650),
            issue_date=REF - timedelta(days=365))
        trs = TotalReturnSwap(
            underlying=bond, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365))
        simm = trs_simm_sensitivities(trs, curve)
        assert simm.risk_class == "GIRR"
        risk_classes = {s["risk_class"] for s in simm.delta_sensitivities}
        assert "GIRR" in risk_classes
        assert "CSR" in risk_classes

    def test_equity_has_vega(self):
        curve = _curve()
        trs = TotalReturnSwap(
            underlying=100.0, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365),
            sigma=0.25)
        simm = trs_simm_sensitivities(trs, curve)
        assert simm.total_vega > 0


# ---- KVA ----

class TestTRSKVA:

    def test_kva_positive(self):
        """KVA should be positive (capital has a cost)."""
        curve = _curve()
        trs = TotalReturnSwap(
            underlying=100.0, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365))
        result = trs_kva(trs, curve, hurdle_rate=0.10)
        assert result.kva > 0

    def test_kva_scales_with_hurdle(self):
        """Higher hurdle rate → higher KVA."""
        curve = _curve()
        trs = TotalReturnSwap(
            underlying=100.0, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365))
        k_low = trs_kva(trs, curve, hurdle_rate=0.05)
        k_high = trs_kva(trs, curve, hurdle_rate=0.15)
        assert k_high.kva > k_low.kva

    def test_capital_profile_decays(self):
        """Capital profile should decay over time."""
        curve = _curve()
        trs = TotalReturnSwap(
            underlying=100.0, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365))
        result = trs_kva(trs, curve, n_steps=4)
        # First > last (linear decay)
        assert result.capital_profile[0] > result.capital_profile[-1]

    def test_ead_profile_length(self):
        curve = _curve()
        trs = TotalReturnSwap(
            underlying=100.0, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365))
        result = trs_kva(trs, curve, n_steps=8)
        assert len(result.ead_profile) == 9  # n_steps + 1


# ---- Leverage ratio ----

class TestTRSLeverage:

    def test_leverage_exposure_positive(self):
        curve = _curve()
        trs = TotalReturnSwap(
            underlying=100.0, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365))
        result = trs_leverage_exposure(trs, curve)
        assert result.exposure > 0
        assert result.is_off_balance_sheet

    def test_exposure_decomposition(self):
        """exposure = mtm_component + add_on_component."""
        curve = _curve()
        trs = TotalReturnSwap(
            underlying=100.0, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365))
        result = trs_leverage_exposure(trs, curve)
        assert result.exposure == pytest.approx(
            result.mtm_component + result.add_on_component)

    def test_larger_notional_larger_exposure(self):
        curve = _curve()
        trs_small = TotalReturnSwap(
            underlying=100.0, notional=1_000_000,
            start=REF, end=REF + timedelta(days=365))
        trs_large = TotalReturnSwap(
            underlying=100.0, notional=100_000_000,
            start=REF, end=REF + timedelta(days=365))
        r_small = trs_leverage_exposure(trs_small, curve)
        r_large = trs_leverage_exposure(trs_large, curve)
        assert r_large.exposure > r_small.exposure
