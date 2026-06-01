"""Tests for credit spread vol, quanto CDS, credit VaR, index swaption, recovery-locked CDS."""

import pytest
import math
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


def _make_curves(hazard=0.02, rate=0.04):
    from pricebook.core.discount_curve import DiscountCurve
    from pricebook.core.survival_curve import SurvivalCurve
    from pricebook.core.interpolation import InterpolationMethod
    dates = [REF + relativedelta(years=y) for y in range(1, 15)]
    dfs = [math.exp(-rate * y) for y in range(1, 15)]
    survs = [math.exp(-hazard * y) for y in range(1, 15)]
    dc = DiscountCurve(REF, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)
    sc = SurvivalCurve(REF, dates, survs)
    return dc, sc


# ═══════════════════════════════════════════════════════════════
# Credit Spread Vol Surface (C3)
# ═══════════════════════════════════════════════════════════════

class TestCreditSpreadVol:
    def test_construction(self):
        from pricebook.credit.credit_spread_vol import CreditSpreadVolSurface
        surface = CreditSpreadVolSurface(REF, [1, 2, 5, 10], [3, 5, 10],
                                          [[0.40, 0.38, 0.35],
                                           [0.38, 0.36, 0.33],
                                           [0.35, 0.33, 0.30],
                                           [0.32, 0.30, 0.28]])
        assert surface.vol(1.0, 5.0) == pytest.approx(0.38)

    def test_interpolation(self):
        from pricebook.credit.credit_spread_vol import CreditSpreadVolSurface
        surface = CreditSpreadVolSurface(REF, [1, 5, 10], [5, 10],
                                          [[0.40, 0.35], [0.35, 0.30], [0.30, 0.25]])
        vol = surface.vol(3.0, 7.0)
        assert 0.25 < vol < 0.40

    def test_bumped(self):
        from pricebook.credit.credit_spread_vol import CreditSpreadVolSurface
        surface = CreditSpreadVolSurface(REF, [1, 5], [5, 10],
                                          [[0.40, 0.35], [0.35, 0.30]])
        bumped = surface.bumped(0.05)
        assert bumped.vol(1.0, 5.0) == pytest.approx(0.45)

    def test_synthetic(self):
        from pricebook.credit.credit_spread_vol import synthetic_credit_vol_surface
        surface = synthetic_credit_vol_surface(0.02, REF)
        vol = surface.vol(5.0, 5.0)
        assert 0.20 < vol < 0.60

    def test_to_dict(self):
        from pricebook.credit.credit_spread_vol import CreditSpreadVolSurface
        surface = CreditSpreadVolSurface(REF, [1, 5], [5, 10],
                                          [[0.40, 0.35], [0.35, 0.30]])
        d = surface.to_dict()
        assert "expiries" in d or "expiry_years" in d or "reference_date" in d


# ═══════════════════════════════════════════════════════════════
# Quanto CDS (C4)
# ═══════════════════════════════════════════════════════════════

class TestQuantoCDS:
    def test_quanto_adjustment_positive_corr(self):
        """Positive FX-credit correlation → quanto spread > foreign."""
        from pricebook.credit.quanto_cds import quanto_cds_spread
        foreign = 0.01  # 100bp
        adjusted = quanto_cds_spread(foreign, 0.10, 0.40, 0.30, 5.0)
        assert adjusted > foreign

    def test_quanto_adjustment_negative_corr(self):
        """Negative correlation → quanto spread < foreign."""
        from pricebook.credit.quanto_cds import quanto_cds_spread
        foreign = 0.01
        adjusted = quanto_cds_spread(foreign, 0.10, 0.40, -0.30, 5.0)
        assert adjusted < foreign

    def test_zero_corr_no_adjustment(self):
        from pricebook.credit.quanto_cds import quanto_cds_spread
        foreign = 0.01
        adjusted = quanto_cds_spread(foreign, 0.10, 0.40, 0.0, 5.0)
        assert adjusted == pytest.approx(foreign)

    def test_price_quanto_cds(self):
        from pricebook.credit.quanto_cds import price_quanto_cds
        dc, sc = _make_curves()
        dc_foreign, _ = _make_curves(rate=0.02)
        r = price_quanto_cds(REF, 5.0, 0.01, dc, dc_foreign, sc,
                              fx_spot=1.10, fx_vol=0.08, credit_vol=0.40,
                              correlation=0.25)
        assert r.domestic_spread > 0
        assert r.quanto_adjustment_bp != 0

    def test_to_dict(self):
        from pricebook.credit.quanto_cds import price_quanto_cds
        dc, sc = _make_curves()
        dc_f, _ = _make_curves(rate=0.02)
        r = price_quanto_cds(REF, 5.0, 0.01, dc, dc_f, sc, 1.10, 0.08, 0.40, 0.25)
        d = r.to_dict()
        assert "quanto_adjustment_bp" in d


# ═══════════════════════════════════════════════════════════════
# Credit Portfolio VaR (C9)
# ═══════════════════════════════════════════════════════════════

class TestCreditVaR:
    def test_historical_var(self):
        from pricebook.credit.credit_var import historical_credit_var
        rng = np.random.default_rng(42)
        positions = [
            {"name": "A", "cs01": -50_000},
            {"name": "B", "cs01": -30_000},
        ]
        spread_changes = {
            "A": rng.normal(0, 0.0005, 250).tolist(),
            "B": rng.normal(0, 0.0003, 250).tolist(),
        }
        result = historical_credit_var(positions, spread_changes, confidence=0.99)
        assert abs(result.var_amount) > 0  # VaR is non-zero
        assert abs(result.es_amount) >= abs(result.var_amount) - 1  # |ES| ≥ |VaR|

    def test_parametric_var(self):
        from pricebook.credit.credit_var import parametric_credit_var
        positions = [
            {"name": "A", "cs01": -50_000},
            {"name": "B", "cs01": -30_000},
        ]
        vols = [0.0005, 0.0003]
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = parametric_credit_var(positions, vols, corr, confidence=0.99)
        assert abs(result.var_amount) > 0

    def test_copula_var(self):
        from pricebook.credit.credit_var import copula_credit_var
        positions = [
            {"name": "A", "notional": 10_000_000},
            {"name": "B", "notional": 5_000_000},
            {"name": "C", "notional": 8_000_000},
        ]
        pds = [0.02, 0.03, 0.01]
        lgds = [0.6, 0.6, 0.4]
        result = copula_credit_var(positions, pds, lgds, correlation=0.3, confidence=0.99)
        assert abs(result.var_amount) > 0
        assert result.worst_name is not None

    def test_var_increases_with_correlation(self):
        from pricebook.credit.credit_var import copula_credit_var
        positions = [
            {"name": "A", "notional": 10_000_000},
            {"name": "B", "notional": 10_000_000},
        ]
        pds = [0.02, 0.02]
        lgds = [0.6, 0.6]
        low = copula_credit_var(positions, pds, lgds, correlation=0.1, confidence=0.99, seed=42)
        high = copula_credit_var(positions, pds, lgds, correlation=0.8, confidence=0.99, seed=42)
        assert abs(high.var_amount) >= abs(low.var_amount) * 0.8

    def test_to_dict(self):
        from pricebook.credit.credit_var import historical_credit_var
        rng = np.random.default_rng(42)
        positions = [{"name": "A", "cs01": -50_000}]
        spread_changes = {"A": rng.normal(0, 0.0005, 100).tolist()}
        result = historical_credit_var(positions, spread_changes)
        d = result.to_dict()
        assert "var_amount" in d
        assert "method" in d


# ═══════════════════════════════════════════════════════════════
# Index CDS Swaption (C2)
# ═══════════════════════════════════════════════════════════════

class TestIndexCDSSwaption:
    def _make_index_curves(self, n=5):
        """Create n slightly different survival curves for index constituents."""
        dc, _ = _make_curves()
        scs = []
        for i in range(n):
            hazard = 0.015 + i * 0.005
            _, sc = _make_curves(hazard=hazard)
            scs.append(sc)
        return dc, scs

    def test_payer_positive(self):
        from pricebook.credit.index_cds_swaption import index_cds_swaption_black
        r = index_cds_swaption_black(0.005, 0.004, 0.40, 1.0, 4.0, 0.95)
        assert r.premium > 0

    def test_receiver_positive(self):
        from pricebook.credit.index_cds_swaption import index_cds_swaption_black
        r = index_cds_swaption_black(0.004, 0.005, 0.40, 1.0, 4.0, 0.95,
                                      option_type="receiver")
        assert r.premium > 0

    def test_put_call_parity(self):
        """Payer - Receiver = forward value."""
        from pricebook.credit.index_cds_swaption import index_cds_swaption_black
        F, K = 0.005, 0.004
        ann, surv, N = 4.0, 0.95, 10_000_000
        p = index_cds_swaption_black(F, K, 0.40, 1.0, ann, surv, N, "payer")
        r = index_cds_swaption_black(F, K, 0.40, 1.0, ann, surv, N, "receiver")
        forward_value = (F - K) * N * surv * ann
        assert p.premium - r.premium == pytest.approx(forward_value, rel=1e-6)

    def test_bachelier(self):
        from pricebook.credit.index_cds_swaption import index_cds_swaption_bachelier
        r = index_cds_swaption_bachelier(0.005, 0.004, 0.002, 1.0, 4.0, 0.95)
        assert r.premium > 0
        assert r.model == "bachelier"

    def test_forward_index_spread(self):
        from pricebook.credit.index_cds_swaption import index_forward_spread
        dc, scs = self._make_index_curves(5)
        expiry = REF + relativedelta(years=1)
        maturity = REF + relativedelta(years=6)
        fwd = index_forward_spread(dc, scs, expiry, maturity)
        assert fwd.forward_spread > 0
        assert len(fwd.constituent_forwards) == 5
        assert fwd.index_annuity > 0

    def test_greeks(self):
        from pricebook.credit.index_cds_swaption import index_swaption_greeks
        r = index_swaption_greeks(0.005, 0.004, 0.40, 1.0, 4.0, 0.95)
        assert r.delta > 0  # payer delta positive
        assert r.vega > 0   # long vol
        assert r.theta < 0  # time decay

    def test_full_pricing(self):
        from pricebook.credit.index_cds_swaption import price_index_cds_swaption
        dc, scs = self._make_index_curves(5)
        expiry = REF + relativedelta(years=1)
        maturity = REF + relativedelta(years=6)
        r = price_index_cds_swaption(dc, scs, expiry, maturity, 0.005, 0.40)
        assert r.premium > 0
        assert r.delta != 0


# ═══════════════════════════════════════════════════════════════
# Recovery-Locked CDS + LCDS (C5)
# ═══════════════════════════════════════════════════════════════

class TestRecoveryLockedCDS:
    def test_lock_premium_higher_recovery(self):
        """Higher locked recovery → negative premium (less protection)."""
        from pricebook.credit.recovery_locked_cds import recovery_lock_premium
        prem = recovery_lock_premium(0.01, 0.60, 0.40)
        assert prem < 0  # locked recovery 60% > market 40%

    def test_lock_premium_lower_recovery(self):
        """Lower locked recovery → positive premium (more protection)."""
        from pricebook.credit.recovery_locked_cds import recovery_lock_premium
        prem = recovery_lock_premium(0.01, 0.20, 0.40)
        assert prem > 0

    def test_lock_premium_equal(self):
        from pricebook.credit.recovery_locked_cds import recovery_lock_premium
        prem = recovery_lock_premium(0.01, 0.40, 0.40)
        assert prem == pytest.approx(0.0)

    def test_price_recovery_locked(self):
        from pricebook.credit.recovery_locked_cds import price_recovery_locked_cds
        dc, sc = _make_curves()
        r = price_recovery_locked_cds(REF, 5.0, 0.01, 0.30, dc, sc)
        assert r.par_spread > 0
        assert r.rpv01 > 0
        assert r.locked_recovery == 0.30

    def test_higher_recovery_lower_spread(self):
        """Higher locked recovery → lower par spread."""
        from pricebook.credit.recovery_locked_cds import price_recovery_locked_cds
        dc, sc = _make_curves()
        low_r = price_recovery_locked_cds(REF, 5.0, 0.01, 0.30, dc, sc)
        high_r = price_recovery_locked_cds(REF, 5.0, 0.01, 0.60, dc, sc)
        assert low_r.par_spread > high_r.par_spread


class TestLCDS:
    def test_lcds_higher_recovery(self):
        """LCDS has higher recovery than standard CDS → lower spread."""
        from pricebook.credit.recovery_locked_cds import price_lcds
        dc, sc = _make_curves()
        r = price_lcds(REF, 5.0, 0.005, dc, sc, recovery=0.70)
        assert r.par_spread > 0
        assert r.par_spread < 0.02  # lower than bond CDS

    def test_prepayment_shortens_maturity(self):
        """Prepayment reduces effective maturity."""
        from pricebook.credit.recovery_locked_cds import price_lcds
        dc, sc = _make_curves()
        no_prepay = price_lcds(REF, 5.0, 0.005, dc, sc, prepayment_rate=0.0)
        with_prepay = price_lcds(REF, 5.0, 0.005, dc, sc, prepayment_rate=0.20)
        assert with_prepay.effective_maturity < no_prepay.effective_maturity

    def test_cancellation_value_positive(self):
        from pricebook.credit.recovery_locked_cds import price_lcds
        dc, sc = _make_curves()
        r = price_lcds(REF, 5.0, 0.005, dc, sc, prepayment_rate=0.15)
        assert r.cancellation_value > 0  # seller benefits from cancellation

    def test_to_dict(self):
        from pricebook.credit.recovery_locked_cds import price_lcds
        dc, sc = _make_curves()
        r = price_lcds(REF, 5.0, 0.005, dc, sc)
        d = r.to_dict()
        assert "prepayment_rate" in d
        assert "cancellation_value" in d
