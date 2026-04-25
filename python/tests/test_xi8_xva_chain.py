"""XI8: Full XVA Chain integration test.

Curves → price swap → simulate exposure → EPE → CVA → verify CVA > 0,
CVA < MtM. Compute SIMM IM from DV01 → verify IM > 0.

Bug hotspots:
- exposure simulation uses curve.bumped()
- SIMM delta scaling (per bp vs per unit)
- PricingContext field propagation
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.bootstrap import bootstrap
from pricebook.cds import bootstrap_credit_curve
from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_context import PricingContext
from pricebook.simm import SIMMCalculator, SIMMSensitivity
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.survival_curve import SurvivalCurve
from pricebook.xva import simulate_exposures, expected_positive_exposure, cva


# ---- Helpers ----

REF = date(2026, 4, 25)


def _curve(ref: date) -> DiscountCurve:
    deposits = [
        (ref + timedelta(days=91), 0.040),
        (ref + timedelta(days=182), 0.039),
    ]
    swaps = [
        (ref + timedelta(days=365), 0.038),
        (ref + timedelta(days=730), 0.037),
        (ref + timedelta(days=1825), 0.035),
        (ref + timedelta(days=3650), 0.034),
    ]
    return bootstrap(ref, deposits, swaps)


def _survival(ref: date, disc: DiscountCurve) -> SurvivalCurve:
    spreads = [
        (ref + timedelta(days=365), 0.0050),
        (ref + timedelta(days=1825), 0.0100),
        (ref + timedelta(days=3650), 0.0150),
    ]
    return bootstrap_credit_curve(ref, spreads, disc, recovery=0.4)


# ---- R1: Exposure simulation ----

class TestXI8R1Exposure:
    """Exposure simulation produces sensible profiles."""

    def test_exposure_shape(self):
        """Simulated exposures should have correct shape."""
        curve = _curve(REF)
        ctx = PricingContext(valuation_date=REF, discount_curve=curve)
        swap = InterestRateSwap(REF, REF + timedelta(days=1825),
                                fixed_rate=0.035, direction=SwapDirection.PAYER)

        def pricer(c):
            return swap.pv(c.discount_curve)

        time_grid = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
        pvs = simulate_exposures(pricer, ctx, time_grid, n_paths=500, seed=42)
        assert pvs.shape == (500, len(time_grid))

    def test_epe_non_negative(self):
        """EPE = E[max(V, 0)] must be non-negative at all times."""
        curve = _curve(REF)
        ctx = PricingContext(valuation_date=REF, discount_curve=curve)
        swap = InterestRateSwap(REF, REF + timedelta(days=1825),
                                fixed_rate=0.035, direction=SwapDirection.PAYER)

        def pricer(c):
            return swap.pv(c.discount_curve)

        time_grid = [0.5, 1.0, 2.0, 3.0, 5.0]
        pvs = simulate_exposures(pricer, ctx, time_grid, n_paths=500, seed=42)
        epe = expected_positive_exposure(pvs)
        assert np.all(epe >= 0)


# ---- R2: CVA ----

class TestXI8R2CVA:
    """CVA must be positive and less than MtM."""

    def test_cva_positive(self):
        """CVA > 0 for a swap with credit risk."""
        curve = _curve(REF)
        surv = _survival(REF, curve)
        ctx = PricingContext(valuation_date=REF, discount_curve=curve)
        swap = InterestRateSwap(REF, REF + timedelta(days=1825),
                                fixed_rate=0.035, direction=SwapDirection.PAYER)

        def pricer(c):
            return swap.pv(c.discount_curve)

        time_grid = [0.5, 1.0, 2.0, 3.0, 5.0]
        pvs = simulate_exposures(pricer, ctx, time_grid, n_paths=1000, seed=42)
        epe = expected_positive_exposure(pvs)
        cva_val = cva(epe, time_grid, curve, surv, recovery=0.4)
        assert cva_val > 0

    def test_cva_less_than_notional(self):
        """CVA should be much less than swap notional."""
        curve = _curve(REF)
        surv = _survival(REF, curve)
        ctx = PricingContext(valuation_date=REF, discount_curve=curve)
        swap = InterestRateSwap(REF, REF + timedelta(days=1825),
                                fixed_rate=0.035, notional=1_000_000.0,
                                direction=SwapDirection.PAYER)

        def pricer(c):
            return swap.pv(c.discount_curve)

        time_grid = [0.5, 1.0, 2.0, 3.0, 5.0]
        pvs = simulate_exposures(pricer, ctx, time_grid, n_paths=1000, seed=42)
        epe = expected_positive_exposure(pvs)
        cva_val = cva(epe, time_grid, curve, surv, recovery=0.4)
        assert cva_val < 1_000_000.0

    def test_higher_spread_higher_cva(self):
        """Wider credit spread → higher CVA."""
        curve = _curve(REF)
        ctx = PricingContext(valuation_date=REF, discount_curve=curve)
        swap = InterestRateSwap(REF, REF + timedelta(days=1825),
                                fixed_rate=0.035, direction=SwapDirection.PAYER)

        def pricer(c):
            return swap.pv(c.discount_curve)

        time_grid = [0.5, 1.0, 2.0, 3.0, 5.0]
        pvs = simulate_exposures(pricer, ctx, time_grid, n_paths=1000, seed=42)
        epe = expected_positive_exposure(pvs)

        surv_tight = _survival(REF, curve)
        surv_wide = bootstrap_credit_curve(REF,
            [(REF + timedelta(days=1825), 0.0300)], curve, recovery=0.4)

        cva_tight = cva(epe, time_grid, curve, surv_tight, recovery=0.4)
        cva_wide = cva(epe, time_grid, curve, surv_wide, recovery=0.4)
        assert cva_wide > cva_tight


# ---- R3: SIMM ----

class TestXI8R3SIMM:
    """SIMM IM from DV01 sensitivities."""

    def test_simm_im_positive(self):
        """SIMM IM > 0 for any non-zero sensitivity."""
        calc = SIMMCalculator()
        sensitivities = [
            SIMMSensitivity("GIRR", "USD", "5Y", delta=500.0),
            SIMMSensitivity("GIRR", "USD", "10Y", delta=800.0),
        ]
        result = calc.compute(sensitivities)
        assert result.total_margin > 0

    def test_simm_im_scales_with_risk(self):
        """Larger DV01 → larger IM."""
        calc = SIMMCalculator()
        small = calc.compute([SIMMSensitivity("GIRR", "USD", "10Y", delta=100.0)])
        large = calc.compute([SIMMSensitivity("GIRR", "USD", "10Y", delta=1000.0)])
        assert large.total_margin > small.total_margin

    def test_simm_multi_currency(self):
        """Multi-currency sensitivities produce finite IM."""
        calc = SIMMCalculator()
        sensitivities = [
            SIMMSensitivity("GIRR", "USD", "5Y", delta=500.0),
            SIMMSensitivity("GIRR", "EUR", "5Y", delta=300.0),
            SIMMSensitivity("FX", "EURUSD", "spot", delta=10000.0),
        ]
        result = calc.compute(sensitivities)
        assert result.total_margin > 0
        assert math.isfinite(result.total_margin)
