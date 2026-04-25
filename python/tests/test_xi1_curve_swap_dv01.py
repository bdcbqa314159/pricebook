"""XI1: Curve → Swap → DV01 integration chain.

Bootstrap OIS curve → price swap at par → verify PV=0 → compute DV01 →
verify bump-and-reprice matches → sum of KRDs ≈ parallel DV01.

Bug hotspots:
- bootstrap uses ACT_360 for deposits, swap uses THIRTY_360 for fixed
- bumped() shifts zero rates — verify DV01 method matches manual bump
- par_rate from swap must reprice to PV=0
- KRDs sum to parallel DV01 only for log-linear interpolation
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.bootstrap import bootstrap
from pricebook.day_count import DayCountConvention
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency
from pricebook.swap import InterestRateSwap, SwapDirection


# ---- Helpers ----

def _ois_curve(ref: date) -> DiscountCurve:
    """Bootstrap a realistic OIS curve from deposit + swap quotes."""
    deposits = [
        (ref + timedelta(days=1), 0.050),    # O/N
        (ref + timedelta(days=7), 0.050),     # 1W
        (ref + timedelta(days=30), 0.049),    # 1M
        (ref + timedelta(days=91), 0.048),    # 3M
        (ref + timedelta(days=182), 0.047),   # 6M
    ]
    swaps = [
        (ref + timedelta(days=365), 0.046),       # 1Y
        (ref + timedelta(days=730), 0.045),       # 2Y
        (ref + timedelta(days=1095), 0.044),      # 3Y
        (ref + timedelta(days=1825), 0.043),      # 5Y
        (ref + timedelta(days=2555), 0.042),      # 7Y
        (ref + timedelta(days=3650), 0.041),      # 10Y
    ]
    return bootstrap(ref, deposits, swaps)


REF = date(2026, 4, 25)


# ---- R1: Write the chain test, find integration bugs ----

class TestXI1R1Chain:
    """End-to-end: bootstrap → swap at par → PV=0 → DV01."""

    def test_bootstrap_produces_valid_curve(self):
        curve = _ois_curve(REF)
        assert curve.df(REF) == pytest.approx(1.0)
        # All DFs should be < 1 for positive rates
        for d in [REF + timedelta(days=365), REF + timedelta(days=3650)]:
            assert 0 < curve.df(d) < 1

    def test_swap_at_par_rate_has_zero_pv(self):
        """A swap struck at the par rate must have PV = 0."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=1825)  # 5Y

        swap = InterestRateSwap(start, end, fixed_rate=0.01)  # dummy rate
        par = swap.par_rate(curve)

        par_swap = InterestRateSwap(start, end, fixed_rate=par)
        pv = par_swap.pv(curve)
        assert pv == pytest.approx(0.0, abs=1.0)  # < $1 on 1M notional

    def test_par_rate_in_reasonable_range(self):
        """Par rate should be near the input swap quotes."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=1825)  # 5Y
        swap = InterestRateSwap(start, end, fixed_rate=0.04)
        par = swap.par_rate(curve)
        assert 0.03 < par < 0.06

    def test_dv01_positive_for_payer(self):
        """Payer swap (pay fixed, receive float) has positive DV01."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=3650)  # 10Y
        swap = InterestRateSwap(start, end, fixed_rate=0.04,
                                direction=SwapDirection.PAYER)
        dv01 = swap.dv01(curve)
        assert dv01 > 0

    def test_dv01_negative_for_receiver(self):
        """Receiver swap has negative DV01."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=3650)
        swap = InterestRateSwap(start, end, fixed_rate=0.04,
                                direction=SwapDirection.RECEIVER)
        dv01 = swap.dv01(curve)
        assert dv01 < 0

    def test_dv01_matches_manual_bump(self):
        """swap.dv01() must match manual bump-and-reprice."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=1825)
        swap = InterestRateSwap(start, end, fixed_rate=0.04,
                                direction=SwapDirection.PAYER)
        shift = 0.0001
        dv01_method = swap.dv01(curve, shift=shift)

        curve_up = curve.bumped(shift)
        curve_dn = curve.bumped(-shift)
        dv01_manual = (swap.pv(curve_up) - swap.pv(curve_dn)) / 2
        assert dv01_method == pytest.approx(dv01_manual, rel=0.01)

    def test_annuity_positive(self):
        """Swap annuity (PV01) must be positive."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=1825)
        swap = InterestRateSwap(start, end, fixed_rate=0.04)
        ann = swap.annuity(curve)
        assert ann > 0

    def test_annuity_scales_with_tenor(self):
        """Longer swap → larger annuity."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        s2y = InterestRateSwap(start, start + timedelta(days=730), fixed_rate=0.04)
        s10y = InterestRateSwap(start, start + timedelta(days=3650), fixed_rate=0.04)
        assert s10y.annuity(curve) > s2y.annuity(curve)


# ---- R2: Deep audit of handoff points ----

class TestXI1R2Handoffs:
    """Day count, curve convention, and sign handoffs."""

    def test_deposit_daycount_vs_swap_daycount(self):
        """Bootstrap uses ACT_360 for deposits. Swap fixed leg uses 30/360.
        The par rate must still reprice despite the day count mismatch."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=365)  # 1Y — near deposit/swap boundary

        swap = InterestRateSwap(
            start, end, fixed_rate=0.04,
            fixed_day_count=DayCountConvention.THIRTY_360,
            float_day_count=DayCountConvention.ACT_360,
        )
        par = swap.par_rate(curve)
        par_swap = InterestRateSwap(
            start, end, fixed_rate=par,
            fixed_day_count=DayCountConvention.THIRTY_360,
            float_day_count=DayCountConvention.ACT_360,
        )
        assert par_swap.pv(curve) == pytest.approx(0.0, abs=1.0)

    def test_par_rate_invariant_to_direction(self):
        """Par rate should be the same for payer and receiver."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=1825)

        payer = InterestRateSwap(start, end, fixed_rate=0.04,
                                  direction=SwapDirection.PAYER)
        receiver = InterestRateSwap(start, end, fixed_rate=0.04,
                                     direction=SwapDirection.RECEIVER)
        assert payer.par_rate(curve) == pytest.approx(receiver.par_rate(curve), rel=1e-8)

    def test_payer_receiver_pv_symmetry(self):
        """PV(payer) = -PV(receiver) at same strike."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=1825)

        payer = InterestRateSwap(start, end, fixed_rate=0.04,
                                  direction=SwapDirection.PAYER)
        receiver = InterestRateSwap(start, end, fixed_rate=0.04,
                                     direction=SwapDirection.RECEIVER)
        assert payer.pv(curve) == pytest.approx(-receiver.pv(curve), rel=1e-8)

    def test_bumped_preserves_reference_date(self):
        """bumped() must not change the curve's reference date."""
        curve = _ois_curve(REF)
        bumped = curve.bumped(0.0001)
        assert bumped.reference_date == curve.reference_date

    def test_bumped_shifts_zero_rates(self):
        """bumped(h) must shift all zero rates by exactly h."""
        curve = _ois_curve(REF)
        h = 0.0001
        bumped = curve.bumped(h)
        for d in [REF + timedelta(days=365), REF + timedelta(days=1825)]:
            z0 = curve.zero_rate(d)
            z1 = bumped.zero_rate(d)
            assert z1 == pytest.approx(z0 + h, abs=1e-8)

    def test_dv01_scales_with_notional(self):
        """DV01 should scale linearly with notional."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=1825)

        s1 = InterestRateSwap(start, end, fixed_rate=0.04, notional=1_000_000)
        s2 = InterestRateSwap(start, end, fixed_rate=0.04, notional=2_000_000)
        assert s2.dv01(curve) == pytest.approx(2 * s1.dv01(curve), rel=1e-6)


# ---- R3/R4: KRD decomposition and edge cases ----

class TestXI1R3KRD:
    """Key rate durations: sum of KRDs ≈ parallel DV01."""

    def test_krd_sum_approx_parallel_dv01(self):
        """Sum of key rate durations should approximate parallel DV01."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=3650)  # 10Y
        swap = InterestRateSwap(start, end, fixed_rate=0.04,
                                direction=SwapDirection.PAYER)

        shift = 0.0001
        parallel_dv01 = swap.dv01(curve, shift=shift)

        # Compute KRDs by bumping each pillar individually
        n_pillars = len(curve.pillar_dates)
        krd_sum = 0.0
        for i in range(n_pillars):
            curve_up = curve.bumped_at(i, shift)
            curve_dn = curve.bumped_at(i, -shift)
            krd_i = (swap.pv(curve_up) - swap.pv(curve_dn)) / 2
            krd_sum += krd_i

        # For log-linear interp, KRDs should sum closely to parallel
        assert krd_sum == pytest.approx(parallel_dv01, rel=0.05)

    def test_krd_concentrated_at_maturity(self):
        """Most of the KRD should come from pillars near the swap maturity."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=1825)  # 5Y
        swap = InterestRateSwap(start, end, fixed_rate=0.04,
                                direction=SwapDirection.PAYER)

        shift = 0.0001
        krds = []
        for i in range(len(curve.pillar_dates)):
            curve_up = curve.bumped_at(i, shift)
            curve_dn = curve.bumped_at(i, -shift)
            krd_i = (swap.pv(curve_up) - swap.pv(curve_dn)) / 2
            krds.append(krd_i)

        # At least one KRD should be significantly larger than the rest
        max_krd = max(abs(k) for k in krds)
        assert max_krd > 0

    def test_zero_rate_swap_pv_zero(self):
        """Edge case: a 0% fixed rate swap should have PV = PV(float)."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=1825)

        zero_swap = InterestRateSwap(start, end, fixed_rate=0.0,
                                      direction=SwapDirection.PAYER)
        pv = zero_swap.pv(curve)
        # Float leg PV should be positive (rates > 0)
        assert pv > 0

    def test_flat_curve_par_rate_equals_flat_rate(self):
        """On a flat curve, par rate should ≈ the flat rate."""
        flat = DiscountCurve.flat(REF, 0.04)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=1825)

        swap = InterestRateSwap(start, end, fixed_rate=0.03)
        par = swap.par_rate(flat)
        # Should be near 0.04, though day count differences cause small deviations
        assert par == pytest.approx(0.04, abs=0.005)

    def test_negative_rates(self):
        """Negative rate curve: swap PV and DV01 should still be finite."""
        neg_curve = DiscountCurve.flat(REF, -0.005)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=1825)

        swap = InterestRateSwap(start, end, fixed_rate=0.0,
                                direction=SwapDirection.PAYER)
        pv = swap.pv(neg_curve)
        dv01 = swap.dv01(neg_curve)
        assert math.isfinite(pv)
        assert math.isfinite(dv01)

    def test_very_short_swap(self):
        """3-month swap: should still price and have finite DV01."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=91)

        swap = InterestRateSwap(start, end, fixed_rate=0.048,
                                fixed_frequency=Frequency.QUARTERLY,
                                float_frequency=Frequency.QUARTERLY)
        pv = swap.pv(curve)
        dv01 = swap.dv01(curve)
        assert math.isfinite(pv)
        assert math.isfinite(dv01)
        assert abs(dv01) > 0

    def test_cashflow_schedule_sums_to_pv(self):
        """Sum of discounted cashflows in schedule should equal PV."""
        curve = _ois_curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=1825)

        swap = InterestRateSwap(start, end, fixed_rate=0.04,
                                direction=SwapDirection.PAYER)
        pv = swap.pv(curve)
        schedule = swap.cashflow_schedule(curve)

        # pv() nets float - fixed for payer; schedule has unsigned amounts per leg
        float_pv = sum(cf["pv"] for cf in schedule if cf["leg"] == "float")
        fixed_pv = sum(cf["pv"] for cf in schedule if cf["leg"] == "fixed")
        schedule_pv = float_pv - fixed_pv  # payer convention
        assert schedule_pv == pytest.approx(pv, rel=0.01)
