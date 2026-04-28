"""Tests for tenor basis calibration: TenorBasis, bootstrap_tenor_basis."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.ibor_curve import (
    IBORCurve, bootstrap_ibor,
    EURIBOR_3M_CONVENTIONS, EURIBOR_6M_CONVENTIONS,
)
from pricebook.schedule import Frequency, generate_schedule
from pricebook.tenor_basis import TenorBasis, bootstrap_tenor_basis


REF = date(2026, 4, 27)


def _ois():
    return DiscountCurve.flat(REF, 0.03)


def _ibor_3m(ois):
    swaps = [
        (REF + timedelta(days=365), 0.032),
        (REF + timedelta(days=730), 0.033),
        (REF + timedelta(days=1825), 0.035),
        (REF + timedelta(days=3650), 0.037),
    ]
    return bootstrap_ibor(REF, EURIBOR_3M_CONVENTIONS, ois, swaps=swaps)


def _basis_quotes():
    """3M vs 6M basis swap quotes (spread on 6M leg)."""
    return [
        (REF + timedelta(days=730), 0.0005),   # 2Y: 5bp
        (REF + timedelta(days=1825), 0.0010),  # 5Y: 10bp
        (REF + timedelta(days=3650), 0.0012),  # 10Y: 12bp
    ]


# ---- TenorBasis ----

class TestTenorBasis:

    def test_spread_interpolation(self):
        basis = TenorBasis(
            reference_date=REF,
            short_tenor=EURIBOR_3M_CONVENTIONS,
            long_tenor=EURIBOR_6M_CONVENTIONS,
            dates=[REF + timedelta(days=365), REF + timedelta(days=3650)],
            spreads=[0.0005, 0.0015],
        )
        # Midpoint should be interpolated
        mid = REF + timedelta(days=2007)  # ~5.5Y
        s = basis.spread(mid)
        assert 0.0005 < s < 0.0015

    def test_spread_flat_extrapolation(self):
        basis = TenorBasis(
            reference_date=REF,
            short_tenor=EURIBOR_3M_CONVENTIONS,
            long_tenor=EURIBOR_6M_CONVENTIONS,
            dates=[REF + timedelta(days=365)],
            spreads=[0.0008],
        )
        # Single point → constant
        assert basis.spread(REF + timedelta(days=100)) == pytest.approx(0.0008)
        assert basis.spread(REF + timedelta(days=3650)) == pytest.approx(0.0008)

    def test_forward_spread(self):
        basis = TenorBasis(
            reference_date=REF,
            short_tenor=EURIBOR_3M_CONVENTIONS,
            long_tenor=EURIBOR_6M_CONVENTIONS,
            dates=[REF + timedelta(days=365), REF + timedelta(days=3650)],
            spreads=[0.0005, 0.0015],
        )
        fs = basis.forward_spread(REF + timedelta(days=365), REF + timedelta(days=730))
        assert math.isfinite(fs)
        assert fs > 0

    def test_empty_basis(self):
        basis = TenorBasis(
            reference_date=REF,
            short_tenor=EURIBOR_3M_CONVENTIONS,
            long_tenor=EURIBOR_6M_CONVENTIONS,
            dates=[], spreads=[],
        )
        assert basis.spread(REF + timedelta(days=365)) == 0.0


# ---- Bootstrap ----

class TestBootstrapTenorBasis:

    def test_basic_bootstrap(self):
        """Bootstrap 6M curve from 3M + basis quotes."""
        ois = _ois()
        ibor_3m = _ibor_3m(ois)
        ibor_6m, basis = bootstrap_tenor_basis(
            REF, ibor_3m, ois,
            basis_swap_quotes=_basis_quotes(),
            long_tenor_conventions=EURIBOR_6M_CONVENTIONS,
        )
        assert ibor_6m.tenor_months == 6
        assert len(basis.dates) == 3
        assert len(basis.spreads) == 3

    def test_basis_swap_repricing(self):
        """The bootstrapped 6M curve should reprice basis swaps at quoted spreads."""
        ois = _ois()
        ibor_3m = _ibor_3m(ois)
        quotes = _basis_quotes()
        ibor_6m, basis = bootstrap_tenor_basis(
            REF, ibor_3m, ois,
            basis_swap_quotes=quotes,
            long_tenor_conventions=EURIBOR_6M_CONVENTIONS,
        )

        short_conv = EURIBOR_3M_CONVENTIONS
        long_conv = EURIBOR_6M_CONVENTIONS

        for mat, quoted_spread in quotes:
            short_sched = generate_schedule(REF, mat, short_conv.float_frequency)
            long_sched = generate_schedule(REF, mat, long_conv.float_frequency)

            # PV of short leg (3M + spread)
            pv_short = sum(
                (ibor_3m.forward_rate(short_sched[i-1], short_sched[i]) + quoted_spread)
                * year_fraction(short_sched[i-1], short_sched[i], short_conv.float_day_count)
                * ois.df(short_sched[i])
                for i in range(1, len(short_sched))
            )

            # PV of long leg (6M flat)
            pv_long = sum(
                ibor_6m.forward_rate(long_sched[i-1], long_sched[i])
                * year_fraction(long_sched[i-1], long_sched[i], long_conv.float_day_count)
                * ois.df(long_sched[i])
                for i in range(1, len(long_sched))
            )

            assert pv_short == pytest.approx(pv_long, abs=1e-6), \
                f"Basis swap {mat} failed: PV_short={pv_short:.8f}, PV_long={pv_long:.8f}"

    def test_zero_spread_identical_curves(self):
        """Zero basis quotes → 6M curve ≈ 3M curve at same tenors."""
        ois = _ois()
        ibor_3m = _ibor_3m(ois)
        zero_quotes = [
            (REF + timedelta(days=730), 0.0),
            (REF + timedelta(days=1825), 0.0),
        ]
        ibor_6m, basis = bootstrap_tenor_basis(
            REF, ibor_3m, ois,
            basis_swap_quotes=zero_quotes,
            long_tenor_conventions=EURIBOR_6M_CONVENTIONS,
        )
        # Forward rates should be close (not identical due to schedule differences)
        d1 = REF + timedelta(days=365)
        d2 = REF + timedelta(days=546)
        fwd_3m = ibor_3m.forward_rate(d1, d2)
        fwd_6m = ibor_6m.forward_rate(d1, d2)
        assert fwd_3m == pytest.approx(fwd_6m, abs=0.002)

    def test_positive_basis(self):
        """With positive basis quotes, 6M forwards > 3M forwards."""
        ois = _ois()
        ibor_3m = _ibor_3m(ois)
        ibor_6m, basis = bootstrap_tenor_basis(
            REF, ibor_3m, ois,
            basis_swap_quotes=_basis_quotes(),
            long_tenor_conventions=EURIBOR_6M_CONVENTIONS,
        )
        # At 5Y point, 6M should be above 3M
        d1 = REF + timedelta(days=1460)
        d2 = REF + timedelta(days=1642)
        assert ibor_6m.forward_rate(d1, d2) > ibor_3m.forward_rate(d1, d2)

    def test_basis_spreads_positive(self):
        """Extracted basis spreads should be positive when quotes are positive."""
        ois = _ois()
        ibor_3m = _ibor_3m(ois)
        _, basis = bootstrap_tenor_basis(
            REF, ibor_3m, ois,
            basis_swap_quotes=_basis_quotes(),
            long_tenor_conventions=EURIBOR_6M_CONVENTIONS,
        )
        for s in basis.spreads:
            assert s > 0

    def test_unsorted_quotes_raises(self):
        ois = _ois()
        ibor_3m = _ibor_3m(ois)
        bad = [(REF + timedelta(days=1825), 0.001), (REF + timedelta(days=730), 0.0005)]
        with pytest.raises(ValueError, match="sorted"):
            bootstrap_tenor_basis(REF, ibor_3m, ois, bad, EURIBOR_6M_CONVENTIONS)
