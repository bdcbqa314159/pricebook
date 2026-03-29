"""Slice 6 round-trip validation.

CIP holds, FX swap at fair rates has PV=0, basis curve reprices forwards.
"""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.currency import Currency, CurrencyPair
from pricebook.fx_forward import FXForward
from pricebook.fx_swap import FXSwap
from pricebook.xccy_basis import implied_basis_spread, bootstrap_basis_curve
from pricebook.discount_curve import DiscountCurve


REF = date(2024, 1, 15)
EURUSD = CurrencyPair(Currency.EUR, Currency.USD)
GBPUSD = CurrencyPair(Currency.GBP, Currency.USD)
SPOT_EURUSD = 1.10
SPOT_GBPUSD = 1.27


def _flat_curve(ref: date, rate: float) -> DiscountCurve:
    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    dates = [date.fromordinal(ref.toordinal() + int(t * 365)) for t in tenors]
    dfs = [math.exp(-rate * t) for t in tenors]
    return DiscountCurve(ref, dates, dfs)


class TestCIPHolds:
    """Forward computed from two curves matches direct formula."""

    def test_cip_eurusd(self):
        eur = _flat_curve(REF, rate=0.04)
        usd = _flat_curve(REF, rate=0.05)
        mat = REF + relativedelta(years=2)
        fwd = FXForward.forward_rate(SPOT_EURUSD, mat, eur, usd)
        expected = SPOT_EURUSD * eur.df(mat) / usd.df(mat)
        assert fwd == pytest.approx(expected, rel=1e-10)

    def test_cip_gbpusd(self):
        gbp = _flat_curve(REF, rate=0.045)
        usd = _flat_curve(REF, rate=0.05)
        mat = REF + relativedelta(years=3)
        fwd = FXForward.forward_rate(SPOT_GBPUSD, mat, gbp, usd)
        expected = SPOT_GBPUSD * gbp.df(mat) / usd.df(mat)
        assert fwd == pytest.approx(expected, rel=1e-10)

    def test_triangular_consistency(self):
        """EUR/USD and GBP/USD forwards should be consistent with EUR/GBP."""
        eur = _flat_curve(REF, rate=0.04)
        gbp = _flat_curve(REF, rate=0.045)
        usd = _flat_curve(REF, rate=0.05)
        mat = REF + relativedelta(years=1)

        fwd_eurusd = FXForward.forward_rate(SPOT_EURUSD, mat, eur, usd)
        fwd_gbpusd = FXForward.forward_rate(SPOT_GBPUSD, mat, gbp, usd)
        fwd_eurgbp_implied = fwd_eurusd / fwd_gbpusd

        spot_eurgbp = SPOT_EURUSD / SPOT_GBPUSD
        fwd_eurgbp_direct = FXForward.forward_rate(spot_eurgbp, mat, eur, gbp)

        assert fwd_eurgbp_implied == pytest.approx(fwd_eurgbp_direct, rel=1e-6)


class TestFXSwapAtFair:
    """FX swap at fair forward rates has PV = 0."""

    def test_eurusd_swap_pv_zero(self):
        eur = _flat_curve(REF, rate=0.04)
        usd = _flat_curve(REF, rate=0.05)
        near = REF + relativedelta(months=1)
        far = REF + relativedelta(years=1)
        near_rate = FXForward.forward_rate(SPOT_EURUSD, near, eur, usd)
        far_rate = FXForward.forward_rate(SPOT_EURUSD, far, eur, usd)
        swap = FXSwap(EURUSD, near, far, near_rate, far_rate)
        assert swap.pv(SPOT_EURUSD, eur, usd) == pytest.approx(0.0, abs=1.0)

    def test_gbpusd_swap_pv_zero(self):
        gbp = _flat_curve(REF, rate=0.045)
        usd = _flat_curve(REF, rate=0.05)
        near = REF + relativedelta(months=3)
        far = REF + relativedelta(years=2)
        near_rate = FXForward.forward_rate(SPOT_GBPUSD, near, gbp, usd)
        far_rate = FXForward.forward_rate(SPOT_GBPUSD, far, gbp, usd)
        swap = FXSwap(GBPUSD, near, far, near_rate, far_rate)
        assert swap.pv(SPOT_GBPUSD, gbp, usd) == pytest.approx(0.0, abs=1.0)


class TestBasisCurveReprices:
    """Basis-adjusted curve reprices all market forwards."""

    def test_eurusd_with_negative_basis(self):
        eur = _flat_curve(REF, rate=0.04)
        usd = _flat_curve(REF, rate=0.05)

        # Synthetic market forwards with -15bp basis
        basis = -0.0015
        tenors = [0.5, 1.0, 2.0, 3.0, 5.0]
        forwards = []
        for t in tenors:
            mat = date.fromordinal(REF.toordinal() + int(t * 365))
            df_eur = eur.df(mat)
            usd_z = usd.zero_rate(mat)
            df_usd_adj = math.exp(-(usd_z + basis) * t)
            forwards.append((mat, SPOT_EURUSD * df_eur / df_usd_adj))

        adj = bootstrap_basis_curve(REF, SPOT_EURUSD, forwards, eur, usd)

        for mat, fwd_market in forwards:
            fwd_repriced = FXForward.forward_rate(SPOT_EURUSD, mat, eur, adj)
            assert fwd_repriced == pytest.approx(fwd_market, rel=1e-6)

    def test_implied_basis_round_trips(self):
        """Implied basis from market forwards should be consistent."""
        eur = _flat_curve(REF, rate=0.04)
        usd = _flat_curve(REF, rate=0.05)
        target_basis = -0.002  # -20bp

        mat = REF + relativedelta(years=2)
        t = 2.0
        df_eur = eur.df(mat)
        usd_z = usd.zero_rate(mat)
        fwd_market = SPOT_EURUSD * df_eur / math.exp(-(usd_z + target_basis) * t)

        recovered_basis = implied_basis_spread(SPOT_EURUSD, mat, fwd_market, eur, usd)
        assert recovered_basis == pytest.approx(target_basis, abs=1e-4)
