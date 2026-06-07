"""Tests for pricebook.models.g2pp_calibration (G2++ Hull-White calibration)."""

from datetime import date

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.g2pp_calibration import (
    g2pp_swaption_price,
    g2pp_implied_vol,
    calibrate_g2pp,
)

REF = date(2024, 1, 15)
curve = DiscountCurve.flat(REF, 0.04)

# Standard G2++ parameters used across tests
A = 0.05
B = 0.08
S1 = 0.01
S2 = 0.008
RHO = -0.7

# ATM forward swap rate on a flat 4% curve ≈ 4%
ATM_STRIKE = 0.04

SWAPTION_VOLS = {
    (1, 5): 0.15,
    (5, 5): 0.12,
    (1, 10): 0.14,
    (5, 10): 0.11,
}


# ---------------------------------------------------------------------------
# g2pp_swaption_price
# ---------------------------------------------------------------------------


def test_swaption_price_atm_positive():
    """ATM payer swaption price must be strictly positive."""
    price = g2pp_swaption_price(A, B, S1, S2, RHO, curve,
                                expiry_years=1.0, tenor_years=5.0,
                                strike=ATM_STRIKE, is_payer=True)
    assert price > 0.0


def test_swaption_payer_receiver_put_call_parity():
    """Payer + receiver must equal the forward swap value (put-call parity)."""
    exp, tenor, strike = 2.0, 5.0, ATM_STRIKE
    payer = g2pp_swaption_price(A, B, S1, S2, RHO, curve, exp, tenor,
                                strike, is_payer=True)
    receiver = g2pp_swaption_price(A, B, S1, S2, RHO, curve, exp, tenor,
                                   strike, is_payer=False)

    # Forward swap value: (P(exp) - P(exp+tenor)) for ATM is small but finite
    from pricebook.core.day_count import date_from_year_fraction
    pay_times = [exp + k for k in range(1, int(round(tenor)) + 1)]
    annuity = sum(curve.df(date_from_year_fraction(REF, t)) for t in pay_times)
    df_start = curve.df(date_from_year_fraction(REF, exp))
    df_end = curve.df(date_from_year_fraction(REF, exp + tenor))
    fwd_swap_pv = (df_start - df_end) - strike * annuity

    # payer - receiver = forward value
    assert (payer - receiver) == pytest.approx(fwd_swap_pv, abs=2e-4)


def test_swaption_price_short_expiry_positive():
    """Swaption price with 6m expiry / 2y tenor should be positive."""
    price = g2pp_swaption_price(A, B, S1, S2, RHO, curve,
                                expiry_years=0.5, tenor_years=2.0,
                                strike=ATM_STRIKE, is_payer=True)
    assert price > 0.0


def test_swaption_price_long_expiry_positive():
    """Swaption price with 10y expiry / 10y tenor should be positive."""
    price = g2pp_swaption_price(A, B, S1, S2, RHO, curve,
                                expiry_years=10.0, tenor_years=10.0,
                                strike=ATM_STRIKE, is_payer=True)
    assert price > 0.0


# ---------------------------------------------------------------------------
# g2pp_implied_vol
# ---------------------------------------------------------------------------


def test_implied_vol_positive():
    """Implied vol should be a positive number for reasonable ATM swaption."""
    iv = g2pp_implied_vol(A, B, S1, S2, RHO, curve,
                          expiry_years=1.0, tenor_years=5.0,
                          strike=ATM_STRIKE)
    assert iv > 0.0


def test_implied_vol_in_plausible_range():
    """Implied vol should be in (0, 1) — typical swaption vols are 5%–50%."""
    iv = g2pp_implied_vol(A, B, S1, S2, RHO, curve,
                          expiry_years=2.0, tenor_years=5.0,
                          strike=ATM_STRIKE)
    assert 0.0 < iv < 1.0


# ---------------------------------------------------------------------------
# calibrate_g2pp
# ---------------------------------------------------------------------------


def test_calibrate_g2pp_converged():
    """calibrate_g2pp should report converged=True for reasonable inputs."""
    result = calibrate_g2pp(curve, SWAPTION_VOLS)
    assert result.converged is True


def test_calibrate_g2pp_rmse_threshold():
    """Calibrated RMSE should be below 5% in Black-vol space (50 bp) for 2×2 grid."""
    result = calibrate_g2pp(curve, SWAPTION_VOLS)
    # RMSE in vol units; 0.05 = 500 bp is an extremely loose bound
    assert result.rmse_vol < 0.05
