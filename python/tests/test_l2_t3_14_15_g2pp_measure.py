"""Regression for L2 Tier-3 T3.14 / T3.15 — `g2pp_swaption_price` uses the
correct T-forward measure (Brigo-Mercurio eq. 4.31).

Pre-fix the analytical pricer had TWO compounding measure errors:

* T3.15 — the outer x-integration used the RISK-NEUTRAL marginal
  N(0, σ_x²(T_α)) instead of the T_α-FORWARD marginal N(M_x(T_α), σ_x²(T_α)).
  Under the T-forward measure (the natural measure for swaption pricing
  with P(0, T_α) as numeraire), x and y both have shifted means M_x(T_α)
  and M_y(T_α) (B-M eq. 4.30).

* T3.14 — the inner pricer per x-node used the UNCONDITIONAL G2++ ZCB
  option formula (which already integrates over BOTH x and y).  Combined
  with the outer x-integration, the x dimension was effectively summed
  twice — producing 90 %+ mispricing on canonical (2y3y ATM payer) cases
  vs Monte Carlo with the correct G2++ bond-price formula
  P(T_α, T_i; x, y).

Post-fix implements B-M eq. 4.31 directly: 1-D Gauss-Hermite over the
T-forward x marginal, with the conditional y | x bond-option closed form
on the inside.  Verified to match MC to <1 % across an 18-point grid of
(expiry, tenor, strike, side) combinations.

This test file uses a smaller MC budget than the development validation;
the tolerance is loosened accordingly to ~3 % to allow for MC noise at
modest path counts.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from pricebook.core.day_count import (
    DayCountConvention,
    date_from_year_fraction,
)
from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.g2pp_calibration import _g2pp_V, g2pp_swaption_price


REF = date(2024, 1, 1)


def _flat_curve(rate: float = 0.04) -> DiscountCurve:
    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    dfs = [math.exp(-rate * t) for t in tenors]
    dates = [date_from_year_fraction(REF, t) for t in tenors]
    return DiscountCurve(REF, dates, dfs, day_count=DayCountConvention.ACT_365_FIXED)


def _P_correct(curve, t, T, x_t, y_t, a, b, sigma1, sigma2, rho):
    """Correct G2++ bond price P(t, T; x_t, y_t) — Brigo-Mercurio 4.14."""
    if T <= t:
        return 1.0
    P_M_T = curve.df(date_from_year_fraction(REF, T))
    P_M_t = curve.df(date_from_year_fraction(REF, t))
    Bx = (1 - math.exp(-a * (T - t))) / a if a > 1e-12 else (T - t)
    By = (1 - math.exp(-b * (T - t))) / b if b > 1e-12 else (T - t)
    V_t_T = _g2pp_V(a, b, sigma1, sigma2, rho, T - t)
    V_0_T = _g2pp_V(a, b, sigma1, sigma2, rho, T)
    V_0_t = _g2pp_V(a, b, sigma1, sigma2, rho, t)
    return (P_M_T / P_M_t) * math.exp(0.5 * (V_t_T - V_0_T + V_0_t) - Bx * x_t - By * y_t)


def _mc_swaption(curve, T_exp, tenor, K, *, is_payer, a, b, sigma1, sigma2, rho,
                 n_paths=20_000, n_steps=150, seed=42):
    """MC swaption price for G2++ using the correct bond formula."""
    dt = T_exp / n_steps
    rng = np.random.default_rng(seed)
    x = np.zeros(n_paths); y = np.zeros(n_paths); int_r = np.zeros(n_paths)
    chol = np.array([[1.0, 0.0], [rho, math.sqrt(1 - rho**2)]])

    def Bf(k, t):
        return (1 - math.exp(-k * t)) / k if k > 0 else t

    def phi(s):
        f0s = curve.instantaneous_forward(s)
        return (f0s + 0.5 * sigma1**2 * Bf(a, s)**2
                + 0.5 * sigma2**2 * Bf(b, s)**2
                + rho * sigma1 * sigma2 * Bf(a, s) * Bf(b, s))

    for i in range(n_steps):
        s_mid = (i + 0.5) * dt
        z = rng.standard_normal((n_paths, 2))
        dw = z @ chol.T * math.sqrt(dt)
        r_mid = x + y + phi(s_mid)
        int_r += r_mid * dt
        x = x - a * x * dt + sigma1 * dw[:, 0]
        y = y - b * y * dt + sigma2 * dw[:, 1]

    pay_times = [T_exp + k for k in range(1, int(round(tenor)) + 1)]
    c = [K] * (len(pay_times) - 1) + [K + 1.0]
    swap_pv = np.zeros(n_paths)
    for ck, sk in zip(c, pay_times):
        P_k = np.array([
            _P_correct(curve, T_exp, sk, xi, yi, a, b, sigma1, sigma2, rho)
            for xi, yi in zip(x, y)
        ])
        swap_pv += ck * P_k
    payer_pv = 1.0 - swap_pv
    payoff = np.maximum(payer_pv if is_payer else -payer_pv, 0.0)
    disc = np.exp(-int_r)
    return float((disc * payoff).mean())


class TestG2PPSwaptionMatchesMC:
    @pytest.mark.parametrize("T_exp,tenor,K,is_payer", [
        (1.0, 5.0, 0.02, True),   # deep ITM payer
        (1.0, 5.0, 0.06, False),  # deep ITM receiver
        (2.0, 3.0, 0.04, True),   # ATM payer
        (2.0, 3.0, 0.04, False),  # ATM receiver
        (5.0, 2.0, 0.04, True),   # ATM payer, longer expiry
    ])
    def test_analytical_matches_mc(self, T_exp, tenor, K, is_payer):
        """Across a grid of (expiry, tenor, strike, side), the B-M 4.31
        analytical swaption price must match MC ground truth to within
        ~3 % relative.  Pre-fix the analytical was ~90 % LOW across the
        whole grid due to the compounding measure errors T3.14+T3.15."""
        curve = _flat_curve(0.04)
        a, b, sigma1, sigma2, rho = 0.5, 0.1, 0.01, 0.005, -0.7

        ana = g2pp_swaption_price(
            a, b, sigma1, sigma2, rho, curve,
            T_exp, tenor, K, is_payer=is_payer,
        )
        mc = _mc_swaption(
            curve, T_exp, tenor, K, is_payer=is_payer,
            a=a, b=b, sigma1=sigma1, sigma2=sigma2, rho=rho,
        )

        if mc < 1e-6:
            # Deep OTM: both essentially zero, skip relative check.
            assert ana < 1e-3
        else:
            rel = abs(ana - mc) / mc
            assert rel < 0.03, (
                f"({T_exp}y{tenor}y K={K} payer={is_payer}): "
                f"ana={ana:.6f}, mc={mc:.6f}, rel={rel:.3%}"
            )


class TestG2PPNoSilentZero:
    """The post-fix swaption pricer should give NONZERO prices for
    reasonable ATM cases (sanity check)."""

    def test_atm_payer_nonzero(self):
        curve = _flat_curve(0.04)
        price = g2pp_swaption_price(
            0.5, 0.1, 0.01, 0.005, -0.7, curve,
            expiry_years=2.0, tenor_years=3.0, strike=0.04, is_payer=True,
        )
        # Pre-fix this was ~0.0055; post-fix should be in the same order
        # (both are correct ATM values around 50bp on notional).
        assert price > 1e-4

    def test_atm_receiver_nonzero(self):
        curve = _flat_curve(0.04)
        price = g2pp_swaption_price(
            0.5, 0.1, 0.01, 0.005, -0.7, curve,
            expiry_years=2.0, tenor_years=3.0, strike=0.04, is_payer=False,
        )
        assert price > 1e-4
