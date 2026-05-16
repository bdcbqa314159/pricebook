"""Advanced duration hedging: key rate immunisation, barbell/bullet, LDI.

* :func:`key_rate_immunise` — solve for hedge weights at each pillar.
* :func:`barbell_bullet_analysis` — convexity pick-up comparison.
* :func:`ldi_cashflow_match` — liability-driven cash-flow matching.
* :func:`cross_currency_duration_hedge` — FX-adjusted hedge ratio.

References:
    Reitano, *Non-Parallel Yield Curve Shifts*, JPI, 1992.
    Litterman & Scheinkman, *Common Factors Affecting Bond Returns*, JFI, 1991.
    Martellini, Priaulet & Priaulet, *Fixed-Income Securities*, Wiley, 2003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ImmunisationResult:
    """Key rate immunisation result."""
    hedge_weights: np.ndarray       # weight per hedge instrument
    residual_krd: np.ndarray        # residual KRD after hedging
    max_residual: float
    n_instruments: int


def key_rate_immunise(
    portfolio_krd: list[float],
    hedge_instruments_krd: list[list[float]],
) -> ImmunisationResult:
    """Solve for hedge weights to immunise all key rate durations.

    min ||portfolio_krd + Σ w_i × instrument_krd_i||²

    Args:
        portfolio_krd: (n_pillars,) KRD of the portfolio.
        hedge_instruments_krd: list of (n_pillars,) KRD vectors per instrument.
    """
    p = np.array(portfolio_krd)
    A = np.array(hedge_instruments_krd).T   # (n_pillars, n_instruments)

    # Solve: A w ≈ -p via least squares
    w, _, _, _ = np.linalg.lstsq(A, -p, rcond=None)

    residual = p + A @ w

    return ImmunisationResult(
        hedge_weights=w,
        residual_krd=residual,
        max_residual=float(np.abs(residual).max()),
        n_instruments=len(hedge_instruments_krd),
    )


@dataclass
class BarbellBulletResult:
    """Barbell vs bullet comparison."""
    barbell_dv01: float
    bullet_dv01: float
    barbell_convexity: float
    bullet_convexity: float
    convexity_pickup: float         # barbell − bullet (positive = barbell wins)
    yield_pickup_bps: float         # bullet − barbell yield (typically positive)


def barbell_bullet_analysis(
    short_dv01: float,
    long_dv01: float,
    short_convexity: float,
    long_convexity: float,
    short_yield_pct: float,
    long_yield_pct: float,
    bullet_dv01: float,
    bullet_convexity: float,
    bullet_yield_pct: float,
) -> BarbellBulletResult:
    """Barbell (short + long) vs bullet comparison.

    Barbell: combine short and long bonds, DV01-neutral to bullet.
    Short_weight × short_dv01 + long_weight × long_dv01 = bullet_dv01.

    Key insight: barbell has higher convexity (convexity is convex in maturity),
    but lower yield (negative carry vs bullet).

    Args:
        short/long: barbell wings.
        bullet: intermediate maturity being compared.
    """
    # Solve for barbell weights: w_s × DV01_s + w_l × DV01_l = bullet DV01
    # constraint: w_s + w_l = 1
    if abs(long_dv01 - short_dv01) > 1e-10:
        w_l = (bullet_dv01 - short_dv01) / (long_dv01 - short_dv01)
        w_s = 1 - w_l
    else:
        w_s = w_l = 0.5

    barbell_dv01_check = w_s * short_dv01 + w_l * long_dv01
    barbell_conv = w_s * short_convexity + w_l * long_convexity
    barbell_yield = w_s * short_yield_pct + w_l * long_yield_pct

    conv_pickup = barbell_conv - bullet_convexity
    yield_pickup = (bullet_yield_pct - barbell_yield) * 100

    return BarbellBulletResult(
        barbell_dv01=float(barbell_dv01_check),
        bullet_dv01=float(bullet_dv01),
        barbell_convexity=float(barbell_conv),
        bullet_convexity=float(bullet_convexity),
        convexity_pickup=float(conv_pickup),
        yield_pickup_bps=float(yield_pickup),
    )


@dataclass
class LDIMatchResult:
    """LDI cash-flow matching result."""
    pv_liabilities: float
    pv_assets: float
    surplus: float                  # assets − liabilities
    mismatch_per_bucket: np.ndarray
    max_mismatch: float


def ldi_cashflow_match(
    liability_cashflows: list[tuple[float, float]],   # (time, amount)
    asset_cashflows: list[tuple[float, float]],       # (time, amount)
    rate: float,
) -> LDIMatchResult:
    """Liability-driven investment cash-flow matching.

    For each time bucket, compare asset and liability cash flows.
    PV of surplus should be ≈ 0 for a well-matched portfolio.

    Args:
        liability_cashflows: scheduled pension/insurance payments.
        asset_cashflows: bond portfolio cash flows.
        rate: discount rate.
    """
    # Build time grid from both
    all_times = sorted(set(t for t, _ in liability_cashflows) |
                       set(t for t, _ in asset_cashflows))

    pv_liab = 0.0
    pv_asset = 0.0
    mismatches = []

    for t in all_times:
        liab_cf = sum(a for ti, a in liability_cashflows if ti == t)
        asset_cf = sum(a for ti, a in asset_cashflows if ti == t)
        df = math.exp(-rate * t)
        pv_liab += liab_cf * df
        pv_asset += asset_cf * df
        mismatches.append(asset_cf - liab_cf)

    mismatch_arr = np.array(mismatches)

    return LDIMatchResult(
        pv_liabilities=float(pv_liab),
        pv_assets=float(pv_asset),
        surplus=float(pv_asset - pv_liab),
        mismatch_per_bucket=mismatch_arr,
        max_mismatch=float(np.abs(mismatch_arr).max()) if len(mismatch_arr) > 0 else 0.0,
    )


@dataclass
class CrossCurrencyHedgeResult:
    """Cross-currency duration hedge."""
    domestic_dv01: float
    foreign_dv01: float
    fx_rate: float
    hedge_ratio: float              # foreign notional per domestic notional
    fx_adjusted_hedge: float


def cross_currency_duration_hedge(
    domestic_dv01: float,
    foreign_dv01: float,
    fx_rate: float,
    fx_vol: float = 0.0,
    correlation: float = 0.0,
) -> CrossCurrencyHedgeResult:
    """Cross-currency duration hedge ratio.

    Hedge domestic DV01 using foreign bonds:
        w = -domestic_DV01 / (foreign_DV01 × FX)

    Adjusted for FX correlation (Quanto effect):
        w_adj = w × (1 + ρ × σ_FX × T) ≈ w

    Args:
        domestic_dv01: DV01 of domestic portfolio.
        foreign_dv01: DV01 of foreign hedge instrument.
        fx_rate: FX rate (foreign per domestic).
        fx_vol: FX volatility (for correlation adjustment).
        correlation: rate-FX correlation.
    """
    if abs(foreign_dv01 * fx_rate) < 1e-10:
        return CrossCurrencyHedgeResult(domestic_dv01, foreign_dv01, fx_rate, 0.0, 0.0)

    hedge_ratio = -domestic_dv01 / (foreign_dv01 * fx_rate)
    fx_adj = hedge_ratio * (1 + correlation * fx_vol)

    return CrossCurrencyHedgeResult(
        domestic_dv01=float(domestic_dv01),
        foreign_dv01=float(foreign_dv01),
        fx_rate=float(fx_rate),
        hedge_ratio=float(hedge_ratio),
        fx_adjusted_hedge=float(fx_adj),
    )
