"""Cross-currency (quanto) CDS pricing.

Adjustments for CDS traded in a currency different from the reference
entity's domestic currency, capturing wrong-way risk between FX and
credit.

* :class:`QuantoCDSResult` — pricing result with quanto adjustment.
* :func:`quanto_cds_spread` — adjusted spread for cross-currency CDS.
* :func:`price_quanto_cds` — full quanto CDS pricing.
* :func:`quanto_adjustment_factor` — convexity adjustment factor.

References:
    Ehlers & Schönbucher, *The Influence of FX Risk on Credit Spreads*,
    ETH Zürich Working Paper, 2006.
    Brigo, Pede & Petrelli, *Multi-currency credit default swaps*,
    Int. J. Theor. Appl. Finance, 2019.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve


@dataclass
class QuantoCDSResult:
    """Quanto CDS pricing result."""
    domestic_spread: float
    foreign_spread: float
    quanto_adjustment_bp: float
    fx_hedge_notional: float
    correlation: float

    def to_dict(self) -> dict:
        return vars(self)


# ---- Quanto adjustment ----

def quanto_adjustment_factor(
    fx_vol: float,
    credit_vol: float,
    correlation: float,
    T: float,
) -> float:
    """Convexity adjustment for quanto CDS.

    Factor: exp(rho * sigma_FX * sigma_credit * T).

    When rho > 0 (wrong-way risk: FX depreciates when credit worsens),
    the adjustment increases the spread.

    Args:
        fx_vol: annualised FX vol.
        credit_vol: annualised credit spread vol.
        correlation: FX-credit correlation rho.
        T: time to maturity in years.
    """
    return math.exp(correlation * fx_vol * credit_vol * T)


def quanto_cds_spread(
    foreign_spread: float,
    fx_vol: float,
    credit_vol: float,
    correlation: float,
    T: float,
) -> float:
    """Quanto-adjusted CDS spread.

    spread_quanto = foreign_spread * exp(rho * sigma_FX * sigma_credit * T).

    When rho > 0 (wrong-way: FX depreciates when credit worsens),
    quanto spread > foreign spread.

    Args:
        foreign_spread: CDS spread in foreign currency.
        fx_vol: annualised FX vol.
        credit_vol: annualised credit spread vol.
        correlation: FX-credit correlation.
        T: time to maturity in years.
    """
    adj = quanto_adjustment_factor(fx_vol, credit_vol, correlation, T)
    return foreign_spread * adj


def price_quanto_cds(
    reference_date: date,
    maturity: float,
    foreign_spread: float,
    domestic_discount: DiscountCurve,
    foreign_discount: DiscountCurve,
    survival_curve: SurvivalCurve,
    fx_spot: float,
    fx_vol: float,
    credit_vol: float,
    correlation: float,
    recovery: float = 0.4,
    notional: float = 1_000_000,
) -> QuantoCDSResult:
    """Full quanto CDS pricing.

    Computes domestic and foreign CDS spreads, the quanto adjustment
    in basis points, and the FX hedge notional.

    The FX hedge notional covers the expected loss on default:
    notional * (1 - recovery) * default_probability.

    Args:
        reference_date: valuation date.
        maturity: CDS maturity in years.
        foreign_spread: CDS spread in foreign currency.
        domestic_discount: domestic risk-free discount curve.
        foreign_discount: foreign risk-free discount curve.
        survival_curve: credit survival curve.
        fx_spot: FX spot rate (domestic per foreign).
        fx_vol: annualised FX vol.
        credit_vol: annualised credit spread vol.
        correlation: FX-credit correlation.
        recovery: recovery rate.
        notional: CDS notional.
    """
    # Quanto adjustment
    adj = quanto_adjustment_factor(fx_vol, credit_vol, correlation, maturity)
    domestic_spread = foreign_spread * adj
    quanto_bp = (domestic_spread - foreign_spread) * 10_000

    # Default probability from survival curve
    from datetime import timedelta
    mat_date = reference_date + timedelta(days=round(maturity * 365.25))
    surv = survival_curve.survival(mat_date)
    default_prob = 1.0 - surv

    # FX hedge notional: expected loss in foreign currency
    fx_hedge = notional * (1.0 - recovery) * default_prob

    return QuantoCDSResult(
        domestic_spread=domestic_spread,
        foreign_spread=foreign_spread,
        quanto_adjustment_bp=quanto_bp,
        fx_hedge_notional=fx_hedge,
        correlation=correlation,
    )
