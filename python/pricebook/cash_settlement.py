"""Cash-settlement utilities for swaptions and structured products.

    from pricebook.cash_settlement import cash_annuity

References:
    Pucci, M. (2012b). Pricing Index-Linked Hybrids. SSRN 2056277, Eq 3.
"""

from __future__ import annotations


def cash_annuity(
    swap_rate: float,
    year_fractions: list[float],
    times_to_payment: list[float],
) -> float:
    """Cash annuity for cash-settled swaptions (Pucci 2012b, Eq 3).

    Â(S) = Σ y_i / (1 + y_i S)^{yf(T, T_i)}

    Flat-curve proxy: replaces market discounts with synthetic discounts
    derived from a single yield S. Â is a deterministic function of S alone.

    Args:
        swap_rate: the swap rate S.
        year_fractions: y_i for each coupon period.
        times_to_payment: yf(T, T_i) for each T_i (from fixing T to payment).
    """
    total = 0.0
    for yi, tau_i in zip(year_fractions, times_to_payment):
        denom = 1 + yi * swap_rate
        if denom <= 0:
            continue
        total += yi / denom ** tau_i
    return total
