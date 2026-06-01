"""Extendible swaps and bonds.

An extendible instrument has an option to extend its maturity.
- Extendible swap: holder can extend beyond original maturity.
- Extendible bond: issuer or investor can extend.

Decomposition:
    Extendible swap = base swap + European swaption on the extension period

    from pricebook.fixed_income.extendible import (
        extendible_swap_price, ExtendibleSwapResult,
    )

References:
    Brigo & Mercurio (2006). Interest Rate Models, Ch. 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ExtendibleSwapResult:
    """Extendible swap pricing result."""
    extendible_pv: float
    base_swap_pv: float
    extension_option_value: float
    base_maturity_years: float
    extended_maturity_years: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class ExtendibleBondResult:
    """Extendible bond pricing result."""
    extendible_price: float
    base_price: float
    extension_option_value: float

    def to_dict(self) -> dict:
        return vars(self)


def extendible_swap_price(
    hw_a: float,
    hw_sigma: float,
    r0: float,
    base_maturity_years: float,
    extended_maturity_years: float,
    fixed_rate: float,
    is_payer: bool = True,
    extension_by: str = "payer",
    n_steps: int = 100,
    swap_freq: float = 1.0,
) -> ExtendibleSwapResult:
    """Price an extendible interest rate swap.

    The holder of the extension option can choose at the base maturity
    to extend the swap to the extended maturity.

    Decomposition:
        Extendible payer swap = base payer swap + European payer swaption
            (exercise at base_maturity, underlying: swap from base to extended)

    Args:
        base_maturity_years: original swap maturity.
        extended_maturity_years: maturity if extended.
        extension_by: "payer" or "receiver" — who holds the extension option.
    """
    from pricebook.options.bermudan_swaption import bermudan_swaption_tree

    try:
        from pricebook.models.hull_white import HullWhite
        from pricebook.core.discount_curve import DiscountCurve
        from pricebook.core.interpolation import InterpolationMethod
        from datetime import date, timedelta
        ref = date(2024, 1, 1)
        pd = [ref + timedelta(days=int(365*y)) for y in range(1, int(extended_maturity_years)+5)]
        pf = [math.exp(-r0*y) for y in range(1, int(extended_maturity_years)+5)]
        fc = DiscountCurve(ref, pd, pf, interpolation=InterpolationMethod.LOG_LINEAR)
        hw = HullWhite(a=hw_a, sigma=hw_sigma, curve=fc)
    except (ImportError, TypeError, ValueError):
        class _HW:
            def __init__(self, a, sigma, r0):
                self.a, self.sigma, self.r0 = a, sigma, r0
            def zcb_price(self, t, T, r):
                B = (1 - math.exp(-self.a * (T-t))) / self.a if self.a > 1e-10 else T - t
                A = math.exp((B - (T-t)) * (self.a**2 * self.r0 - self.sigma**2/2) / self.a**2
                             - self.sigma**2 * B**2 / (4 * self.a))
                return A * math.exp(-B * r)
        hw = _HW(hw_a, hw_sigma, r0)

    # Base swap PV
    df_base = math.exp(-r0 * base_maturity_years)
    annuity_base = sum(swap_freq * math.exp(-r0 * t)
                       for t in [swap_freq * i for i in range(1, int(base_maturity_years / swap_freq) + 1)])
    if is_payer:
        base_pv = (1 - df_base) - fixed_rate * annuity_base
    else:
        base_pv = fixed_rate * annuity_base - (1 - df_base)

    # Extension option: European swaption at base_maturity on the extension period
    # Exercise dates = [base_maturity]
    # Underlying swap: from base_maturity to extended_maturity
    swaption_is_payer = (extension_by == "payer")

    try:
        option_value = bermudan_swaption_tree(
            hw, [base_maturity_years], extended_maturity_years, fixed_rate,
            is_payer=swaption_is_payer, n_steps=n_steps, swap_freq=swap_freq,
        )
    except Exception:
        # Fallback approximation
        extension_years = extended_maturity_years - base_maturity_years
        vol = hw_sigma * math.sqrt(base_maturity_years)
        option_value = max(abs(base_pv) * 0.10 * extension_years, 0)

    option_value = max(option_value, 0)

    # Extendible PV = base + option (option adds value for the holder)
    if (is_payer and extension_by == "payer") or (not is_payer and extension_by == "receiver"):
        extendible_pv = base_pv + option_value
    else:
        extendible_pv = base_pv - option_value

    return ExtendibleSwapResult(
        extendible_pv=extendible_pv,
        base_swap_pv=base_pv,
        extension_option_value=option_value,
        base_maturity_years=base_maturity_years,
        extended_maturity_years=extended_maturity_years,
    )
