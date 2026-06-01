"""Synthetic curve market data for all 33 currencies.

Generates realistic deposit + swap inputs for testing curve construction
without external data feeds.

    from pricebook.curves.synthetic_market_data import synthetic_curve_inputs

    deps, swaps = synthetic_curve_inputs("USD", ref_date)
    curve = build_curves("USD", ref_date, deps, swaps, method="nelson_siegel")

References:
    Realistic rates as of late 2024.
"""

from __future__ import annotations

from datetime import date
from dateutil.relativedelta import relativedelta


# ═══════════════════════════════════════════════════════════════
# Per-currency base rates and slopes
# ═══════════════════════════════════════════════════════════════

_MARKET_RATES = {
    # G10
    "USD": (0.050, 0.002),   # 5.0% base, +2bp/year slope
    "EUR": (0.030, 0.003),   # 3.0%
    "GBP": (0.045, 0.002),   # 4.5%
    "JPY": (0.001, 0.003),   # 0.1% (near zero)
    "CHF": (0.012, 0.002),   # 1.2%
    "CAD": (0.042, 0.002),   # 4.2%
    "AUD": (0.043, 0.002),   # 4.3%
    "NZD": (0.050, 0.001),   # 5.0%
    # Nordics
    "SEK": (0.035, 0.002),   # 3.5%
    "NOK": (0.045, 0.001),   # 4.5%
    "DKK": (0.033, 0.003),   # 3.3% (pegged near EUR)
    # LatAm
    "BRL": (0.110, -0.005),  # 11.0%, inverted
    "MXN": (0.105, -0.003),  # 10.5%, inverted
    "CLP": (0.055, -0.002),  # 5.5%
    "COP": (0.097, -0.003),  # 9.7%
    "PEN": (0.052, 0.002),   # 5.2%
    "ARS": (0.400, -0.010),  # 40% extreme
    # CEE
    "PLN": (0.058, -0.001),  # 5.8%
    "CZK": (0.045, -0.001),  # 4.5%
    "HUF": (0.065, -0.002),  # 6.5%
    "TRY": (0.450, -0.010),  # 45% extreme
    # Asia
    "CNY": (0.018, 0.002),   # 1.8%
    "KRW": (0.035, 0.001),   # 3.5%
    "INR": (0.065, -0.001),  # 6.5%
    "SGD": (0.035, 0.001),   # 3.5%
    "HKD": (0.040, 0.001),   # 4.0% (pegged to USD)
    "THB": (0.025, 0.002),   # 2.5%
    "IDR": (0.060, -0.001),  # 6.0%
    "MYR": (0.030, 0.001),   # 3.0%
    "PHP": (0.055, -0.001),  # 5.5%
    # Other
    "ZAR": (0.082, -0.002),  # 8.2%
    "ILS": (0.045, 0.001),   # 4.5%
}


def synthetic_curve_inputs(
    currency: str,
    reference_date: date,
) -> tuple[list[tuple[date, float]], list[tuple[date, float]]]:
    """Generate synthetic deposit + swap inputs for a currency.

    Returns (deposits, swaps) suitable for `build_curves()` or `bootstrap()`.

    Args:
        currency: ISO currency code.
        reference_date: valuation date.

    Returns:
        (deposits, swaps) where:
        - deposits: [(maturity_date, rate), ...] for 1M, 3M, 6M.
        - swaps: [(maturity_date, rate), ...] for 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y.
    """
    ccy = currency.upper()
    base, slope = _MARKET_RATES.get(ccy, (0.04, 0.002))

    # Deposits (short end) — slightly below base at front end
    deposit_tenors = [1, 3, 6]  # months
    deposits = []
    for m in deposit_tenors:
        mat = reference_date + relativedelta(months=m)
        rate = base - 0.002 * (1 - m / 12)  # starts slightly below base
        deposits.append((mat, max(rate, -0.01)))

    # Swaps (long end)
    swap_tenors = [1, 2, 3, 5, 7, 10, 15, 20, 30]  # years
    swaps = []
    for y in swap_tenors:
        mat = reference_date + relativedelta(years=y)
        rate = base + slope * y
        swaps.append((mat, rate))

    return deposits, swaps


def list_synthetic_currencies() -> list[str]:
    """Return all currencies with synthetic market data."""
    return sorted(_MARKET_RATES.keys())
