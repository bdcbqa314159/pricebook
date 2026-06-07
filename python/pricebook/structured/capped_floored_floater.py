"""Capped, floored, and collar floating rate notes.

* :func:`floored_floater`           — FRN with minimum coupon (long floorlet strip).
* :func:`collar_floater`            — FRN with both cap and floor (short caplets, long floorlets).
* :func:`reverse_floater`           — fixed minus leveraged floating, floored at zero.
* :func:`inverse_floater_duration`  — effective duration of a reverse/inverse floater.

Pricing methodology
-------------------
Each optionlet (caplet or floorlet) is priced independently via the
Black-76 model on the relevant forward rate.  The FRN itself is priced at
par under the assumption that the floating rate resets to exactly the
forward rate at each period (textbook FRN identity: par between reset dates).
The option strip is added (floor) or subtracted (cap) from par.

    ``floored FRN  = par FRN  + floorlet strip``
    ``collar FRN   = par FRN  − caplet strip  + floorlet strip``
    ``reverse FRN  = fixed-rate bond − cap strip on floating``

References
----------
Fabozzi, F. J. (2005). *The Handbook of Fixed Income Securities*, 7th ed.
    McGraw-Hill.  Chapter 12: Floating-Rate Securities.
Tuckman, B. and Serrat, A. (2011). *Fixed Income Securities*, 3rd ed.
    Wiley.  Chapter 17: Caps, Floors and Swaptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.models.black76 import black76_price, OptionType


# ---------------------------------------------------------------------------
# Helper: Black-76 caplet / floorlet
# ---------------------------------------------------------------------------

def _caplet(forward: float, strike: float, vol: float, df: float, dt: float) -> float:
    """Black-76 caplet price (call on forward rate, notional = 1)."""
    return black76_price(forward, strike, vol, dt, df, OptionType.CALL) * dt


def _floorlet(forward: float, strike: float, vol: float, df: float, dt: float) -> float:
    """Black-76 floorlet price (put on forward rate, notional = 1)."""
    return black76_price(forward, strike, vol, dt, df, OptionType.PUT) * dt


def _plain_frn_price(
    forward_rates: list[float],
    discount_factors: list[float],
    dt: float,
    notional: float,
    spread: float,
) -> float:
    """Price of a plain floating-rate note with optional spread.

    Under the standard FRN identity, a floating-rate note prices at par
    between coupon reset dates.  Adding a fixed spread over floating raises
    the price above par by the PV of the spread strip.

    Returns the full price (par + spread PV).
    """
    n = len(forward_rates)
    spread_pv = sum(discount_factors[i] * spread * dt for i in range(n))
    principal_pv = discount_factors[-1] if discount_factors else 1.0
    # par FRN = principal PV + floating coupon PV (the two cancel to notional
    # only when the discount curve equals the forward curve).  We use the
    # textbook identity: par FRN price = notional.
    return notional * (1.0 + spread_pv)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CappedFlooredFloaterResult:
    """Result of a capped / floored floater pricing calculation."""

    price: float
    bond_component: float       # plain FRN component (par + spread PV)
    cap_value: float            # PV of embedded cap strip (positive = cost to holder)
    floor_value: float          # PV of embedded floor strip (positive = benefit to holder)
    collar_value: float         # floor_value - cap_value (net option value to holder)
    effective_spread: float     # approximate spread to Libor implied by option-adjusted price
    n_periods: int

    def to_dict(self) -> dict:
        return vars(self)


# ---------------------------------------------------------------------------
# 1. Floored floater
# ---------------------------------------------------------------------------

def floored_floater(
    forward_rates: list[float],
    floor_rate: float,
    vol: float,
    discount_factors: list[float],
    dt: float,
    notional: float = 100.0,
    spread: float = 0.0,
) -> CappedFlooredFloaterResult:
    """Price a floating rate note with a minimum coupon (floor).

    The holder of a floored FRN is long a strip of floorlets.  The floor
    guarantees a minimum coupon of ``floor_rate`` each period::

        coupon_i = max(forward_i + spread, floor_rate)
        price    = plain FRN + sum_i [ floorlet(forward_i, floor_rate) × notional ]

    Args:
        forward_rates:    list of length *n* of forward Libor rates, one per period.
        floor_rate:       minimum coupon rate (e.g. 0.02 for 2%).
        vol:              flat Black-76 lognormal vol for each floorlet.
        discount_factors: list of length *n* of period discount factors df(0, t_i).
        dt:               period length in years (e.g. 0.25 for quarterly).
        notional:         face value (default 100).
        spread:           fixed spread over floating added to each coupon.

    Returns:
        :class:`CappedFlooredFloaterResult`.
    """
    n = len(forward_rates)
    if len(discount_factors) < n:
        raise ValueError("discount_factors must have at least as many entries as forward_rates.")

    bond = _plain_frn_price(forward_rates, discount_factors, dt, notional, spread)

    floor_strip = sum(
        _floorlet(forward_rates[i] + spread, floor_rate, vol, discount_factors[i], dt)
        * notional
        for i in range(n)
    )

    price = bond + floor_strip
    eff_spread = (price / notional - 1.0) / (n * dt) if n * dt > 0 else 0.0

    return CappedFlooredFloaterResult(
        price=price,
        bond_component=bond,
        cap_value=0.0,
        floor_value=floor_strip,
        collar_value=floor_strip,
        effective_spread=eff_spread,
        n_periods=n,
    )


# ---------------------------------------------------------------------------
# 2. Collar floater
# ---------------------------------------------------------------------------

def collar_floater(
    forward_rates: list[float],
    cap_rate: float,
    floor_rate: float,
    vol: float,
    discount_factors: list[float],
    dt: float,
    notional: float = 100.0,
    spread: float = 0.0,
) -> CappedFlooredFloaterResult:
    """Price a collar floating rate note (cap + floor).

    The holder sells a caplet strip (capping upside) and buys a floorlet
    strip (guaranteeing a minimum coupon)::

        coupon_i = min(max(forward_i + spread, floor_rate), cap_rate)
        price    = plain FRN - cap strip + floor strip

    For a *zero-cost collar*, cap_rate and floor_rate are chosen so that
    the cap and floor strips have equal value.

    Args:
        cap_rate:   maximum coupon rate (holder is short cap).
        floor_rate: minimum coupon rate (holder is long floor).
        vol:        flat lognormal vol applied to both caps and floors.

    Returns:
        :class:`CappedFlooredFloaterResult`.
    """
    if cap_rate <= floor_rate:
        raise ValueError(f"cap_rate ({cap_rate}) must exceed floor_rate ({floor_rate}).")

    n = len(forward_rates)
    if len(discount_factors) < n:
        raise ValueError("discount_factors must have at least as many entries as forward_rates.")

    bond = _plain_frn_price(forward_rates, discount_factors, dt, notional, spread)

    cap_strip = sum(
        _caplet(forward_rates[i] + spread, cap_rate, vol, discount_factors[i], dt)
        * notional
        for i in range(n)
    )
    floor_strip = sum(
        _floorlet(forward_rates[i] + spread, floor_rate, vol, discount_factors[i], dt)
        * notional
        for i in range(n)
    )

    price = bond - cap_strip + floor_strip
    collar_val = floor_strip - cap_strip
    eff_spread = (price / notional - 1.0) / (n * dt) if n * dt > 0 else 0.0

    return CappedFlooredFloaterResult(
        price=price,
        bond_component=bond,
        cap_value=cap_strip,
        floor_value=floor_strip,
        collar_value=collar_val,
        effective_spread=eff_spread,
        n_periods=n,
    )


# ---------------------------------------------------------------------------
# 3. Reverse floater
# ---------------------------------------------------------------------------

def reverse_floater(
    fixed_rate: float,
    forward_rates: list[float],
    vol: float,
    discount_factors: list[float],
    dt: float,
    notional: float = 100.0,
    leverage: float = 1.0,
    floor_rate: float = 0.0,
) -> CappedFlooredFloaterResult:
    """Price a reverse (inverse) floater with an embedded floor.

    Coupon each period::

        coupon_i = max(fixed_rate - leverage × forward_i, floor_rate)

    The floor at ``floor_rate`` is equivalent to the issuer writing a
    cap on the floating rate::

        price = fixed-rate bond - leverage × cap strip on floating

    where the cap strike on Libor is ``(fixed_rate - floor_rate) / leverage``.

    The fixed-rate bond component is priced as::

        bond = notional × sum_i [ fixed_rate × dt × df_i ] + notional × df_n

    Args:
        fixed_rate:    fixed component of the reverse coupon (e.g. 0.10 for 10%).
        forward_rates: list of *n* forward Libor rates, one per period.
        vol:           flat Black-76 lognormal vol for cap on Libor.
        discount_factors: list of *n* period discount factors.
        dt:            period length in years.
        notional:      face value (default 100).
        leverage:      multiplier on floating rate (default 1.0).
        floor_rate:    minimum coupon (default 0).

    Returns:
        :class:`CappedFlooredFloaterResult`.
    """
    n = len(forward_rates)
    if len(discount_factors) < n:
        raise ValueError("discount_factors must have at least as many entries as forward_rates.")

    # Fixed-rate bond component
    coupon_pv = sum(discount_factors[i] * fixed_rate * dt for i in range(n))
    principal_pv = discount_factors[n - 1]
    bond = notional * (coupon_pv + principal_pv)

    # Cap strike on floating: floor_rate = fixed_rate - leverage × cap_strike
    cap_strike = (fixed_rate - floor_rate) / leverage if leverage > 0 else math.inf

    # Embedded cap strip (issuer is long, so the holder pays for it via lower coupon)
    cap_strip = sum(
        _caplet(forward_rates[i], cap_strike, vol, discount_factors[i], dt)
        * notional * leverage
        for i in range(n)
    )

    price = bond - cap_strip
    # Effective spread vs Libor (approximate)
    libor_par = notional  # par FRN benchmark
    eff_spread = (price - libor_par) / (notional * n * dt) if n * dt > 0 else 0.0

    return CappedFlooredFloaterResult(
        price=price,
        bond_component=bond,
        cap_value=cap_strip,
        floor_value=0.0,
        collar_value=-cap_strip,
        effective_spread=eff_spread,
        n_periods=n,
    )


# ---------------------------------------------------------------------------
# 4. Inverse floater duration
# ---------------------------------------------------------------------------

def inverse_floater_duration(
    fixed_rate: float,
    floating_rate: float,
    leverage: float,
    maturity: float,
) -> float:
    """Effective (modified) duration of a reverse / inverse floater.

    For a reverse floater with coupon ``C = fixed_rate - leverage × Libor``,
    the effective duration is *amplified* relative to an equivalent fixed-rate
    bond.  Using the decomposition:

        ``reverse FRN = (1 + leverage) × fixed-rate bond − leverage × FRN``

    and noting that the FRN has near-zero duration, we obtain:

        ``D_effective ≈ (1 + leverage) × D_fixed``

    where ``D_fixed`` is the modified duration of the fixed-rate bond priced
    at the current floating rate.  This is the standard Fabozzi approximation.

    Args:
        fixed_rate:    coupon rate on the fixed-rate bond component.
        floating_rate: current floating rate used to discount cash flows.
        leverage:      multiplier on the floating leg (e.g. 2.0).
        maturity:      maturity in years.

    Returns:
        Approximate effective duration in years (positive).

    Notes:
        A more precise estimate requires full cash-flow modelling and an
        OAS/option-adjusted spread calculation to strip the cap value.
    """
    if floating_rate <= 0.0 or maturity <= 0.0:
        raise ValueError("floating_rate and maturity must be positive.")

    # Approximate modified duration of a bullet fixed-rate bond
    # D_mod = [1 - (1 + r)^{-T}] / (r × (1 + (1-1/(1+r)^T)/r × c)) ... simplified:
    # Use Macaulay formula for annual coupon bond and convert to modified.
    r = floating_rate
    c = fixed_rate
    T = maturity

    if abs(c) < 1e-10:
        # Zero-coupon bond: D_mac = T
        d_mac = T
    else:
        # Macaulay duration of annual-coupon bond priced at r
        # D_mac = [sum_{t=1}^{T} t * c/(1+r)^t + T/(1+r)^T] / price
        # price  = c * [1-(1+r)^{-T}]/r + (1+r)^{-T}
        discount = (1.0 + r) ** (-T)
        annuity = (1.0 - discount) / r if r != 0 else T
        price = c * annuity + discount

        numerator = 0.0
        for t in range(1, int(T) + 1):
            numerator += t * c * (1.0 + r) ** (-t)
        numerator += T * (1.0 + r) ** (-T)

        d_mac = numerator / price if price > 0 else T

    d_mod_fixed = d_mac / (1.0 + r)

    # Amplified duration of the reverse floater
    return (1.0 + leverage) * d_mod_fixed
