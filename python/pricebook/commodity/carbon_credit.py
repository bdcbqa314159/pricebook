"""Carbon and emission credit pricing: EU ETS, voluntary carbon markets.

Covers European Union Allowances (EUAs), Certified Emission Reductions (CERs),
California Carbon Allowances (CCAs), and voluntary market credits (VCS, Gold Standard).

* :class:`CarbonFuturesResult` — cost-of-carry futures pricing result.
* :class:`CarbonOptionResult` — Black-76 option pricing result with Greeks.
* :func:`carbon_futures_price` — fair value of carbon allowance futures.
* :func:`carbon_option_price` — Black-76 option price on carbon futures.
* :func:`marginal_abatement_cost` — equilibrium price from abatement supply curve.
* :func:`compliance_value` — net value of a compliance position.
* :func:`carbon_spread` — spread between two carbon market prices.
* :func:`voluntary_credit_discount` — haircut model for voluntary credits.

References:
    Carmona, R., Fehr, M., Hinz, J., & Porchet, A. (2009). Market design for
        emission trading schemes. *SIAM Review*, 51(3), 465–521.
    Hintermann, B. (2010). Allowance price drivers in the first phase of the
        EU ETS. *Journal of Environmental Economics and Management*, 59(1), 43–56.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.models.black76 import (
    OptionType,
    black76_price,
    black76_delta,
    black76_gamma,
    black76_vega,
    black76_theta,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CarbonFuturesResult:
    """Result of carbon allowance futures pricing."""
    fair_value: float         # futures fair value (€/ton or $/ton)
    convenience_yield: float  # implied convenience yield
    carry_cost: float         # financing + storage cost over tenor
    spot_equivalent: float    # PV of futures (discounted fair value)

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class CarbonOptionResult:
    """Black-76 option pricing result for carbon futures."""
    price: float    # option premium
    delta: float    # dPrice/dFutures
    gamma: float    # d²Price/dFutures²
    vega: float     # dPrice/dVol (per 1% move in vol)
    theta: float    # dPrice/dTime (per calendar day)

    def to_dict(self) -> dict:
        return vars(self)


# ---------------------------------------------------------------------------
# Carbon futures pricing
# ---------------------------------------------------------------------------

def carbon_futures_price(
    spot: float,
    rate: float,
    storage_cost: float,
    convenience_yield: float,
    T: float,
) -> CarbonFuturesResult:
    """Fair value of a carbon allowance futures contract (cost of carry).

    EUA futures pricing follows the standard commodity cost-of-carry model:

        F = S × exp((r + u − y) × T)

    where *u* is the storage/registry cost and *y* is the convenience yield.
    For carbon allowances the storage cost is typically negligible (registry fee
    only) and convenience yield captures the regulatory compliance option value.

    Args:
        spot: current spot price (€/ton CO₂).
        rate: risk-free rate (continuously compounded, annual).
        storage_cost: annual registry/holding cost as fraction of spot.
        convenience_yield: annual convenience yield as fraction of spot.
        T: time to futures expiry in years.

    Returns:
        :class:`CarbonFuturesResult` with fair value and decomposition.
    """
    carry = rate + storage_cost - convenience_yield
    fair_value = spot * math.exp(carry * T)
    carry_cost = spot * (math.exp(carry * T) - 1.0)
    df = math.exp(-rate * T)
    spot_equivalent = fair_value * df

    return CarbonFuturesResult(
        fair_value=fair_value,
        convenience_yield=convenience_yield,
        carry_cost=carry_cost,
        spot_equivalent=spot_equivalent,
    )


# ---------------------------------------------------------------------------
# Carbon option pricing
# ---------------------------------------------------------------------------

def carbon_option_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: str = "call",
) -> CarbonOptionResult:
    """Black-76 option price on carbon futures.

    Treats the carbon futures as the underlying; the futures price equals
    *spot* here (caller should pass the futures price directly for accuracy).
    Discount factor df = exp(−r × T).

    Args:
        spot: futures price (or spot as proxy) in €/ton.
        strike: option strike price.
        rate: risk-free rate (annual, continuous).
        vol: implied volatility (annual).
        T: time to expiry in years.
        option_type: ``"call"`` or ``"put"``.

    Returns:
        :class:`CarbonOptionResult` with price and Greeks.
    """
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if vol <= 0:
        raise ValueError("vol must be positive")

    ot = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT
    df = math.exp(-rate * T)

    price = black76_price(spot, strike, vol, T, df, ot)
    delta = black76_delta(spot, strike, vol, T, df, ot)
    gamma = black76_gamma(spot, strike, vol, T, df)
    vega = black76_vega(spot, strike, vol, T, df) * 0.01   # per 1% vol
    theta = black76_theta(spot, strike, vol, T, df, ot) / 365.0  # per day

    return CarbonOptionResult(price=price, delta=delta, gamma=gamma,
                              vega=vega, theta=theta)


# ---------------------------------------------------------------------------
# Marginal abatement cost
# ---------------------------------------------------------------------------

def marginal_abatement_cost(
    current_price: float,
    abatement_curve: list[tuple[float, float]],
) -> dict:
    """Find equilibrium abatement level at the current carbon price.

    The abatement supply curve is a list of ``(cost_per_ton, quantity_mtons)``
    pairs sorted by ascending cost. Technologies with cost ≤ current_price are
    economically viable; the equilibrium quantity is the cumulative abatement
    from all viable technologies.

    Args:
        current_price: current carbon allowance price (€/ton).
        abatement_curve: list of ``(cost, quantity)`` pairs representing
            discrete abatement technologies ordered by ascending cost.

    Returns:
        dict with keys ``equilibrium_quantity``, ``viable_technologies``,
        ``marginal_technology_cost``, and ``total_abatement_value``.
    """
    sorted_curve = sorted(abatement_curve, key=lambda x: x[0])
    equilibrium_quantity = 0.0
    viable = 0
    marginal_cost = 0.0

    for cost, qty in sorted_curve:
        if cost <= current_price:
            equilibrium_quantity += qty
            viable += 1
            marginal_cost = cost
        else:
            break

    total_value = equilibrium_quantity * current_price

    return {
        "equilibrium_quantity": equilibrium_quantity,
        "viable_technologies": viable,
        "marginal_technology_cost": marginal_cost,
        "total_abatement_value": total_value,
    }


# ---------------------------------------------------------------------------
# Compliance position value
# ---------------------------------------------------------------------------

def compliance_value(
    allowances_held: float,
    emissions: float,
    spot_price: float,
    penalty_per_ton: float,
) -> dict:
    """Value of a compliance position at current spot price.

    A regulated entity holds allowances and has a known emissions obligation.
    Surplus allowances can be sold at spot; a deficit is covered by buying
    allowances or paying the statutory penalty (whichever is cheaper).

    Args:
        allowances_held: EUAs or CCAs held (tons CO₂).
        emissions: actual or projected emissions (tons CO₂).
        spot_price: current allowance spot price (€/ton).
        penalty_per_ton: statutory penalty for non-compliance (€/ton).

    Returns:
        dict with ``position`` (surplus/deficit), ``position_value``,
        ``compliance_cost``, and ``net_value``.
    """
    position = allowances_held - emissions
    if position >= 0:
        # Surplus: can sell or bank allowances
        position_value = position * spot_price
        compliance_cost = 0.0
    else:
        # Deficit: buy at spot or pay penalty (take cheaper option)
        effective_cost = min(spot_price, penalty_per_ton)
        position_value = position * effective_cost   # negative
        compliance_cost = abs(position) * effective_cost

    net_value = allowances_held * spot_price - compliance_cost

    return {
        "position": position,
        "position_value": position_value,
        "compliance_cost": compliance_cost,
        "net_value": net_value,
    }


# ---------------------------------------------------------------------------
# Cross-market spread
# ---------------------------------------------------------------------------

def carbon_spread(
    eua_price: float,
    cca_price: float,
    fx_rate: float = 1.0,
) -> dict:
    """Spread between EU ETS (EUA) and California (CCA) carbon markets.

    Converts CCA price to EUR using *fx_rate* (USD/EUR), then computes
    the absolute and percentage spread. A positive spread means EUAs are
    trading at a premium to CCAs.

    Args:
        eua_price: EUA price in EUR/ton.
        cca_price: CCA price in USD/ton.
        fx_rate: USD per EUR (e.g. 1.08 means 1 EUR = 1.08 USD).

    Returns:
        dict with ``eua_price``, ``cca_price_eur``, ``spread_eur``,
        and ``spread_pct``.
    """
    cca_eur = cca_price / fx_rate
    spread = eua_price - cca_eur
    spread_pct = spread / cca_eur * 100.0 if cca_eur != 0 else float("nan")

    return {
        "eua_price": eua_price,
        "cca_price_eur": cca_eur,
        "spread_eur": spread,
        "spread_pct": spread_pct,
    }


# ---------------------------------------------------------------------------
# Voluntary credit discount
# ---------------------------------------------------------------------------

# Project type base haircuts (fraction of registry price)
_PROJECT_TYPE_HAIRCUTS: dict[str, float] = {
    "forestry":       0.30,   # REDD+, afforestation
    "renewable":      0.15,   # wind, solar, hydro
    "cookstove":      0.25,   # clean cooking
    "methane":        0.10,   # landfill, livestock
    "soil":           0.35,   # agricultural sequestration
    "direct_air":     0.05,   # DAC — high permanence
    "other":          0.20,
}

_VINTAGE_DECAY_RATE = 0.03   # 3% additional discount per year of age


def voluntary_credit_discount(
    registry_price: float,
    vintage_years: int,
    project_type: str,
    is_verified: bool = True,
) -> dict:
    """Estimate fair value haircut for voluntary carbon credits.

    Combines a project-type base haircut, a vintage age decay (older credits
    trade at a discount due to concerns over additionality and permanence),
    and an unverified credit penalty.

    Args:
        registry_price: listed registry price per ton (USD).
        vintage_years: age of credit in years from issuance to today.
        project_type: one of ``"forestry"``, ``"renewable"``, ``"cookstove"``,
            ``"methane"``, ``"soil"``, ``"direct_air"``, or ``"other"``.
        is_verified: ``True`` if credit holds third-party verification
            (VCS, Gold Standard, etc.).

    Returns:
        dict with ``registry_price``, ``total_haircut``, ``fair_value``,
        ``type_haircut``, ``vintage_haircut``, and ``verification_haircut``.
    """
    type_haircut = _PROJECT_TYPE_HAIRCUTS.get(project_type.lower(), 0.20)
    vintage_haircut = min(_VINTAGE_DECAY_RATE * vintage_years, 0.50)
    verification_haircut = 0.0 if is_verified else 0.20

    total_haircut = min(type_haircut + vintage_haircut + verification_haircut, 0.90)
    fair_value = registry_price * (1.0 - total_haircut)

    return {
        "registry_price": registry_price,
        "total_haircut": total_haircut,
        "fair_value": fair_value,
        "type_haircut": type_haircut,
        "vintage_haircut": vintage_haircut,
        "verification_haircut": verification_haircut,
    }
