"""Structured notes: capital-protected, dual digital, bonus certificate, participation note.

Common wealth management / private banking products:

* :func:`capital_protected_note` — zero-coupon bond + call option.
* :func:`dual_digital` — pays if TWO conditions met simultaneously.
* :func:`bonus_certificate` — down-and-in put + cap.
* :func:`participation_note` — capital protection + equity participation.

References:
    Bouzoubaa & Osseiran (2010). Exotic Options and Hybrids, Ch. 9-11.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.black76 import black76_price, OptionType


@dataclass
class CapitalProtectedResult:
    price: float
    bond_floor: float
    option_value: float
    participation: float
    protection_level: float
    def to_dict(self) -> dict:
        return vars(self)


def capital_protected_note(
    spot: float, rate: float, dividend_yield: float, vol: float, T: float,
    protection: float = 1.0, participation: float | None = None,
    notional: float = 1000,
) -> CapitalProtectedResult:
    """Capital-protected note: 100% principal + participation x upside.

    Structure = zero-coupon bond (protection) + participation x ATM call.
    If participation not given: solve for max participation at zero cost.
    """
    df = math.exp(-rate * T)
    bond_floor = protection * notional * df
    budget = notional - bond_floor  # available for option

    F = spot * math.exp((rate - dividend_yield) * T)
    call = black76_price(F, spot, vol, T, df, OptionType.CALL)

    if participation is None:
        denom = call * notional / spot if spot > 0 else 0.0
        participation = budget / denom if denom > 1e-10 else 0.0

    option_value = participation * call * notional / spot
    price = bond_floor + option_value

    return CapitalProtectedResult(float(price), float(bond_floor),
                                   float(option_value), float(participation), protection)


@dataclass
class DualDigitalResult:
    price: float
    prob_both: float
    payout: float
    def to_dict(self) -> dict:
        return vars(self)


def dual_digital(
    spot1: float, spot2: float, barrier1: float, barrier2: float,
    rate: float, div1: float, div2: float, vol1: float, vol2: float,
    correlation: float, T: float,
    is_above1: bool = True, is_above2: bool = True,
    payout: float = 1.0,
    n_paths: int = 20_000, seed: int | None = 42,
) -> DualDigitalResult:
    """Dual digital: pays if BOTH conditions met at expiry.

    Condition 1: S1(T) >= barrier1 (or <= if is_above1=False).
    Condition 2: S2(T) >= barrier2 (or <= if is_above2=False).
    Payoff = payout if both conditions true, else 0.
    """
    from pricebook.mc_migrate import correlated_gbm_paths

    corr = np.array([[1.0, correlation], [correlation, 1.0]])
    paths = correlated_gbm_paths([spot1, spot2], [rate - div1, rate - div2],
                                  [vol1, vol2], corr, T, 1, n_paths, seed or 42)
    S1_T = paths[:, -1, 0]
    S2_T = paths[:, -1, 1]

    cond1 = S1_T >= barrier1 if is_above1 else S1_T <= barrier1
    cond2 = S2_T >= barrier2 if is_above2 else S2_T <= barrier2
    both = cond1 & cond2

    df = math.exp(-rate * T)
    prob = float(both.mean())
    price = df * payout * prob

    return DualDigitalResult(float(price), prob, payout)


@dataclass
class BonusCertificateResult:
    price: float
    bonus_level: float
    barrier: float
    barrier_hit_probability: float
    def to_dict(self) -> dict:
        return vars(self)


def bonus_certificate(
    spot: float, rate: float, dividend_yield: float, vol: float, T: float,
    bonus_level: float, barrier: float,
    notional: float = 1.0,
    n_paths: int = 20_000, n_steps: int = 252, seed: int | None = 42,
) -> BonusCertificateResult:
    """Bonus certificate: if barrier never hit, receive max(S_T, bonus_level).

    Structure = stock + down-and-in put at bonus_level.
    If barrier hit: certificate = stock (no bonus).
    If barrier not hit: certificate = max(S_T, bonus_level).
    """
    from pricebook.mc_migrate import gbm_paths

    paths_arr = gbm_paths(spot, rate - dividend_yield, vol, T, n_steps, n_paths, seed or 42)
    df = math.exp(-rate * T)

    hit = np.any(paths_arr <= barrier, axis=1)
    S_T = paths_arr[:, -1]

    # If hit: just stock. If not hit: max(S_T, bonus_level)
    payoff = np.where(hit, S_T, np.maximum(S_T, bonus_level))

    price = df * float(payoff.mean()) * notional / spot

    return BonusCertificateResult(float(price), bonus_level, barrier, float(hit.mean()))


# ---------------------------------------------------------------------------
# 4. Outperformance certificate
# ---------------------------------------------------------------------------

@dataclass
class OutperformanceCertResult:
    """Outperformance certificate result."""
    price: float
    participation: float
    cap: float | None
    expected_return: float
    def to_dict(self) -> dict:
        return vars(self)


def outperformance_certificate(
    spot: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    T: float,
    participation: float = 1.5,
    cap: float | None = None,
    notional: float = 1.0,
) -> OutperformanceCertResult:
    """Outperformance certificate: leveraged upside, full downside.

    Payoff at T:
    - If S_T >= S_0: notional x (1 + participation x (S_T/S_0 - 1)), capped at cap.
    - If S_T < S_0: notional x S_T/S_0 (full loss).

    Structure = stock + (participation - 1) x ATM call - (participation) x OTM call at cap.
    Participation > 1 amplifies gains. Cap limits maximum payout.

    Args:
        participation: upside multiplier (e.g., 1.5 = 150%).
        cap: maximum return (e.g., 0.30 = 30% cap). None = uncapped.
    """
    F = spot * math.exp((rate - dividend_yield) * T)
    df = math.exp(-rate * T)

    # Stock component
    stock_pv = spot * math.exp(-dividend_yield * T) * df * notional / spot

    # Extra calls for participation
    extra_calls = (participation - 1) * black76_price(F, spot, vol, T, df, OptionType.CALL) * notional / spot

    # Cap via short call
    cap_cost = 0.0
    if cap is not None:
        cap_strike = spot * (1 + cap)
        cap_cost = participation * black76_price(F, cap_strike, vol, T, df, OptionType.CALL) * notional / spot

    price = stock_pv + extra_calls - cap_cost
    expected_ret = (F / spot - 1) * participation

    return OutperformanceCertResult(float(price), participation, cap, float(expected_ret))


# ---------------------------------------------------------------------------
# Participation note
# ---------------------------------------------------------------------------

@dataclass
class ParticipationNoteResult:
    """Participation note pricing result."""
    price: float
    bond_floor: float           # PV of zero-coupon bond (capital protection)
    option_value: float         # PV of the equity participation
    participation_rate: float   # effective participation after cost
    protection_level: float     # fraction of principal protected (0 to 1)

    def to_dict(self) -> dict:
        return vars(self)


def participation_note(
    notional: float,
    spot: float,
    rate: float,
    vol: float,
    T: float,
    protection: float = 1.0,
    participation: float | None = None,
    cap: float | None = None,
    dividend_yield: float = 0.0,
) -> ParticipationNoteResult:
    """Capital-protected participation note.

    Structure: zero-coupon bond (for capital protection) + call option
    (for equity upside). The participation rate is determined by how
    much option budget remains after funding the bond floor.

    If participation is None, solve for max participation at zero cost.
    If cap is given, sell a call at cap to fund higher participation.

    Args:
        protection: fraction of principal protected (e.g. 1.0 = 100%, 0.9 = 90%).
        participation: if None, solve for zero-cost participation rate.
        cap: optional upside cap as fraction above spot (e.g. 0.5 = 50% cap).
    """
    from pricebook.black76 import black76_price as _b76, OptionType

    # Bond floor: PV of protected principal
    df = math.exp(-rate * T)
    bond_floor = protection * notional * df

    # Option budget: notional - bond_floor
    option_budget = notional - bond_floor
    if option_budget <= 0:
        return ParticipationNoteResult(
            float(bond_floor), float(bond_floor), 0.0, 0.0, protection)

    # ATM call price
    F = spot * math.exp((rate - dividend_yield) * T)
    atm_call = _b76(F, spot, vol, T, df, OptionType.CALL) * notional / spot

    if cap is not None:
        cap_strike = spot * (1 + cap)
        cap_value = _b76(F, cap_strike, vol, T, df, OptionType.CALL) * notional / spot
    else:
        cap_value = 0.0

    # Net call cost per unit participation
    call_cost_per_unit = atm_call - cap_value

    if participation is None:
        # Solve: participation × call_cost = option_budget
        if call_cost_per_unit > 1e-10:
            participation = option_budget / call_cost_per_unit
        else:
            participation = 0.0

    option_value = participation * call_cost_per_unit
    price = bond_floor + option_value

    return ParticipationNoteResult(
        price=float(price),
        bond_floor=float(bond_floor),
        option_value=float(option_value),
        participation_rate=float(participation),
        protection_level=protection,
    )
