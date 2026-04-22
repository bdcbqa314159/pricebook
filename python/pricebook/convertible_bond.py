"""Convertible bonds: full conversion mechanics, soft/hard call, CoCo, exchangeable.

* :class:`ConvertibleBond` — convertible bond with full conversion analytics.
* :func:`convertible_delta_hedge` — equity hedge ratio.
* :func:`convertible_soft_call` — soft call with trigger.
* :func:`contingent_convertible` — CoCo with conversion trigger.
* :func:`exchangeable_bond` — exchangeable into different issuer's shares.
* :func:`mandatory_convertible` — forced conversion at maturity.

References:
    Tsiveriotis & Fernandes, *Valuing Convertible Bonds with Credit Risk*, JF, 1998.
    Ayache, Forsyth & Vetzal, *Valuation of Convertible Bonds with Credit Risk*,
    J. Derivatives, 2003.
    Calamos, *Convertible Securities*, McGraw-Hill, 2003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Convertible bond ----

@dataclass
class ConvertibleResult:
    """Convertible bond pricing result."""
    price: float
    bond_floor: float           # straight bond value
    conversion_value: float     # conversion_ratio × spot
    conversion_premium: float   # (price - conversion_value) / conversion_value
    parity: float               # conversion_value / notional
    equity_delta: float
    n_paths: int


@dataclass
class ConvertibleBond:
    """Convertible bond specification.

    Standard CB: a bond that can be converted into a fixed number of shares
    (conversion ratio) at the holder's discretion.

    Args:
        notional: face value.
        coupon_rate: annual coupon rate.
        maturity_years: time to maturity.
        conversion_ratio: shares per unit of notional.
        n_coupons_per_year: payment frequency.
    """
    notional: float
    coupon_rate: float
    maturity_years: float
    conversion_ratio: float
    n_coupons_per_year: int = 2

    def parity(self, spot: float) -> float:
        """Conversion value / notional."""
        return self.conversion_ratio * spot / self.notional

    def conversion_price(self) -> float:
        """Stock price at which conversion_value = notional."""
        return self.notional / self.conversion_ratio

    def price(
        self,
        spot: float,
        rate: float,
        equity_vol: float,
        credit_spread: float = 0.0,
        dividend_yield: float = 0.0,
        n_paths: int = 10_000,
        n_steps: int | None = None,
        seed: int | None = 42,
    ) -> ConvertibleResult:
        """MC pricing with optimal holder conversion.

        At each step: max(continuation_value, conversion_value).

        Uses Tsiveriotis-Fernandes credit adjustment: discount using
        r + λ where λ = credit_spread, and pay recovery on default.
        Simplified: no default jump, but credit spread shifts discount.
        """
        if n_steps is None:
            n_steps = int(self.maturity_years * 12)

        rng = np.random.default_rng(seed)
        dt = self.maturity_years / n_steps
        sqrt_dt = math.sqrt(dt)
        disc_rate = rate + credit_spread

        # Coupon times (in steps)
        coupon_step_interval = max(n_steps // max(int(self.maturity_years * self.n_coupons_per_year), 1), 1)
        coupon_amount = self.notional * self.coupon_rate / self.n_coupons_per_year

        # Simulate equity paths
        S = np.full((n_paths, n_steps + 1), float(spot))
        drift = (rate - dividend_yield - 0.5 * equity_vol**2) * dt
        for step in range(n_steps):
            z = rng.standard_normal(n_paths)
            S[:, step + 1] = S[:, step] * np.exp(drift + equity_vol * sqrt_dt * z)

        # Backward induction with LSM regression for continuation value
        # Terminal: max(notional, conversion_value)
        V = np.maximum(float(self.notional), self.conversion_ratio * S[:, -1]).astype(float)

        for step in range(n_steps - 1, -1, -1):
            V *= math.exp(-disc_rate * dt)
            if step > 0 and step % coupon_step_interval == 0:
                V += coupon_amount

            conv_val = self.conversion_ratio * S[:, step]
            # LSM: estimate E[V(t+1)|S(t)] via polynomial regression
            s_t = S[:, step]
            if step > 0 and s_t.std() > 1e-10:
                # Basis: 1, S, S^2 (Longstaff-Schwartz)
                X = np.column_stack([np.ones(n_paths), s_t, s_t**2])
                try:
                    coeffs = np.linalg.lstsq(X, V, rcond=None)[0]
                    continuation = X @ coeffs
                except np.linalg.LinAlgError:
                    continuation = V
                V = np.where(conv_val > continuation, conv_val, V)
            else:
                V = np.maximum(V, conv_val)

        price = float(V.mean())

        # Bond floor: straight bond at same credit spread
        bond_floor = 0.0
        for i in range(1, int(self.maturity_years * self.n_coupons_per_year) + 1):
            t = i / self.n_coupons_per_year
            bond_floor += coupon_amount * math.exp(-disc_rate * t)
        bond_floor += self.notional * math.exp(-disc_rate * self.maturity_years)

        conversion_val = self.conversion_ratio * spot
        premium = (price - conversion_val) / max(conversion_val, 1e-6)

        # Delta: ∂V/∂S via common-random-numbers pathwise bump
        # Re-price on bumped spot with same random paths
        bump_frac = 0.01
        spot_up = spot * (1 + bump_frac)
        rng_up = np.random.default_rng(seed)
        S_up = np.full((n_paths, n_steps + 1), float(spot_up))
        for step in range(n_steps):
            z = rng_up.standard_normal(n_paths)
            S_up[:, step + 1] = S_up[:, step] * np.exp(drift + equity_vol * sqrt_dt * z)

        V_up = np.maximum(float(self.notional), self.conversion_ratio * S_up[:, -1]).astype(float)
        for step in range(n_steps - 1, -1, -1):
            V_up *= math.exp(-disc_rate * dt)
            if step > 0 and step % coupon_step_interval == 0:
                V_up += coupon_amount
            conv_up = self.conversion_ratio * S_up[:, step]
            s_up_t = S_up[:, step]
            if step > 0 and s_up_t.std() > 1e-10:
                X_up = np.column_stack([np.ones(n_paths), s_up_t, s_up_t**2])
                try:
                    c_up = np.linalg.lstsq(X_up, V_up, rcond=None)[0]
                    cont_up = X_up @ c_up
                except np.linalg.LinAlgError:
                    cont_up = V_up
                V_up = np.where(conv_up > cont_up, conv_up, V_up)
            else:
                V_up = np.maximum(V_up, conv_up)
        up_price = float(V_up.mean())

        # Note: _compute_delta used fresh RNG — use pathwise diff for accuracy
        delta = (up_price - price) / (spot_up - spot)

        return ConvertibleResult(
            price=price,
            bond_floor=float(bond_floor),
            conversion_value=float(conversion_val),
            conversion_premium=float(premium),
            parity=float(conversion_val / self.notional),
            equity_delta=float(delta),
            n_paths=n_paths,
        )

    def _compute_delta(self, spot_up, spot_base, rate, vol, cs, q, n_paths, n_steps, seed, base_price):
        rng = np.random.default_rng(seed)
        dt = self.maturity_years / n_steps
        sqrt_dt = math.sqrt(dt)
        disc_rate = rate + cs

        coupon_step_interval = max(n_steps // max(int(self.maturity_years * self.n_coupons_per_year), 1), 1)
        coupon_amount = self.notional * self.coupon_rate / self.n_coupons_per_year

        S = np.full((n_paths, n_steps + 1), spot_up)
        drift = (rate - q - 0.5 * vol**2) * dt
        for step in range(n_steps):
            z = rng.standard_normal(n_paths)
            S[:, step + 1] = S[:, step] * np.exp(drift + vol * sqrt_dt * z)

        V = np.maximum(float(self.notional), self.conversion_ratio * S[:, -1]).astype(float)
        for step in range(n_steps - 1, -1, -1):
            V *= math.exp(-disc_rate * dt)
            if step > 0 and step % coupon_step_interval == 0:
                V += coupon_amount
            V = np.maximum(V, self.conversion_ratio * S[:, step])

        up_price = float(V.mean())
        return (up_price - base_price) / (spot_up - spot_base)


# ---- Convertible delta hedge ----

@dataclass
class DeltaHedgeResult:
    """Convertible delta hedge result."""
    convertible_delta: float
    shares_to_short: float
    hedge_notional: float
    residual_pnl: float         # unhedged (gamma + higher-order)


def convertible_delta_hedge(
    cb_result: ConvertibleResult,
    spot: float,
    cb_notional_traded: float = 1.0,
) -> DeltaHedgeResult:
    """Compute shares to short for delta-neutral CB hedge.

    Delta-hedge: long CB, short delta × conversion_ratio shares.

    Args:
        cb_result: result from ConvertibleBond.price().
        spot: equity spot.
        cb_notional_traded: face value of CBs traded.
    """
    shares = cb_result.equity_delta * cb_notional_traded
    hedge_notional = shares * spot

    # Residual: gamma P&L + basis + higher order (here just report nominal)
    return DeltaHedgeResult(
        convertible_delta=cb_result.equity_delta,
        shares_to_short=float(shares),
        hedge_notional=float(hedge_notional),
        residual_pnl=0.0,
    )


# ---- Soft call ----

@dataclass
class SoftCallResult:
    """Soft call convertible result."""
    price: float
    price_no_call: float
    call_option_value: float        # value forgone by holder
    trigger_level: float


def convertible_soft_call(
    cb: ConvertibleBond,
    spot: float,
    rate: float,
    equity_vol: float,
    soft_call_trigger: float,       # as fraction of conversion price (e.g. 1.3)
    soft_call_years_after: float = 3.0,
    credit_spread: float = 0.0,
    dividend_yield: float = 0.0,
    n_paths: int = 10_000,
    n_steps: int | None = None,
    seed: int | None = 42,
) -> SoftCallResult:
    """Soft call convertible: issuer can call at par (or call price) if stock
    trades above trigger × conversion price for some period, after protection expires.

    Holder's rational response: convert if called (since stock is above trigger,
    conversion value > par).

    Effect: caps upside when in-the-money.

    Args:
        soft_call_trigger: trigger (e.g. 1.3 for 130% of conversion price).
        soft_call_years_after: call protection expiry.
    """
    if n_steps is None:
        n_steps = int(cb.maturity_years * 12)

    # Straight pricing (no call) for comparison
    base = cb.price(spot, rate, equity_vol, credit_spread, dividend_yield,
                     n_paths, n_steps, seed)

    # With soft call: when stock > trigger × conv_price and past protection,
    # issuer calls and holder converts.
    rng = np.random.default_rng(seed)
    dt = cb.maturity_years / n_steps
    sqrt_dt = math.sqrt(dt)
    disc_rate = rate + credit_spread
    conv_price = cb.conversion_price()
    trigger_level = soft_call_trigger * conv_price

    coupon_step_interval = max(n_steps // max(int(cb.maturity_years * cb.n_coupons_per_year), 1), 1)
    coupon_amount = cb.notional * cb.coupon_rate / cb.n_coupons_per_year
    protection_step = int(soft_call_years_after * n_steps / cb.maturity_years)

    S = np.full((n_paths, n_steps + 1), spot)
    drift = (rate - dividend_yield - 0.5 * equity_vol**2) * dt
    for step in range(n_steps):
        z = rng.standard_normal(n_paths)
        S[:, step + 1] = S[:, step] * np.exp(drift + equity_vol * sqrt_dt * z)

    V = np.maximum(float(cb.notional), cb.conversion_ratio * S[:, -1]).astype(float)
    for step in range(n_steps - 1, -1, -1):
        V *= math.exp(-disc_rate * dt)
        if step > 0 and step % coupon_step_interval == 0:
            V += coupon_amount
        # Holder conversion
        conv_val = cb.conversion_ratio * S[:, step]
        V = np.maximum(V, conv_val)
        # Issuer call (after protection): cap value at conversion_value if above trigger
        if step >= protection_step:
            called = S[:, step] >= trigger_level
            V = np.where(called, np.minimum(V, conv_val), V)

    price_soft = float(V.mean())
    call_value = base.price - price_soft

    return SoftCallResult(
        price=float(price_soft),
        price_no_call=float(base.price),
        call_option_value=float(call_value),
        trigger_level=float(trigger_level),
    )


# ---- Contingent convertible (CoCo) ----

@dataclass
class CoCoResult:
    """Contingent convertible result."""
    price: float
    conversion_probability: float
    trigger_level: float
    loss_on_trigger: float


def contingent_convertible(
    notional: float,
    coupon_rate: float,
    maturity_years: float,
    conversion_trigger: float,      # equity level triggering conversion
    conversion_ratio: float,
    spot: float,
    rate: float,
    equity_vol: float,
    credit_spread: float = 0.0,
    loss_absorption: float = 0.5,   # fraction of principal lost on conversion
    n_paths: int = 10_000,
    n_steps: int | None = None,
    seed: int | None = 42,
) -> CoCoResult:
    """Contingent convertible: mandatorily converts when equity crosses trigger.

    Unlike regular CB, CoCo is issuer-friendly: triggers absorb losses.
    Common for bank capital (AT1 bonds).

    At conversion:
    - Bondholder receives conversion_ratio × S(t), often worth < par.
    - Principal loss = (1 − loss_absorption) × notional.

    Args:
        conversion_trigger: equity level triggering conversion.
        loss_absorption: fraction preserved on conversion (0.5 = 50% loss).
    """
    if n_steps is None:
        n_steps = int(maturity_years * 12)

    rng = np.random.default_rng(seed)
    dt = maturity_years / n_steps
    sqrt_dt = math.sqrt(dt)
    disc_rate = rate + credit_spread

    n_coupons_per_year = 2
    coupon_step_interval = max(n_steps // max(int(maturity_years * n_coupons_per_year), 1), 1)
    coupon_amount = notional * coupon_rate / n_coupons_per_year

    S = np.full((n_paths, n_steps + 1), spot)
    drift = (rate - 0.5 * equity_vol**2) * dt
    for step in range(n_steps):
        z = rng.standard_normal(n_paths)
        S[:, step + 1] = S[:, step] * np.exp(drift + equity_vol * sqrt_dt * z)

    # Check trigger per path
    triggered = np.zeros(n_paths, dtype=bool)
    trigger_step = np.full(n_paths, n_steps)
    for step in range(1, n_steps + 1):
        new_trigger = ~triggered & (S[:, step] <= conversion_trigger)
        trigger_step = np.where(new_trigger, step, trigger_step)
        triggered |= new_trigger

    pv = np.zeros(n_paths)
    for p in range(n_paths):
        if triggered[p]:
            ts = trigger_step[p]
            # Pay coupons until trigger, then conversion payoff
            for i in range(1, n_steps + 1):
                if i >= ts:
                    break
                if i % coupon_step_interval == 0:
                    pv[p] += coupon_amount * math.exp(-disc_rate * i * dt)
            # Conversion payoff (scaled)
            conv_val = conversion_ratio * S[p, ts] * loss_absorption
            pv[p] += conv_val * math.exp(-disc_rate * ts * dt)
        else:
            # Full coupons + principal
            for i in range(1, n_steps + 1):
                if i % coupon_step_interval == 0:
                    pv[p] += coupon_amount * math.exp(-disc_rate * i * dt)
            pv[p] += notional * math.exp(-disc_rate * maturity_years)

    price = float(pv.mean())
    conv_prob = float(triggered.mean())
    loss_on_trigger = notional * (1 - loss_absorption)

    return CoCoResult(
        price=price,
        conversion_probability=conv_prob,
        trigger_level=float(conversion_trigger),
        loss_on_trigger=float(loss_on_trigger),
    )


# ---- Exchangeable bond ----

@dataclass
class ExchangeableResult:
    """Exchangeable bond result (convertible into different issuer's stock)."""
    price: float
    bond_floor: float
    option_value: float
    underlying_spot: float


def exchangeable_bond(
    notional: float,
    coupon_rate: float,
    maturity_years: float,
    conversion_ratio: float,
    underlying_spot: float,         # stock of DIFFERENT issuer
    rate: float,
    equity_vol: float,
    issuer_credit_spread: float,
    underlying_dividend_yield: float = 0.0,
    n_paths: int = 10_000,
    n_steps: int | None = None,
    seed: int | None = 42,
) -> ExchangeableResult:
    """Exchangeable bond: convertible into another issuer's stock.

    Issuer retains stock on their balance sheet; creditworthiness of
    the issuer matters for the bond part, creditworthiness of the
    underlying issuer matters for the equity.

    Simplified: treat like regular convertible with underlying stock
    but issuer credit spread for discounting.
    """
    cb = ConvertibleBond(notional=notional, coupon_rate=coupon_rate,
                           maturity_years=maturity_years,
                           conversion_ratio=conversion_ratio)
    result = cb.price(underlying_spot, rate, equity_vol,
                       issuer_credit_spread,
                       underlying_dividend_yield,
                       n_paths, n_steps, seed)

    option_value = result.price - result.bond_floor

    return ExchangeableResult(
        price=result.price,
        bond_floor=result.bond_floor,
        option_value=float(option_value),
        underlying_spot=underlying_spot,
    )


# ---- Mandatory convertible ----

@dataclass
class MandatoryConvertibleResult:
    """Mandatory convertible bond result."""
    price: float
    min_shares: float
    max_shares: float
    low_strike: float
    high_strike: float


def mandatory_convertible(
    notional: float,
    coupon_rate: float,
    maturity_years: float,
    low_strike: float,              # protection level
    high_strike: float,             # cap level
    spot: float,
    rate: float,
    equity_vol: float,
    dividend_yield: float = 0.0,
    n_paths: int = 20_000,
    seed: int | None = 42,
) -> MandatoryConvertibleResult:
    """Mandatory convertible (MC): forced conversion at maturity.

    Typical structure: holder receives N shares at maturity, where
        N × S_T = notional if high_strike ≤ S_T
        N × S_T = notional × S_T/high_strike if low_strike ≤ S_T < high_strike
        N × S_T = notional × S_T/low_strike if S_T < low_strike (more shares)

    Plus coupons along the way.

    Args:
        low_strike: price below which holder gets more shares (protection).
        high_strike: price above which shares capped.
    """
    rng = np.random.default_rng(seed)
    sqrt_T = math.sqrt(maturity_years)

    n_coupons_per_year = 2
    total_coupons = int(maturity_years * n_coupons_per_year)
    coupon_amount = notional * coupon_rate / n_coupons_per_year

    # PV of coupons
    coupon_pv = sum(coupon_amount * math.exp(-rate * (i + 1) / n_coupons_per_year)
                     for i in range(total_coupons))

    # Terminal stock
    drift = (rate - dividend_yield - 0.5 * equity_vol**2) * maturity_years
    Z = rng.standard_normal(n_paths)
    S_T = spot * np.exp(drift + equity_vol * sqrt_T * Z)

    # Terminal payoff: notional-equivalent shares
    shares_low = notional / low_strike   # protection: more shares if S < low
    shares_high = notional / high_strike  # capped: fewer shares if S > high

    payoff = np.where(
        S_T >= high_strike, shares_high * S_T,
        np.where(S_T >= low_strike, notional,   # at-par region
                  shares_low * S_T)
    )

    df = math.exp(-rate * maturity_years)
    terminal_pv = df * float(payoff.mean())

    price = coupon_pv + terminal_pv

    return MandatoryConvertibleResult(
        price=float(price),
        min_shares=float(shares_high),
        max_shares=float(shares_low),
        low_strike=low_strike,
        high_strike=high_strike,
    )
