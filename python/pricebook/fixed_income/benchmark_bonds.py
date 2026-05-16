"""Benchmark government bonds: UST, Bund, Gilt, JGB, OAT, BTP.

Factory methods for standard benchmark bonds with correct market conventions.
Curve fitting from benchmark yields. Trading strategy construction.

    from pricebook.fixed_income.benchmark_bonds import (
        BenchmarkUniverse, create_ust_universe, create_bund_universe,
        par_yield_curve, fitted_curve_nss,
        roll_down_ranking, carry_ranking, rv_scorecard,
        duration_neutral_spread, butterfly_trade, barbell_vs_bullet,
    )

References:
    Tuckman & Serrat (2012). Fixed Income Securities, Ch. 1-5.
    Nelson & Siegel (1987). Parsimonious Modeling of Yield Curves. JB.
    Svensson (1994). Estimating and Interpreting Forward Interest Rates. IMF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from dateutil.relativedelta import relativedelta

from pricebook.fixed_income.bond import FixedRateBond
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.schedule import Frequency


# ---------------------------------------------------------------------------
# Benchmark bond conventions
# ---------------------------------------------------------------------------

@dataclass
class BondConvention:
    """Market convention for a sovereign bond market."""
    country: str
    day_count: DayCountConvention
    frequency: Frequency
    settlement_days: int
    quoting: str             # "32nds" or "decimal"
    standard_tenors: list[int]

    def create_bond(self, ref_date: date, tenor_years: int, coupon: float,
                    face: float = 100.0) -> FixedRateBond:
        issue = ref_date - relativedelta(months=6)  # approximate recent issue
        maturity = ref_date + relativedelta(years=tenor_years)
        return FixedRateBond(
            issue_date=issue, maturity=maturity, coupon_rate=coupon,
            frequency=self.frequency, face_value=face,
            day_count=self.day_count, settlement_days=self.settlement_days,
        )


UST = BondConvention("US", DayCountConvention.ACT_ACT_ICMA, Frequency.SEMI_ANNUAL, 1, "32nds", [2, 3, 5, 7, 10, 20, 30])
BUND = BondConvention("DE", DayCountConvention.ACT_ACT_ICMA, Frequency.ANNUAL, 2, "decimal", [2, 5, 10, 30])
GILT = BondConvention("GB", DayCountConvention.ACT_ACT_ICMA, Frequency.SEMI_ANNUAL, 1, "decimal", [5, 10, 30])
JGB = BondConvention("JP", DayCountConvention.ACT_365_FIXED, Frequency.SEMI_ANNUAL, 2, "decimal", [5, 10, 20, 30, 40])
OAT = BondConvention("FR", DayCountConvention.ACT_ACT_ICMA, Frequency.ANNUAL, 2, "decimal", [2, 5, 10, 30])
BTP = BondConvention("IT", DayCountConvention.ACT_ACT_ICMA, Frequency.SEMI_ANNUAL, 2, "decimal", [3, 5, 7, 10, 15, 30])

CONVENTIONS = {"UST": UST, "Bund": BUND, "Gilt": GILT, "JGB": JGB, "OAT": OAT, "BTP": BTP}


# ---------------------------------------------------------------------------
# Benchmark universe
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkBond:
    """One benchmark bond in the universe."""
    label: str               # e.g., "UST 10Y"
    tenor: int
    bond: FixedRateBond
    market_yield: float
    market_price: float | None = None

    def to_dict(self) -> dict:
        return {"label": self.label, "tenor": self.tenor,
                "yield": self.market_yield, "price": self.market_price}


@dataclass
class BenchmarkUniverse:
    """Collection of benchmark bonds for one sovereign market."""
    market: str
    reference_date: date
    convention: BondConvention
    bonds: list[BenchmarkBond]

    def yields(self) -> dict[int, float]:
        return {b.tenor: b.market_yield for b in self.bonds}

    def bond_at(self, tenor: int) -> BenchmarkBond | None:
        for b in self.bonds:
            if b.tenor == tenor:
                return b
        return None

    def to_dict(self) -> dict:
        return {"market": self.market, "date": self.reference_date.isoformat(),
                "bonds": [b.to_dict() for b in self.bonds]}


def create_universe(
    market: str,
    reference_date: date,
    yields: dict[int, float],
    coupons: dict[int, float] | None = None,
) -> BenchmarkUniverse:
    """Create a benchmark universe from market yields.

    Args:
        market: "UST", "Bund", "Gilt", "JGB", "OAT", "BTP".
        yields: {tenor_years: yield} e.g., {2: 0.045, 10: 0.039}.
        coupons: {tenor: coupon_rate}. If None, coupon = yield (par bonds).
    """
    conv = CONVENTIONS.get(market)
    if conv is None:
        raise ValueError(f"Unknown market '{market}'. Available: {list(CONVENTIONS)}")

    bonds = []
    for tenor in sorted(yields.keys()):
        coupon = (coupons or {}).get(tenor, yields[tenor])
        bond = conv.create_bond(reference_date, tenor, coupon)
        mkt_yield = yields[tenor]
        settle = bond.settlement_date(reference_date)
        mkt_price = bond._price_from_ytm(mkt_yield, settle)
        bonds.append(BenchmarkBond(
            label=f"{market} {tenor}Y", tenor=tenor,
            bond=bond, market_yield=mkt_yield, market_price=mkt_price,
        ))

    return BenchmarkUniverse(market, reference_date, conv, bonds)


# Convenience factories
def create_ust_universe(ref_date: date, yields: dict[int, float], **kw) -> BenchmarkUniverse:
    return create_universe("UST", ref_date, yields, **kw)

def create_bund_universe(ref_date: date, yields: dict[int, float], **kw) -> BenchmarkUniverse:
    return create_universe("Bund", ref_date, yields, **kw)

def create_gilt_universe(ref_date: date, yields: dict[int, float], **kw) -> BenchmarkUniverse:
    return create_universe("Gilt", ref_date, yields, **kw)

def create_jgb_universe(ref_date: date, yields: dict[int, float], **kw) -> BenchmarkUniverse:
    return create_universe("JGB", ref_date, yields, **kw)


# ---------------------------------------------------------------------------
# Par yield curve from benchmark bonds
# ---------------------------------------------------------------------------

def par_yield_curve(
    universe: BenchmarkUniverse,
    discount_curve: DiscountCurve,
) -> dict[int, float]:
    """Compute par yields at each benchmark tenor.

    Par yield = coupon rate that makes bond price = 100 (par).
    Different from market yield when bonds trade away from par.
    """
    par_yields = {}
    for b in universe.bonds:
        bond = b.bond
        settle = bond.settlement_date(universe.reference_date)
        # Par yield: solve for coupon such that dirty price = 100
        # Approximation: par_yield ≈ (1 - df(T)) / annuity
        T = year_fraction(settle, bond.maturity, bond.day_count)
        df_T = discount_curve.df(bond.maturity)
        freq = bond.frequency.value
        periods_per_year = 12 / freq
        n_periods = max(1, int(T * periods_per_year))
        annuity = sum(
            discount_curve.df(settle + relativedelta(months=int(freq * (i + 1))))
            / periods_per_year
            for i in range(n_periods)
        )
        if annuity > 1e-10:
            par_yields[b.tenor] = (1 - df_T) / annuity
        else:
            par_yields[b.tenor] = b.market_yield

    return par_yields


# ---------------------------------------------------------------------------
# Nelson-Siegel-Svensson curve fitting
# ---------------------------------------------------------------------------

@dataclass
class NSSParams:
    """Nelson-Siegel-Svensson parameters."""
    beta0: float   # level (long rate)
    beta1: float   # slope (short - long)
    beta2: float   # curvature (hump)
    beta3: float   # second hump (Svensson extension)
    tau1: float    # decay factor 1
    tau2: float    # decay factor 2

    def yield_at(self, T: float) -> float:
        """NSS yield at maturity T."""
        if T <= 0:
            return self.beta0 + self.beta1
        x1 = T / self.tau1
        x2 = T / self.tau2
        e1 = (1 - math.exp(-x1)) / x1
        e2 = e1 - math.exp(-x1)
        e3 = (1 - math.exp(-x2)) / x2 - math.exp(-x2)
        return self.beta0 + self.beta1 * e1 + self.beta2 * e2 + self.beta3 * e3

    def to_dict(self) -> dict:
        return vars(self)


def fitted_curve_nss(
    universe: BenchmarkUniverse,
) -> NSSParams:
    """Fit Nelson-Siegel-Svensson to benchmark yields.

    Minimises sum of squared yield errors across benchmark tenors.
    """
    from scipy.optimize import minimize

    tenors = np.array([b.tenor for b in universe.bonds], dtype=float)
    yields_obs = np.array([b.market_yield for b in universe.bonds])

    def objective(params):
        b0, b1, b2, b3, t1, t2 = params
        if t1 <= 0.1 or t2 <= 0.1:
            return 1e10
        nss = NSSParams(b0, b1, b2, b3, t1, t2)
        fitted = np.array([nss.yield_at(t) for t in tenors])
        return float(np.sum((fitted - yields_obs) ** 2))

    # Initial guess: beta0=long rate, beta1=short-long, beta2/beta3=curvature
    long_y = float(yields_obs[-1])
    short_y = float(yields_obs[0])
    x0 = [long_y, short_y - long_y, 0.0, 0.0, 1.5, 5.0]

    # Bounded optimisation to prevent wild parameters
    bounds = [
        (0.0, 0.20),       # beta0: level
        (-0.10, 0.10),     # beta1: slope
        (-0.10, 0.10),     # beta2: curvature
        (-0.10, 0.10),     # beta3: second hump
        (0.3, 10.0),       # tau1: decay 1
        (0.5, 30.0),       # tau2: decay 2
    ]
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 5000})

    b0, b1, b2, b3, t1, t2 = result.x
    return NSSParams(float(b0), float(b1), float(b2), float(b3), float(t1), float(t2))


# ---------------------------------------------------------------------------
# Trading strategies
# ---------------------------------------------------------------------------

@dataclass
class SpreadTradeResult:
    """Duration-neutral spread trade."""
    short_label: str
    long_label: str
    short_dv01: float
    long_dv01: float
    notional_ratio: float     # long_notional / short_notional for DV01 neutrality
    spread_bps: float
    carry_30d: float

    def to_dict(self) -> dict:
        return vars(self)


def duration_neutral_spread(
    universe: BenchmarkUniverse,
    short_tenor: int,
    long_tenor: int,
    curve: DiscountCurve,
    notional: float = 10_000_000,
) -> SpreadTradeResult:
    """Build a DV01-neutral curve spread trade (steepener or flattener).

    Long the long-tenor bond, short the short-tenor bond,
    with notionals sized so net DV01 = 0.

    E.g., 2s10s steepener: long 10Y, short 2Y.
    """
    from pricebook.desks.bond_trading_desk import bond_risk_metrics

    short_bm = universe.bond_at(short_tenor)
    long_bm = universe.bond_at(long_tenor)
    if short_bm is None or long_bm is None:
        raise ValueError(f"Tenor {short_tenor}Y or {long_tenor}Y not in universe")

    settle = short_bm.bond.settlement_date(universe.reference_date)
    rm_short = bond_risk_metrics(short_bm.bond, curve, settle)
    rm_long = bond_risk_metrics(long_bm.bond, curve, settle)

    # DV01-neutral: short_notional × short_dv01 = long_notional × long_dv01
    if abs(rm_short.dv01_curve) < 1e-10:
        ratio = 1.0
    else:
        ratio = rm_short.dv01_curve / rm_long.dv01_curve

    spread = (long_bm.market_yield - short_bm.market_yield) * 1e4
    # 30d carry: long coupon - short coupon (simplified)
    carry = (long_bm.bond.coupon_rate - short_bm.bond.coupon_rate * ratio) * notional / 100 * 30 / 360

    return SpreadTradeResult(
        short_label=short_bm.label, long_label=long_bm.label,
        short_dv01=rm_short.dv01_curve, long_dv01=rm_long.dv01_curve,
        notional_ratio=ratio, spread_bps=spread, carry_30d=carry,
    )


@dataclass
class ButterflyTradeResult:
    """Duration-neutral butterfly trade."""
    wing_short_label: str
    belly_label: str
    wing_long_label: str
    wing_weights: tuple[float, float]   # (short_wing, long_wing) notional weights
    belly_weight: float
    spread_bps: float

    def to_dict(self) -> dict:
        return {**vars(self), "wing_weights": list(self.wing_weights)}


def butterfly_trade(
    universe: BenchmarkUniverse,
    short_tenor: int,
    belly_tenor: int,
    long_tenor: int,
    curve: DiscountCurve,
) -> ButterflyTradeResult:
    """Build a DV01-neutral butterfly: long belly, short wings.

    E.g., 2s5s10s: long 5Y, short 2Y and 10Y.
    Profits from belly cheapening relative to wings.
    Weights set so total DV01 = 0 and total convexity ≈ 0.
    """
    from pricebook.desks.bond_trading_desk import bond_risk_metrics

    bm_s = universe.bond_at(short_tenor)
    bm_b = universe.bond_at(belly_tenor)
    bm_l = universe.bond_at(long_tenor)
    if None in (bm_s, bm_b, bm_l):
        raise ValueError(f"Tenors {short_tenor}/{belly_tenor}/{long_tenor} not all in universe")

    settle = bm_s.bond.settlement_date(universe.reference_date)
    rm_s = bond_risk_metrics(bm_s.bond, curve, settle)
    rm_b = bond_risk_metrics(bm_b.bond, curve, settle)
    rm_l = bond_risk_metrics(bm_l.bond, curve, settle)

    # Solve: w_s × dv01_s + w_l × dv01_l = dv01_b (DV01 neutral)
    # Additional: w_s + w_l proportional to distance from belly
    # Standard: equal-weighted wings in DV01 space
    total_wing_dv01 = rm_s.dv01_curve + rm_l.dv01_curve
    if total_wing_dv01 > 1e-10:
        w_s = rm_b.dv01_curve * rm_l.dv01_curve / (rm_s.dv01_curve * total_wing_dv01)
        w_l = rm_b.dv01_curve * rm_s.dv01_curve / (rm_l.dv01_curve * total_wing_dv01)
    else:
        w_s = w_l = 0.5

    spread = (2 * bm_b.market_yield - bm_s.market_yield - bm_l.market_yield) * 1e4

    return ButterflyTradeResult(
        wing_short_label=bm_s.label, belly_label=bm_b.label,
        wing_long_label=bm_l.label,
        wing_weights=(w_s, w_l), belly_weight=1.0,
        spread_bps=spread,
    )


@dataclass
class BarbellBulletResult:
    """Barbell vs bullet comparison."""
    barbell_yield: float
    bullet_yield: float
    pickup_bps: float
    barbell_convexity: float
    bullet_convexity: float
    breakeven_shift_bps: float    # parallel shift where barbell = bullet

    def to_dict(self) -> dict:
        return vars(self)


def barbell_vs_bullet(
    universe: BenchmarkUniverse,
    short_tenor: int,
    bullet_tenor: int,
    long_tenor: int,
    curve: DiscountCurve,
) -> BarbellBulletResult:
    """Compare barbell (short+long) vs bullet (single maturity) for same duration.

    Barbell has more convexity (good) but lower yield (bad).
    Breakeven = yield shift where convexity gain offsets yield loss.
    """
    from pricebook.desks.bond_trading_desk import bond_risk_metrics

    bm_s = universe.bond_at(short_tenor)
    bm_m = universe.bond_at(bullet_tenor)
    bm_l = universe.bond_at(long_tenor)
    if None in (bm_s, bm_m, bm_l):
        raise ValueError(f"Tenors not in universe")

    settle = bm_s.bond.settlement_date(universe.reference_date)
    rm_s = bond_risk_metrics(bm_s.bond, curve, settle)
    rm_m = bond_risk_metrics(bm_m.bond, curve, settle)
    rm_l = bond_risk_metrics(bm_l.bond, curve, settle)

    # Weight barbell to match bullet duration
    target_dur = rm_m.modified_duration
    if abs(rm_l.modified_duration - rm_s.modified_duration) < 1e-10:
        w = 0.5
    else:
        w = (target_dur - rm_s.modified_duration) / (rm_l.modified_duration - rm_s.modified_duration)
        w = max(0, min(1, w))

    barbell_yield = (1 - w) * bm_s.market_yield + w * bm_l.market_yield
    barbell_conv = (1 - w) * rm_s.convexity + w * rm_l.convexity
    pickup = (bm_m.market_yield - barbell_yield) * 1e4
    conv_diff = barbell_conv - rm_m.convexity

    # Breakeven: yield pickup = 0.5 × convexity_diff × (Δy)²
    if conv_diff > 0 and pickup > 0:
        breakeven = math.sqrt(2 * pickup / 1e4 / conv_diff) * 1e4
    else:
        breakeven = 0.0

    return BarbellBulletResult(
        barbell_yield=barbell_yield, bullet_yield=bm_m.market_yield,
        pickup_bps=pickup, barbell_convexity=barbell_conv,
        bullet_convexity=rm_m.convexity, breakeven_shift_bps=breakeven,
    )


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

@dataclass
class BondRanking:
    """One bond's score in a ranking."""
    label: str
    tenor: int
    score: float
    details: dict

    def to_dict(self) -> dict:
        return {"label": self.label, "tenor": self.tenor,
                "score": self.score, **self.details}


def roll_down_ranking(
    universe: BenchmarkUniverse,
    curve: DiscountCurve,
    horizon_days: int = 90,
) -> list[BondRanking]:
    """Rank bonds by roll-down value (curve unchanged, time passes).

    Roll-down = price gain from moving down the curve over horizon.
    Higher roll-down = steeper local curve = more attractive carry.
    """
    from pricebook.desks.bond_trading_desk import bond_carry_roll

    rankings = []
    for b in universe.bonds:
        settle = b.bond.settlement_date(universe.reference_date)
        try:
            cr = bond_carry_roll(b.bond, curve, horizon_days=horizon_days, settlement=settle)
            rankings.append(BondRanking(
                b.label, b.tenor, cr.roll_down_return,
                {"roll_down": cr.roll_down_return, "net_carry": cr.net_carry,
                 "total": cr.total_carry_and_roll},
            ))
        except (ValueError, ZeroDivisionError):
            pass

    rankings.sort(key=lambda r: r.score, reverse=True)
    return rankings


def carry_ranking(
    universe: BenchmarkUniverse,
    curve: DiscountCurve,
    repo_rate: float = 0.05,
    horizon_days: int = 30,
) -> list[BondRanking]:
    """Rank bonds by net carry (coupon income - financing cost)."""
    from pricebook.desks.bond_trading_desk import bond_carry_roll

    rankings = []
    for b in universe.bonds:
        settle = b.bond.settlement_date(universe.reference_date)
        try:
            cr = bond_carry_roll(b.bond, curve, repo_rate=repo_rate,
                                  horizon_days=horizon_days, settlement=settle)
            rankings.append(BondRanking(
                b.label, b.tenor, cr.net_carry,
                {"coupon": cr.coupon_carry, "financing": cr.funding_cost,
                 "net_carry": cr.net_carry},
            ))
        except (ValueError, ZeroDivisionError):
            pass

    rankings.sort(key=lambda r: r.score, reverse=True)
    return rankings


def rv_scorecard(
    universe: BenchmarkUniverse,
    fitted: NSSParams,
    curve: DiscountCurve,
    repo_rate: float = 0.05,
) -> list[BondRanking]:
    """Composite RV scorecard: rich/cheap + carry + roll-down.

    Score = -rich_cheap_bps + carry_bps + rolldown_bps.
    Positive = attractive (cheap + high carry + good roll).
    """
    from pricebook.desks.bond_trading_desk import bond_carry_roll

    rankings = []
    for b in universe.bonds:
        fitted_yield = fitted.yield_at(float(b.tenor))
        rich_cheap = (b.market_yield - fitted_yield) * 1e4  # positive = cheap

        settle = b.bond.settlement_date(universe.reference_date)
        try:
            cr = bond_carry_roll(b.bond, curve, repo_rate=repo_rate,
                                  horizon_days=30, settlement=settle)
            carry_bps = cr.net_carry * 1e4 / max(b.market_price or 100, 1)
            roll_bps = cr.roll_down_return * 1e4 / max(b.market_price or 100, 1)
        except (ValueError, ZeroDivisionError):
            carry_bps = roll_bps = 0.0

        score = rich_cheap + carry_bps + roll_bps

        rankings.append(BondRanking(
            b.label, b.tenor, score,
            {"rich_cheap_bps": rich_cheap, "carry_bps": carry_bps,
             "roll_bps": roll_bps, "fitted_yield": fitted_yield},
        ))

    rankings.sort(key=lambda r: r.score, reverse=True)
    return rankings
