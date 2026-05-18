"""Convertible bond trading desk: book, risk, carry, P&L, stress, capital, lifecycle.

Wraps convertible_bond.py pricing into the standard 9-component desk protocol.

Convertibles are hybrid (equity + credit + rates), so the desk layer needs
multi-asset Greeks: equity delta/gamma/vega, credit CS01, rate DV01.

    from pricebook.desks.convertible_bond_desk import (
        cb_risk_metrics, CBRiskMetrics,
        CBBook, CBBookEntry,
        cb_carry_decomposition, CBCarryDecomposition,
        cb_daily_pnl, CBDailyPnL,
        cb_dashboard, CBDashboard,
        cb_stress_suite, CBStressResult,
        cb_capital, CBCapitalResult,
        cb_hedge_recommendations, CBHedgeRecommendation,
        CBLifecycle,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.options.convertible_bond import (
    ConvertibleBond, ConvertibleResult,
    convertible_delta_hedge, DeltaHedgeResult,
)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

@dataclass
class CBRiskMetrics:
    """Convertible bond risk metrics — hybrid Greeks."""
    pv: float
    bond_floor: float
    conversion_value: float
    conversion_premium: float       # (price - conv_value) / conv_value
    parity: float                   # conv_value / notional
    equity_delta: float             # ∂V/∂S
    equity_gamma: float             # ∂²V/∂S²
    vega: float                     # ∂V/∂σ (per 1% vol)
    credit_cs01: float              # ∂V per 1bp credit spread
    rate_dv01: float                # ∂V per 1bp rate shift
    notional: float

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "bond_floor": self.bond_floor,
            "conversion_value": self.conversion_value,
            "conversion_premium": self.conversion_premium,
            "parity": self.parity, "equity_delta": self.equity_delta,
            "equity_gamma": self.equity_gamma, "vega": self.vega,
            "credit_cs01": self.credit_cs01, "rate_dv01": self.rate_dv01,
            "notional": self.notional,
        }


def cb_risk_metrics(
    cb: ConvertibleBond,
    spot: float,
    rate: float,
    equity_vol: float,
    credit_spread: float = 0.0,
    dividend_yield: float = 0.0,
    bump_spot: float = 0.01,
    bump_vol: float = 0.01,
    bump_spread: float = 0.0001,
    bump_rate: float = 0.0001,
    n_paths: int = 10_000,
    n_steps: int | None = None,
    seed: int | None = 42,
) -> CBRiskMetrics:
    """Compute convertible bond risk metrics via bump-and-reprice.

    All Greeks computed with common seed for noise reduction.
    """
    base = cb.price(spot, rate, equity_vol, credit_spread,
                    dividend_yield, n_paths, n_steps, seed)

    # Equity delta & gamma
    spot_up = spot * (1 + bump_spot)
    spot_dn = spot * (1 - bump_spot)
    r_up = cb.price(spot_up, rate, equity_vol, credit_spread,
                    dividend_yield, n_paths, n_steps, seed)
    r_dn = cb.price(spot_dn, rate, equity_vol, credit_spread,
                    dividend_yield, n_paths, n_steps, seed)
    delta = (r_up.price - r_dn.price) / (spot_up - spot_dn)
    gamma = (r_up.price - 2 * base.price + r_dn.price) / ((spot * bump_spot) ** 2)

    # Vega (per 1% vol move)
    r_vol_up = cb.price(spot, rate, equity_vol + bump_vol, credit_spread,
                        dividend_yield, n_paths, n_steps, seed)
    vega = (r_vol_up.price - base.price) / (bump_vol * 100)

    # Credit CS01 (per 1bp spread)
    r_cs_up = cb.price(spot, rate, equity_vol, credit_spread + bump_spread,
                       dividend_yield, n_paths, n_steps, seed)
    cs01 = r_cs_up.price - base.price

    # Rate DV01 (per 1bp rate shift)
    r_rate_up = cb.price(spot, rate + bump_rate, equity_vol, credit_spread,
                         dividend_yield, n_paths, n_steps, seed)
    dv01 = r_rate_up.price - base.price

    return CBRiskMetrics(
        pv=base.price,
        bond_floor=base.bond_floor,
        conversion_value=base.conversion_value,
        conversion_premium=base.conversion_premium,
        parity=base.parity,
        equity_delta=delta,
        equity_gamma=gamma,
        vega=vega,
        credit_cs01=cs01,
        rate_dv01=dv01,
        notional=cb.notional,
    )


# ---------------------------------------------------------------------------
# Book management
# ---------------------------------------------------------------------------

@dataclass
class CBBookEntry:
    """A single convertible bond position."""
    trade_id: str
    cb: ConvertibleBond
    spot: float
    rate: float
    equity_vol: float
    credit_spread: float = 0.0
    dividend_yield: float = 0.0
    issuer: str = ""
    underlying: str = ""
    direction: int = 1          # +1 long, -1 short
    hedge_ratio: float = 0.0   # current delta hedge fraction



    def to_dict(self) -> dict:
        return vars(self)
class CBBook:
    """Convertible bond position book."""

    def __init__(self, name: str = "cb_book"):
        self.name = name
        self._entries: list[CBBookEntry] = []

    def add(self, entry: CBBookEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[CBBookEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def total_notional(self) -> float:
        return sum(e.cb.notional for e in self._entries)

    def by_issuer(self) -> dict[str, list[CBBookEntry]]:
        result: dict[str, list[CBBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.issuer, []).append(e)
        return result

    def by_underlying(self) -> dict[str, list[CBBookEntry]]:
        result: dict[str, list[CBBookEntry]] = {}
        for e in self._entries:
            result.setdefault(e.underlying, []).append(e)
        return result

    def aggregate_risk(self, n_paths: int = 5_000, seed: int = 42) -> dict:
        """Aggregate risk across all positions."""
        total_pv = 0.0
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_cs01 = 0.0
        total_dv01 = 0.0
        total_notional = 0.0

        for e in self._entries:
            rm = cb_risk_metrics(e.cb, e.spot, e.rate, e.equity_vol,
                                e.credit_spread, e.dividend_yield,
                                n_paths=n_paths, seed=seed)
            d = e.direction
            total_pv += d * rm.pv
            total_delta += d * rm.equity_delta * e.cb.notional
            total_gamma += d * rm.equity_gamma * e.cb.notional
            total_vega += d * rm.vega * e.cb.notional
            total_cs01 += d * rm.credit_cs01 * e.cb.notional / 100
            total_dv01 += d * rm.rate_dv01 * e.cb.notional / 100
            total_notional += e.cb.notional

        return {
            "total_pv": total_pv,
            "total_delta": total_delta,
            "total_gamma": total_gamma,
            "total_vega": total_vega,
            "total_cs01": total_cs01,
            "total_dv01": total_dv01,
            "n_positions": len(self._entries),
            "total_notional": total_notional,
        }


# ---------------------------------------------------------------------------
# Carry decomposition
# ---------------------------------------------------------------------------

@dataclass
class CBCarryDecomposition:
    """Convertible bond carry attribution.

    For a delta-hedged CB position:
    - coupon_carry: coupon income from the bond
    - funding_cost: repo/balance sheet cost of holding the bond
    - short_rebate: interest earned on short equity hedge proceeds
    - gamma_carry: expected realised gamma P&L (½ Γ σ² S² dt)
    - net_carry: total daily carry
    """
    coupon_carry: float
    funding_cost: float
    short_rebate: float
    gamma_carry: float
    net_carry: float
    horizon_days: int

    def to_dict(self) -> dict:
        return {
            "coupon": self.coupon_carry, "funding": self.funding_cost,
            "short_rebate": self.short_rebate, "gamma": self.gamma_carry,
            "net": self.net_carry, "horizon_days": self.horizon_days,
        }


def cb_carry_decomposition(
    cb: ConvertibleBond,
    spot: float,
    rate: float,
    equity_vol: float,
    credit_spread: float = 0.0,
    dividend_yield: float = 0.0,
    repo_rate: float = 0.04,
    borrow_cost: float = 0.0,
    horizon_days: int = 1,
    n_paths: int = 5_000,
    seed: int = 42,
) -> CBCarryDecomposition:
    """Decompose convertible bond carry for a delta-hedged position."""
    dt = horizon_days / 365.0
    rm = cb_risk_metrics(cb, spot, rate, equity_vol, credit_spread,
                         dividend_yield, n_paths=n_paths, seed=seed)

    # Coupon carry
    coupon_carry = cb.notional * cb.coupon_rate * dt

    # Funding cost (holding bond)
    funding_cost = rm.pv * repo_rate * dt

    # Short rebate: delta hedge short generates cash earning short rate
    hedge_notional = abs(rm.equity_delta) * spot * cb.notional / 100
    short_rebate = hedge_notional * (rate - borrow_cost) * dt

    # Gamma carry: expected realised gamma P&L
    # E[gamma P&L] = ½ × Γ × σ² × S² × dt
    gamma_carry = 0.5 * rm.equity_gamma * equity_vol**2 * spot**2 * dt * cb.notional / 100

    net = coupon_carry - funding_cost + short_rebate + gamma_carry

    return CBCarryDecomposition(
        coupon_carry=coupon_carry,
        funding_cost=funding_cost,
        short_rebate=short_rebate,
        gamma_carry=gamma_carry,
        net_carry=net,
        horizon_days=horizon_days,
    )


# ---------------------------------------------------------------------------
# Daily P&L
# ---------------------------------------------------------------------------

@dataclass
class CBDailyPnL:
    """Convertible bond daily P&L attribution."""
    date: date
    total: float
    delta_pnl: float        # equity move × delta
    gamma_pnl: float        # ½ × gamma × (ΔS)²
    vega_pnl: float         # vol move × vega
    credit_pnl: float       # spread move × CS01
    rate_pnl: float         # rate move × DV01
    carry_pnl: float
    unexplained: float

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "total": self.total,
            "delta": self.delta_pnl, "gamma": self.gamma_pnl,
            "vega": self.vega_pnl, "credit": self.credit_pnl,
            "rate": self.rate_pnl, "carry": self.carry_pnl,
            "unexplained": self.unexplained,
        }


def cb_daily_pnl(
    cb: ConvertibleBond,
    spot_t0: float, spot_t1: float,
    rate_t0: float, rate_t1: float,
    vol_t0: float, vol_t1: float,
    cs_t0: float, cs_t1: float,
    pnl_date: date,
    dividend_yield: float = 0.0,
    repo_rate: float = 0.04,
    n_paths: int = 5_000,
    seed: int = 42,
) -> CBDailyPnL:
    """Daily P&L attribution for a convertible bond position."""
    rm_t0 = cb_risk_metrics(cb, spot_t0, rate_t0, vol_t0, cs_t0,
                            dividend_yield, n_paths=n_paths, seed=seed)
    pv_t1 = cb.price(spot_t1, rate_t1, vol_t1, cs_t1,
                     dividend_yield, n_paths, seed=seed).price

    total = pv_t1 - rm_t0.pv

    # Attribution
    dS = spot_t1 - spot_t0
    delta_pnl = rm_t0.equity_delta * dS
    gamma_pnl = 0.5 * rm_t0.equity_gamma * dS**2
    vega_pnl = rm_t0.vega * (vol_t1 - vol_t0) * 100   # vega is per 1%
    credit_pnl = rm_t0.credit_cs01 * (cs_t1 - cs_t0) * 1e4  # CS01 is per 1bp
    rate_pnl = rm_t0.rate_dv01 * (rate_t1 - rate_t0) * 1e4

    carry = cb_carry_decomposition(cb, spot_t0, rate_t0, vol_t0, cs_t0,
                                   dividend_yield, repo_rate, n_paths=n_paths, seed=seed)
    carry_pnl = carry.net_carry

    explained = delta_pnl + gamma_pnl + vega_pnl + credit_pnl + rate_pnl + carry_pnl
    unexplained = total - explained

    return CBDailyPnL(
        date=pnl_date, total=total,
        delta_pnl=delta_pnl, gamma_pnl=gamma_pnl,
        vega_pnl=vega_pnl, credit_pnl=credit_pnl,
        rate_pnl=rate_pnl, carry_pnl=carry_pnl,
        unexplained=unexplained,
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class CBDashboard:
    """Convertible bond desk morning summary."""
    date: date
    n_positions: int
    total_notional: float
    total_pv: float
    total_delta: float
    total_gamma: float
    total_vega: float
    total_cs01: float
    avg_conversion_premium: float
    avg_parity: float

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "notional": self.total_notional, "pv": self.total_pv,
            "delta": self.total_delta, "gamma": self.total_gamma,
            "vega": self.total_vega, "cs01": self.total_cs01,
            "avg_premium": self.avg_conversion_premium,
            "avg_parity": self.avg_parity,
        }


def cb_dashboard(
    book: CBBook,
    reference_date: date,
    n_paths: int = 5_000,
    seed: int = 42,
) -> CBDashboard:
    """Build convertible bond desk morning dashboard."""
    risk = book.aggregate_risk(n_paths, seed)

    premiums, parities = [], []
    for e in book.entries:
        r = e.cb.price(e.spot, e.rate, e.equity_vol, e.credit_spread,
                       e.dividend_yield, n_paths, seed=seed)
        premiums.append(r.conversion_premium)
        parities.append(r.parity)

    return CBDashboard(
        date=reference_date,
        n_positions=risk["n_positions"],
        total_notional=risk["total_notional"],
        total_pv=risk["total_pv"],
        total_delta=risk["total_delta"],
        total_gamma=risk["total_gamma"],
        total_vega=risk["total_vega"],
        total_cs01=risk["total_cs01"],
        avg_conversion_premium=float(sum(premiums) / len(premiums)) if premiums else 0.0,
        avg_parity=float(sum(parities) / len(parities)) if parities else 0.0,
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class CBStressResult:
    scenario: str
    description: str
    pnl: float

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "description": self.description,
                "pnl": self.pnl}


def cb_stress_suite(
    book: CBBook,
    n_paths: int = 5_000,
    seed: int = 42,
) -> list[CBStressResult]:
    """Parametric stress scenarios for the CB book."""
    risk = book.aggregate_risk(n_paths, seed)
    delta = risk["total_delta"]
    gamma = risk["total_gamma"]
    vega = risk["total_vega"]
    cs01 = risk["total_cs01"]
    dv01 = risk["total_dv01"]

    # Representative spot for gamma calculation
    spots = [e.spot for e in book.entries]
    avg_spot = sum(spots) / len(spots) if spots else 100.0

    scenarios = [
        ("equity_dn_10", "Equity -10%",
         delta * (-0.10 * avg_spot) + 0.5 * gamma * (0.10 * avg_spot)**2),
        ("equity_up_10", "Equity +10%",
         delta * (0.10 * avg_spot) + 0.5 * gamma * (0.10 * avg_spot)**2),
        ("equity_dn_25", "Equity -25%",
         delta * (-0.25 * avg_spot) + 0.5 * gamma * (0.25 * avg_spot)**2),
        ("vol_up_5", "Vol +5%", vega * 5),
        ("vol_dn_5", "Vol -5%", vega * -5),
        ("spread_up_100", "Credit spreads +100bp", cs01 * 100),
        ("spread_dn_50", "Credit spreads -50bp", cs01 * -50),
        ("rates_up_100", "Rates +100bp", dv01 * 100),
        ("combined_risk_off", "Risk-off: equity -15%, vol +5%, spread +50bp",
         delta * (-0.15 * avg_spot) + 0.5 * gamma * (0.15 * avg_spot)**2
         + vega * 5 + cs01 * 50),
    ]
    return [CBStressResult(n, d, p) for n, d, p in scenarios]


# ---------------------------------------------------------------------------
# Capital
# ---------------------------------------------------------------------------

@dataclass
class CBCapitalResult:
    """Regulatory capital for a convertible bond position.

    Hybrid: GIRR (rate) + CSR (credit) + EQ (equity).
    """
    ead: float
    rwa: float
    capital: float
    girr_charge: float      # rate risk
    csr_charge: float       # credit spread risk
    eq_charge: float        # equity risk

    def to_dict(self) -> dict:
        return {
            "ead": self.ead, "rwa": self.rwa, "capital": self.capital,
            "girr": self.girr_charge, "csr": self.csr_charge, "eq": self.eq_charge,
        }


def cb_capital(
    cb: ConvertibleBond,
    spot: float,
    rate: float,
    equity_vol: float,
    credit_spread: float = 0.0,
    dividend_yield: float = 0.0,
    counterparty_rw: float = 0.20,
    n_paths: int = 5_000,
    seed: int = 42,
) -> CBCapitalResult:
    """SA-CCR / FRTB-SA capital for a convertible bond."""
    rm = cb_risk_metrics(cb, spot, rate, equity_vol, credit_spread,
                         dividend_yield, n_paths=n_paths, seed=seed)

    # GIRR: rate DV01 × risk weight
    girr_rw = 0.015  # ~150bp standardised
    girr = abs(rm.rate_dv01) * girr_rw * cb.notional / 100

    # CSR: CS01 × risk weight
    csr_rw = 0.05  # ~500bp for corporates
    csr = abs(rm.credit_cs01) * csr_rw * cb.notional / 100

    # EQ: delta × spot × risk weight
    eq_rw = 0.25  # 25% for individual equities
    eq_charge = abs(rm.equity_delta) * spot * eq_rw * cb.notional / 100

    total_charge = math.sqrt(girr**2 + csr**2 + eq_charge**2)

    # SA-CCR EAD
    mtm = max(rm.pv * cb.notional / 100, 0)
    sf = 0.005
    mf = math.sqrt(min(cb.maturity_years, 1.0))
    ead = 1.4 * (mtm + cb.notional * sf * mf)
    rwa = ead * counterparty_rw
    capital = rwa * 0.08

    return CBCapitalResult(
        ead=ead, rwa=rwa, capital=capital,
        girr_charge=girr, csr_charge=csr, eq_charge=eq_charge,
    )


# ---------------------------------------------------------------------------
# Hedge recommendations
# ---------------------------------------------------------------------------

@dataclass
class CBHedgeRecommendation:
    risk_type: str
    current: float
    limit: float
    breach_pct: float
    action: str

    def to_dict(self) -> dict:
        return {"risk": self.risk_type, "current": self.current,
                "limit": self.limit, "breach_pct": self.breach_pct,
                "action": self.action}


def cb_hedge_recommendations(
    book: CBBook,
    delta_limit: float = 500_000,
    gamma_limit: float = 50_000,
    vega_limit: float = 100_000,
    cs01_limit: float = 50_000,
    n_paths: int = 5_000,
    seed: int = 42,
) -> list[CBHedgeRecommendation]:
    """Hedge recommendations for the CB book."""
    risk = book.aggregate_risk(n_paths, seed)
    recs = []

    checks = [
        ("equity_delta", abs(risk["total_delta"]), delta_limit,
         "Adjust equity hedge (buy/sell shares)"),
        ("equity_gamma", abs(risk["total_gamma"]), gamma_limit,
         "Trade options to flatten gamma"),
        ("vega", abs(risk["total_vega"]), vega_limit,
         "Trade variance swaps or options to reduce vol exposure"),
        ("credit_cs01", abs(risk["total_cs01"]), cs01_limit,
         "Buy/sell CDS to hedge credit spread risk"),
    ]

    for risk_type, current, limit, action in checks:
        if limit > 0 and current > limit * 0.75:
            recs.append(CBHedgeRecommendation(
                risk_type, current, limit, current / limit, action))

    # Concentration by underlying
    by_und = book.by_underlying()
    total = risk["total_notional"]
    for und, entries in by_und.items():
        und_notional = sum(e.cb.notional for e in entries)
        if total > 0 and und_notional / total > 0.30:
            recs.append(CBHedgeRecommendation(
                "concentration", und_notional / total, 0.30,
                (und_notional / total) / 0.30,
                f"Reduce concentration in {und}"))

    return recs


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class CBEventType:
    MATURITY = "maturity"
    COUPON = "coupon"
    CONVERSION_TRIGGER = "conversion_trigger"
    SOFT_CALL_TRIGGER = "soft_call_trigger"
    PARITY_ALERT = "parity_alert"


class CBLifecycle:
    """Lifecycle management for convertible bond positions."""

    def __init__(self, cb: ConvertibleBond, trade_id: str = ""):
        self._cb = cb
        self._trade_id = trade_id
        self._events: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return sorted(self._events, key=lambda x: x.get("date", ""))

    def maturity_alert(self, remaining_years: float, alert_years: float = 0.5) -> dict | None:
        if 0 < remaining_years <= alert_years:
            return {
                "type": CBEventType.MATURITY,
                "remaining_years": remaining_years,
                "action": "Review conversion vs hold to maturity",
            }
        return None

    def parity_alert(self, spot: float, threshold: float = 1.2) -> dict | None:
        """Alert when parity is significantly above or below 1."""
        parity = self._cb.parity(spot)
        if parity >= threshold:
            return {
                "type": CBEventType.PARITY_ALERT,
                "parity": parity,
                "message": f"Deep ITM (parity={parity:.2f}) — likely to convert",
            }
        if parity <= 1.0 / threshold:
            return {
                "type": CBEventType.PARITY_ALERT,
                "parity": parity,
                "message": f"Deep OTM (parity={parity:.2f}) — trading as straight bond",
            }
        return None

    def record_event(self, event_date: date, event_type: str, details: dict) -> dict:
        event = {"type": event_type, "date": event_date.isoformat(), **details}
        self._events.append(event)
        return event
