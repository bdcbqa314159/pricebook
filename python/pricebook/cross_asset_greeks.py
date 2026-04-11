"""Cross-asset Greeks: unified attribution and multi-factor stress.

* :func:`greek_attribution` — delta/gamma/vega/theta P&L by asset class,
  with carry-vs-convexity decomposition.
* :func:`multi_factor_stress` — cross-asset scenario P&L.
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.options_book import OptionsBook, OptionEntry


# ---- Unified attribution ----

@dataclass
class GreekAttribution:
    """P&L attribution for one asset class."""
    asset_class: str
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    rho_pnl: float
    total_pnl: float

    @property
    def carry(self) -> float:
        """Carry = theta (time decay)."""
        return self.theta_pnl

    @property
    def convexity(self) -> float:
        """Convexity = gamma (second-order spot)."""
        return self.gamma_pnl


@dataclass
class BookGreekAttribution:
    """Full book attribution across all asset classes."""
    total_pnl: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    rho_pnl: float
    by_asset_class: list[GreekAttribution]

    @property
    def carry(self) -> float:
        return self.theta_pnl

    @property
    def convexity(self) -> float:
        return self.gamma_pnl


def greek_attribution(
    book: OptionsBook,
    spot_changes: dict[str, float],
    vol_changes: dict[str, float],
    rate_change: float = 0.0,
    dt_days: float = 1.0,
) -> BookGreekAttribution:
    """Unified Greek attribution across all asset classes.

    For each entry:
        delta_pnl = delta × ΔS
        gamma_pnl = 0.5 × gamma × ΔS²
        vega_pnl  = vega × Δσ
        theta_pnl = theta × Δt
        rho_pnl   = rho × Δr
    """
    agg: dict[str, dict[str, float]] = {}

    for e in book.entries:
        ac = e.asset_class
        if ac not in agg:
            agg[ac] = {"delta": 0.0, "gamma": 0.0, "vega": 0.0,
                       "theta": 0.0, "rho": 0.0}

        ds = spot_changes.get(e.underlying, 0.0)
        dv = vol_changes.get(e.underlying, 0.0)

        agg[ac]["delta"] += e.delta * ds
        agg[ac]["gamma"] += 0.5 * e.gamma * ds * ds
        agg[ac]["vega"] += e.vega * dv
        agg[ac]["theta"] += e.theta * dt_days
        agg[ac]["rho"] += e.rho * rate_change

    by_ac = []
    for ac, d in sorted(agg.items()):
        total = d["delta"] + d["gamma"] + d["vega"] + d["theta"] + d["rho"]
        by_ac.append(GreekAttribution(
            ac, d["delta"], d["gamma"], d["vega"], d["theta"], d["rho"], total,
        ))

    return BookGreekAttribution(
        total_pnl=sum(a.total_pnl for a in by_ac),
        delta_pnl=sum(a.delta_pnl for a in by_ac),
        gamma_pnl=sum(a.gamma_pnl for a in by_ac),
        vega_pnl=sum(a.vega_pnl for a in by_ac),
        theta_pnl=sum(a.theta_pnl for a in by_ac),
        rho_pnl=sum(a.rho_pnl for a in by_ac),
        by_asset_class=by_ac,
    )


# ---- Multi-factor stress ----

@dataclass
class StressScenario:
    """A named multi-factor stress scenario."""
    name: str
    spot_shocks: dict[str, float]    # underlying → ΔS
    vol_shocks: dict[str, float]     # underlying → Δσ
    rate_shock: float = 0.0


@dataclass
class StressResult:
    """P&L under a stress scenario."""
    scenario_name: str
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    rho_pnl: float
    total_pnl: float


def multi_factor_stress(
    book: OptionsBook,
    scenario: StressScenario,
    dt_days: float = 0.0,
) -> StressResult:
    """Compute P&L under a multi-factor stress scenario.

    Applies spot shocks, vol shocks, and rate shock simultaneously.
    """
    attrib = greek_attribution(
        book, scenario.spot_shocks, scenario.vol_shocks,
        scenario.rate_shock, dt_days,
    )
    return StressResult(
        scenario_name=scenario.name,
        delta_pnl=attrib.delta_pnl,
        gamma_pnl=attrib.gamma_pnl,
        vega_pnl=attrib.vega_pnl,
        theta_pnl=attrib.theta_pnl,
        rho_pnl=attrib.rho_pnl,
        total_pnl=attrib.total_pnl,
    )
