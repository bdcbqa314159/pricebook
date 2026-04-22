"""Commodity term structure trading: calendar spreads, steepeners, butterflies.

Builds on :mod:`pricebook.futures` (``calendar_spread``, ``roll_yield``,
``contango_or_backwardation``) with structured trade objects and
DV01-neutral constructions.

* :class:`CommodityCalendarSpread` — matched-notional long near / short far.
* :class:`CommoditySteepener` — long far / short near.
* :class:`CommodityButterfly` — long 2 × mid, short 1 × near + 1 × far.
* :func:`dv01_neutral_quantity` — far-leg quantity for flat parallel sensitivity.
* :func:`curve_structure_monitor` — contango / backwardation / mixed snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


# ---- Calendar spread ----

@dataclass
class CommodityCalendarSpread:
    """Calendar spread with matched quantities on each leg.

    ``direction = +1`` is long near-month / short deferred.
    ``direction = -1`` is the reverse.
    """
    commodity: str
    near_delivery: date
    far_delivery: date
    quantity: float
    direction: int = 1

    def pv(self, curve: dict[date, float]) -> float:
        """PV = direction × qty × (near_fwd − far_fwd)."""
        near = curve.get(self.near_delivery, 0.0)
        far = curve.get(self.far_delivery, 0.0)
        return self.direction * self.quantity * (near - far)

    def parallel_exposure(self) -> float:
        """Sensitivity to a +1 parallel shift of the entire curve.

        Zero by construction: both legs have the same quantity, so a
        uniform shift cancels.
        """
        return 0.0


# ---- Steepener / flattener ----

@dataclass
class CommoditySteepener:
    """Long far / short near — profits from curve steepening.

    ``direction = +1``  steepener (long far, short near).
    ``direction = -1``  flattener (short far, long near).
    """
    commodity: str
    near_delivery: date
    far_delivery: date
    quantity: float
    direction: int = 1

    def pv(self, curve: dict[date, float]) -> float:
        """PV = direction × qty × (far_fwd − near_fwd)."""
        near = curve.get(self.near_delivery, 0.0)
        far = curve.get(self.far_delivery, 0.0)
        return self.direction * self.quantity * (far - near)

    def parallel_exposure(self) -> float:
        """Zero for matched quantities."""
        return 0.0


# ---- Butterfly ----

@dataclass
class CommodityButterfly:
    """Butterfly: long 2 × mid, short 1 × near + 1 × far.

    ``direction = +1``  long curvature (belly rises relative to wings).
    ``direction = -1``  short curvature.

    Weights are 1 : 2 : 1, so the net position is zero and the
    sensitivity to a parallel shift is zero.
    """
    commodity: str
    near_delivery: date
    mid_delivery: date
    far_delivery: date
    quantity: float
    direction: int = 1

    def pv(self, curve: dict[date, float]) -> float:
        """PV = direction × qty × (2 × mid − near − far)."""
        near = curve.get(self.near_delivery, 0.0)
        mid = curve.get(self.mid_delivery, 0.0)
        far = curve.get(self.far_delivery, 0.0)
        return self.direction * self.quantity * (2.0 * mid - near - far)

    def parallel_exposure(self) -> float:
        """Zero: −1 + 2 − 1 = 0."""
        return 0.0

    def steepener_exposure(self) -> float:
        """Time-weighted net position (proxy for steepener sensitivity).

        For evenly spaced deliveries this is zero:
            −1 × t_near  +  2 × t_mid  −  1 × t_far = 0
        when ``t_mid = (t_near + t_far) / 2``.

        Returns a value in day-units × quantity × direction.
        """
        t_n = self.near_delivery.toordinal()
        t_m = self.mid_delivery.toordinal()
        t_f = self.far_delivery.toordinal()
        return float(-t_n + 2 * t_m - t_f) * self.quantity * self.direction


# ---- DV01-neutral helper ----

def dv01_neutral_quantity(
    near_qty: float,
    near_dv01: float = 1.0,
    far_dv01: float = 1.0,
) -> float:
    """Far-leg quantity that makes a two-leg trade DV01-neutral.

    For matched-notional calendars (``near_dv01 == far_dv01``) this
    returns ``near_qty``. When per-unit DV01 differs across deliveries:

        far_qty = near_qty × near_dv01 / far_dv01
    """
    if abs(far_dv01) < 1e-15:
        return near_qty
    return near_qty * near_dv01 / far_dv01


# ---- Curve structure monitor ----

@dataclass
class CurveStructureSnapshot:
    """Full term-structure snapshot for a single commodity."""
    commodity: str
    valuation_date: date
    deliveries: list[date]
    forwards: list[float]
    spreads: list[float]
    structure: str  # "contango", "backwardation", "mixed", "flat"

    @property
    def n_deliveries(self) -> int:
        return len(self.deliveries)


def curve_structure_monitor(
    commodity: str,
    valuation_date: date,
    curve: dict[date, float],
) -> CurveStructureSnapshot:
    """Analyse the full term structure of a commodity curve.

    Returns sorted deliveries, forward prices, consecutive calendar
    spreads (near − far for each pair), and an overall classification.
    """
    sorted_items = sorted(curve.items(), key=lambda kv: kv[0])
    deliveries = [d for d, _ in sorted_items]
    forwards = [f for _, f in sorted_items]

    spreads: list[float] = []
    n_contango = 0
    n_backwardation = 0
    for i in range(len(forwards) - 1):
        s = forwards[i] - forwards[i + 1]
        spreads.append(s)
        if s > 0:
            n_backwardation += 1
        elif s < 0:
            n_contango += 1

    if n_contango > 0 and n_backwardation > 0:
        structure = "mixed"
    elif n_backwardation > 0:
        structure = "backwardation"
    elif n_contango > 0:
        structure = "contango"
    else:
        structure = "flat"

    return CurveStructureSnapshot(
        commodity=commodity,
        valuation_date=valuation_date,
        deliveries=deliveries,
        forwards=forwards,
        spreads=spreads,
        structure=structure,
    )


# ---- Roll-down analysis ----

@dataclass
class RollDownResult:
    """Commodity curve roll-down analysis."""
    current_forward: float
    rolled_forward: float
    roll_pnl_per_unit: float
    roll_days: int
    structure: str  # "contango" or "backwardation"


def commodity_roll_down(
    delivery: date,
    forward_curve: "CommodityForwardCurve",
    roll_days: int = 30,
) -> RollDownResult:
    """Roll-down P&L: what happens as time passes and the position ages.

    If the curve is in backwardation, rolling forward = rolling to higher
    prices = positive carry. In contango, rolling forward = negative carry.

    Args:
        delivery: delivery date of the position.
        forward_curve: current forward curve.
        roll_days: number of days to roll forward.

    Returns:
        RollDownResult with P&L per unit.
    """
    from pricebook.commodity import CommodityForwardCurve
    from datetime import timedelta

    current = forward_curve.forward(delivery)
    rolled_curve = forward_curve.roll_down(roll_days)
    rolled = rolled_curve.forward(delivery)
    pnl = rolled - current

    structure = "backwardation" if pnl > 0 else "contango" if pnl < 0 else "flat"

    return RollDownResult(
        current_forward=current,
        rolled_forward=rolled,
        roll_pnl_per_unit=pnl,
        roll_days=roll_days,
        structure=structure,
    )
