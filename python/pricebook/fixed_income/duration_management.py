"""Duration management: DV01 ladder, curve risk decomposition, hedging.

Mid-office tooling for managing the rate exposure of a bond portfolio.

* :class:`DV01Ladder` — per-tenor key-rate DV01 exposure.
* :func:`parallel_dv01` — total book DV01 (sum of key rates).
* :func:`curve_dv01` — 2s10s steepener and butterfly risk.
* :func:`duration_target_tracking` — book duration vs mandate target.
* :func:`optimal_krd_hedge` — solve for hedge quantities to flatten the
  DV01 ladder using N hedge instruments.
* :func:`barbell_vs_bullet` — compare convexity of barbell vs bullet.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---- DV01 ladder ----

@dataclass
class KeyRateDV01:
    """DV01 at a single tenor point."""
    tenor_label: str
    dv01: float


@dataclass
class DV01Ladder:
    """Per-tenor DV01 exposure for a book."""
    book_name: str
    rungs: list[KeyRateDV01]

    @property
    def total_dv01(self) -> float:
        """Parallel DV01 = sum of all key-rate DV01s."""
        return sum(r.dv01 for r in self.rungs)

    def dv01_at(self, tenor: str) -> float:
        """DV01 at a specific tenor, or 0 if not present."""
        for r in self.rungs:
            if r.tenor_label == tenor:
                return r.dv01
        return 0.0

    def as_dict(self) -> dict[str, float]:
        return {r.tenor_label: r.dv01 for r in self.rungs}


def build_dv01_ladder(
    book_name: str,
    tenor_dv01s: dict[str, float],
) -> DV01Ladder:
    """Build a DV01 ladder from a tenor → DV01 mapping."""
    rungs = [
        KeyRateDV01(tenor, dv01)
        for tenor, dv01 in sorted(tenor_dv01s.items())
    ]
    return DV01Ladder(book_name=book_name, rungs=rungs)


# ---- Curve risk decomposition ----

@dataclass
class CurveDV01:
    """Decomposed curve risk."""
    parallel_dv01: float
    steepener_2s10s: float
    butterfly_2s5s10s: float


def curve_dv01(
    dv01_2y: float,
    dv01_5y: float,
    dv01_10y: float,
) -> CurveDV01:
    """Decompose curve risk into parallel, 2s10s steepener, and butterfly.

    * Parallel = DV01_2Y + DV01_5Y + DV01_10Y.
    * 2s10s steepener = DV01_10Y − DV01_2Y (positive = long the steepener).
    * 2s5s10s butterfly = DV01_2Y + DV01_10Y − 2 × DV01_5Y
      (positive = long wings vs belly).
    """
    return CurveDV01(
        parallel_dv01=dv01_2y + dv01_5y + dv01_10y,
        steepener_2s10s=dv01_10y - dv01_2y,
        butterfly_2s5s10s=dv01_2y + dv01_10y - 2 * dv01_5y,
    )


# ---- Duration target tracking ----

@dataclass
class DurationTarget:
    """Book duration vs mandate target."""
    book_duration: float
    target_duration: float
    deviation: float
    within_band: bool


def duration_target_tracking(
    book_duration: float,
    target_duration: float,
    band: float = 0.5,
) -> DurationTarget:
    """Compare the book's weighted-average duration to a mandate target.

    Args:
        book_duration: current book duration.
        target_duration: mandate target.
        band: allowed deviation (e.g. ±0.5 years).
    """
    deviation = book_duration - target_duration
    return DurationTarget(
        book_duration=book_duration,
        target_duration=target_duration,
        deviation=deviation,
        within_band=abs(deviation) <= band,
    )


# ---- Optimal KRD hedge ----

@dataclass
class HedgeInstrumentKRD:
    """A hedge instrument with per-tenor DV01 profile."""
    name: str
    krd: dict[str, float]  # tenor → DV01 per unit


@dataclass
class KRDHedgeAllocation:
    """Quantity of each hedge instrument."""
    instrument: HedgeInstrumentKRD
    quantity: float


def optimal_krd_hedge(
    book_ladder: dict[str, float],
    instruments: list[HedgeInstrumentKRD],
) -> list[KRDHedgeAllocation]:
    """Solve for hedge quantities to flatten the DV01 ladder.

    Uses a least-squares approach when the system is over- or
    under-determined, and exact inversion when it's square.

    Minimises ``||A·x + b||²`` where A is the instrument KRD matrix
    and b is the book DV01 vector.

    Args:
        book_ladder: {tenor → book DV01}.
        instruments: list of hedge instruments with their KRD profiles.

    Returns:
        list of :class:`KRDHedgeAllocation`.
    """
    tenors = sorted(book_ladder.keys())
    n_tenors = len(tenors)
    n_inst = len(instruments)

    if n_inst == 0 or n_tenors == 0:
        return []

    # Build matrix A (n_tenors × n_inst) and vector b (n_tenors)
    import numpy as np
    A = np.zeros((n_tenors, n_inst))
    b = np.array([book_ladder[t] for t in tenors])

    for j, inst in enumerate(instruments):
        for i, t in enumerate(tenors):
            A[i, j] = inst.krd.get(t, 0.0)

    # Solve: minimise ||A·x + b||² → x = -A⁺·b
    x, _, _, _ = np.linalg.lstsq(A, -b, rcond=None)

    return [
        KRDHedgeAllocation(inst, float(x[j]))
        for j, inst in enumerate(instruments)
    ]


def hedged_ladder(
    book_ladder: dict[str, float],
    allocations: list[KRDHedgeAllocation],
) -> dict[str, float]:
    """Compute the residual DV01 ladder after hedging."""
    result = dict(book_ladder)
    for alloc in allocations:
        for tenor, krd in alloc.instrument.krd.items():
            result[tenor] = result.get(tenor, 0.0) + alloc.quantity * krd
    return result


# ---- Barbell vs bullet ----

@dataclass
class BarbellVsBullet:
    """Convexity comparison: barbell (wings) vs bullet (belly)."""
    barbell_duration: float
    bullet_duration: float
    barbell_convexity: float
    bullet_convexity: float
    convexity_advantage: float  # barbell - bullet (positive = barbell wins)
    recommendation: str


def barbell_vs_bullet(
    short_duration: float,
    short_convexity: float,
    short_weight: float,
    long_duration: float,
    long_convexity: float,
    long_weight: float,
    bullet_duration: float,
    bullet_convexity: float,
) -> BarbellVsBullet:
    """Compare a barbell (short + long bond) to a bullet (mid bond).

    The barbell and bullet should have approximately the same duration
    (duration-matched). The barbell typically has higher convexity
    (better protection against large rate moves).

    Args:
        short_* / long_*: duration, convexity, weight of the barbell wings.
        bullet_*: duration, convexity of the belly bond (weight = 1).
    """
    bb_dur = short_weight * short_duration + long_weight * long_duration
    bb_cvx = short_weight * short_convexity + long_weight * long_convexity

    advantage = bb_cvx - bullet_convexity
    if advantage > 0:
        rec = "barbell"
    elif advantage < 0:
        rec = "bullet"
    else:
        rec = "indifferent"

    return BarbellVsBullet(
        barbell_duration=bb_dur,
        bullet_duration=bullet_duration,
        barbell_convexity=bb_cvx,
        bullet_convexity=bullet_convexity,
        convexity_advantage=advantage,
        recommendation=rec,
    )
