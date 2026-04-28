"""Tenor basis: calibration and spread between IBOR tenors.

Given a calibrated short-tenor IBORCurve (e.g. 3M) and basis swap quotes,
bootstraps the long-tenor IBORCurve (e.g. 6M) and extracts the tenor
basis spread term structure.

    from pricebook.tenor_basis import bootstrap_tenor_basis, TenorBasis

    ibor_6m, basis = bootstrap_tenor_basis(
        ref, ibor_3m, ois,
        basis_swap_quotes=[(2Y, 0.0005), (5Y, 0.0010)],
        long_tenor_conventions=EURIBOR_6M_CONVENTIONS)

References:
    Ametrano & Bianchetti (2013), §3.2 — tenor basis bootstrapping.
    Henrard (2014), Ch. 3 — multi-curve framework with tenor basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.basis_swap import BasisSwap
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.ibor_curve import IBORCurve, IBORConventions
from pricebook.interpolation import InterpolationMethod
from pricebook.schedule import Frequency, generate_schedule
from pricebook.solvers import brentq


@dataclass
class TenorBasis:
    """Spread term structure between two IBOR tenors.

    basis(T) = IBOR_long(T) - IBOR_short(T)

    Stored at calibration pillar dates, linearly interpolated.

    Args:
        reference_date: curve date.
        short_tenor: conventions for the short tenor (e.g. 3M).
        long_tenor: conventions for the long tenor (e.g. 6M).
        dates: pillar dates.
        spreads: basis spread at each pillar.
    """
    reference_date: date
    short_tenor: IBORConventions
    long_tenor: IBORConventions
    dates: list[date]
    spreads: list[float]
    day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED

    def spread(self, d: date) -> float:
        """Interpolated tenor basis spread at date d."""
        if not self.dates:
            return 0.0
        if len(self.dates) == 1:
            return self.spreads[0]

        t = year_fraction(self.reference_date, d, self.day_count)
        times = [year_fraction(self.reference_date, dd, self.day_count) for dd in self.dates]

        if t <= times[0]:
            return self.spreads[0]
        if t >= times[-1]:
            return self.spreads[-1]

        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                frac = (t - times[i]) / (times[i + 1] - times[i])
                return self.spreads[i] + frac * (self.spreads[i + 1] - self.spreads[i])

        return self.spreads[-1]

    def to_dict(self) -> dict:
        return {
            "short_tenor": self.short_tenor.name,
            "long_tenor": self.long_tenor.name,
            "n_pillars": len(self.dates),
            "spreads_bp": [s * 10_000 for s in self.spreads],
        }

    def forward_spread(self, start: date, end: date) -> float:
        """Average basis spread over [start, end] (midpoint evaluation)."""
        from datetime import timedelta
        mid_ord = (start.toordinal() + end.toordinal()) // 2
        mid = date.fromordinal(mid_ord)
        return self.spread(mid)


def bootstrap_tenor_basis(
    reference_date: date,
    short_ibor_curve: IBORCurve,
    discount_curve: DiscountCurve,
    basis_swap_quotes: list[tuple[date, float]],
    long_tenor_conventions: IBORConventions,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
) -> tuple[IBORCurve, TenorBasis]:
    """Bootstrap a long-tenor IBOR curve from short-tenor + basis swap quotes.

    At each maturity, solves for the long-tenor df such that the basis swap
    reprices at the quoted spread (Ametrano & Bianchetti 2013, Eq 3.8):

        Σ (F_short(t_i) + s) × τ_i × D(t_i) = Σ F_long(t_j) × τ_j × D(t_j)

    Convention: spread on short leg (3M + s vs 6M flat).
    The spread compensates the short tenor for lower credit risk.

    Args:
        reference_date: curve date.
        short_ibor_curve: calibrated short-tenor IBORCurve (e.g. 3M).
        discount_curve: OIS curve for discounting.
        basis_swap_quotes: [(maturity, spread)] sorted by maturity.
        long_tenor_conventions: conventions for the long tenor (e.g. 6M).
        interpolation: interpolation for the long-tenor curve.

    Returns:
        (long_ibor_curve, tenor_basis): the calibrated long-tenor curve
        and the extracted spread term structure.
    """
    for i in range(1, len(basis_swap_quotes)):
        if basis_swap_quotes[i][0] <= basis_swap_quotes[i - 1][0]:
            raise ValueError("basis_swap_quotes must be sorted by maturity")

    short_conv = short_ibor_curve.conventions
    long_conv = long_tenor_conventions

    pillar_dates: list[date] = []
    pillar_dfs: list[float] = []
    basis_dates: list[date] = []
    basis_spreads: list[float] = []

    for mat, quoted_spread in basis_swap_quotes:
        # Build basis swap: short leg (flat) vs long leg (+ spread)
        # Convention: leg1 = short (e.g. quarterly), leg2 = long (e.g. semi-annual)
        # PV = leg1.pv(disc, short_proj) - leg2.pv(disc, long_proj) = 0
        # The quoted spread goes on leg2 (long tenor pays spread)

        short_sched = generate_schedule(reference_date, mat, short_conv.float_frequency)
        long_sched = generate_schedule(reference_date, mat, long_conv.float_frequency)

        def objective(
            df_guess: float,
            _mat=mat,
            _spread=quoted_spread,
            _short_sched=short_sched,
            _long_sched=long_sched,
        ) -> float:
            # Build trial long-tenor curve
            trial_dates = pillar_dates + [_mat]
            trial_dfs = pillar_dfs + [df_guess]
            long_proj = DiscountCurve(
                reference_date, trial_dates, trial_dfs,
                day_count=DayCountConvention.ACT_365_FIXED,
                interpolation=interpolation,
            )

            # PV of short leg (3M + spread)
            pv_short = 0.0
            for i in range(1, len(_short_sched)):
                d1, d2 = _short_sched[i - 1], _short_sched[i]
                fwd = short_ibor_curve.forward_rate(d1, d2)
                yf = year_fraction(d1, d2, short_conv.float_day_count)
                pv_short += (fwd + _spread) * yf * discount_curve.df(d2)

            # PV of long leg (6M flat)
            pv_long = 0.0
            for i in range(1, len(_long_sched)):
                d1, d2 = _long_sched[i - 1], _long_sched[i]
                ldf1 = long_proj.df(d1)
                ldf2 = long_proj.df(d2)
                yf = year_fraction(d1, d2, long_conv.float_day_count)
                if yf < 1e-10:
                    continue
                fwd = (ldf1 / ldf2 - 1.0) / yf
                pv_long += fwd * yf * discount_curve.df(d2)

            return pv_short - pv_long

        df_solved = brentq(objective, 1e-6, 3.0)
        pillar_dates.append(mat)
        pillar_dfs.append(df_solved)

        # Extract the basis at this maturity
        long_proj = DiscountCurve(
            reference_date, pillar_dates[:], pillar_dfs[:],
            day_count=DayCountConvention.ACT_365_FIXED,
            interpolation=interpolation,
        )
        # Average forward rate difference over the swap tenor
        n_periods = len(long_sched) - 1
        if n_periods > 0:
            total_basis = 0.0
            for i in range(1, len(long_sched)):
                d1, d2 = long_sched[i - 1], long_sched[i]
                long_fwd = (long_proj.df(d1) / long_proj.df(d2) - 1.0) / max(
                    year_fraction(d1, d2, long_conv.float_day_count), 1e-10)
                short_fwd = short_ibor_curve.forward_rate(d1, d2)
                total_basis += long_fwd - short_fwd
            avg_basis = total_basis / n_periods
        else:
            avg_basis = 0.0

        basis_dates.append(mat)
        basis_spreads.append(avg_basis)

    long_curve = DiscountCurve(
        reference_date, pillar_dates, pillar_dfs,
        day_count=DayCountConvention.ACT_365_FIXED,
        interpolation=interpolation,
    )
    long_ibor = IBORCurve(long_curve, long_conv, discount_curve)

    tenor_basis = TenorBasis(
        reference_date=reference_date,
        short_tenor=short_conv,
        long_tenor=long_conv,
        dates=basis_dates,
        spreads=basis_spreads,
    )

    return long_ibor, tenor_basis
