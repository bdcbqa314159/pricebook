"""Curve bootstrapping from deposits and swap par rates."""

from datetime import date
from typing import TYPE_CHECKING

from pricebook.calibration import (
    CalibrationDiagnostics,
    CalibrationResult,
    ObjectiveKind,
    OptimiserSpec,
)
from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.interpolation import InterpolationMethod
from pricebook.core.schedule import Frequency, StubType, generate_schedule
from pricebook.core.solvers import brentq
from pricebook.core.calendar import Calendar, BusinessDayConvention

if TYPE_CHECKING:
    from pricebook.market_data import MarketSnapshot


def bootstrap(
    reference_date: date,
    deposits: list[tuple[date, float]],
    swaps: list[tuple[date, float]],
    fras: list[tuple[date, date, float]] | None = None,
    futures: list[tuple[date, date, float]] | None = None,
    deposit_day_count: DayCountConvention = DayCountConvention.ACT_360,
    fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360,
    float_day_count: DayCountConvention = DayCountConvention.ACT_360,
    fixed_frequency: Frequency = Frequency.SEMI_ANNUAL,
    float_frequency: Frequency = Frequency.QUARTERLY,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
    calendar: Calendar | None = None,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
    hw_convexity_a: float = 0.0,
    hw_convexity_sigma: float = 0.0,
    turn_of_year_spread: float = 0.0,
    *,
    market_snapshot: "MarketSnapshot | None" = None,
) -> DiscountCurve:
    """
    Bootstrap a discount curve from deposits, FRAs, futures, and swap par rates.

    Args:
        reference_date: Curve reference date (today / spot date).
        deposits: List of (maturity_date, rate) for money market deposits.
                  Must be sorted by maturity.
        swaps: List of (maturity_date, par_rate) for vanilla IRS.
               Must be sorted by maturity, all maturities after last deposit.
        fras: Optional list of (start_date, end_date, rate) for FRAs.
              Each FRA implies df(end) = df(start) / (1 + rate × τ).
              Sorted by end_date, between deposits and swaps.
        futures: Optional list of (start_date, end_date, futures_rate) for
                 IR futures. Convexity-adjusted via Hull-White if hw params provided.
        hw_convexity_a: Hull-White mean reversion for futures convexity adjustment.
        hw_convexity_sigma: Hull-White vol for futures convexity adjustment.
        turn_of_year_spread: Additive spread (in rate terms) for periods crossing year-end.
        deposit_day_count: Day count for deposit year fractions.
        fixed_day_count: Day count for the fixed leg of swaps.
        float_day_count: Day count for the floating leg of swaps.
        fixed_frequency: Payment frequency of the fixed leg.
        float_frequency: Payment frequency of the floating leg.
        interpolation: Interpolation method for the resulting curve.
        calendar: Business day calendar.
        convention: Business day convention.

    Returns:
        A DiscountCurve that reprices all input instruments.
    """
    # Validate inputs are sorted
    for i in range(1, len(deposits)):
        if deposits[i][0] <= deposits[i - 1][0]:
            raise ValueError("deposits must be sorted by maturity")
    for i in range(1, len(swaps)):
        if swaps[i][0] <= swaps[i - 1][0]:
            raise ValueError("swaps must be sorted by maturity")
    if deposits and swaps and swaps[0][0] <= deposits[-1][0]:
        raise ValueError("swap maturities must come after deposit maturities")

    pillar_dates: list[date] = []
    pillar_dfs: list[float] = []

    # --- Short end: deposits give discount factors directly ---
    for mat, rate in deposits:
        yf = year_fraction(reference_date, mat, deposit_day_count)
        pillar_dates.append(mat)
        pillar_dfs.append(1.0 / (1.0 + rate * yf))

    # --- Middle: FRAs give df(end) from df(start) ---
    if fras:
        for start_date, end_date, fra_rate in fras:
            if start_date >= end_date:
                raise ValueError(f"FRA start {start_date} must be before end {end_date}")
            tau = year_fraction(start_date, end_date, deposit_day_count)
            # Interpolate df(start) from existing pillars
            # Use ACT_365_FIXED to match the final curve's internal time axis
            if pillar_dates:
                temp_curve = DiscountCurve(
                    reference_date, pillar_dates, pillar_dfs,
                    day_count=DayCountConvention.ACT_365_FIXED, interpolation=interpolation,
                )
                df_start = temp_curve.df(start_date)
            elif start_date == reference_date:
                # FRA starts today — df(start) = 1.0 is the only sensible value.
                df_start = 1.0
            else:
                # Fix T3.18: pre-fix this branch silently used df_start = 1.0
                # for ANY FRA with no preceding deposits, producing a wildly
                # wrong df(end) when start_date is not today.  Refuse to
                # bootstrap without short-end information so the caller is
                # forced to supply a deposit (or use start_date = ref).
                raise ValueError(
                    f"FRA start_date {start_date} is after reference_date "
                    f"{reference_date} but no deposits provided to anchor "
                    f"df(start_date).  Add a deposit covering start_date or "
                    f"use start_date = reference_date."
                )
            # FRA relationship: df(end) = df(start) / (1 + rate × τ)
            df_end = df_start / (1 + fra_rate * tau)
            # Pin df(start) too — otherwise later pillars (swaps) reshape
            # the interpolation between the prior pillar and end_date,
            # changing df(start) on the final curve and breaking the
            # round-trip on the FRA forward (W3, structural local-bootstrap
            # gap). Skip if start_date already equals an existing pillar.
            if start_date != reference_date and start_date not in pillar_dates:
                pillar_dates.append(start_date)
                pillar_dfs.append(df_start)
            pillar_dates.append(end_date)
            pillar_dfs.append(df_end)

    # --- Middle: futures (with convexity adjustment and TOY) ---
    if futures:
        import math as _math
        for start_date, end_date, fut_rate in futures:
            tau = year_fraction(start_date, end_date, deposit_day_count)
            # Hull-White convexity adjustment: futures_rate > forward_rate.
            #
            # Fix T3.17: pre-fix formula
            #   ca = 0.5 · σ² · B(t1, t2) · [B(0, t2) − B(0, t1)]
            # is missing the T_1 factor.  For small a · T this expanded to
            # 0.5 · σ² · (t_end − t_start)², which underestimates the convexity
            # by a factor of (t_end − t_start) / t_start (i.e. ≈ 4 ×
            # under-stated for a 1y-expiry, 3m-tenor Eurodollar future).
            #
            # Post-fix uses the textbook formula
            #   ca = 0.5 · σ² · B(0, T_1) · B(T_1, T_2)
            # which in the small-a limit reduces to the standard
            # 0.5 · σ² · T_1 · (T_2 − T_1) leading-order HW convexity
            # (Hull, Brigo-Mercurio §3.4).
            conv_adj = 0.0
            if hw_convexity_a > 0 and hw_convexity_sigma > 0:
                t_start = year_fraction(reference_date, start_date, deposit_day_count)
                t_end = year_fraction(reference_date, end_date, deposit_day_count)
                def _B(s, t):
                    return (1 - _math.exp(-hw_convexity_a * (t - s))) / hw_convexity_a
                conv_adj = 0.5 * hw_convexity_sigma**2 * _B(0, t_start) * _B(t_start, t_end)
            fwd_rate = fut_rate - conv_adj

            # Turn-of-year: if period crosses Dec 31, add spread
            if turn_of_year_spread > 0 and start_date.year != end_date.year:
                fwd_rate += turn_of_year_spread

            # Chain: df(end) = df(start) / (1 + fwd × τ)
            # Use ACT_365_FIXED to match the final curve's internal time axis
            if pillar_dates:
                temp_curve = DiscountCurve(
                    reference_date, pillar_dates, pillar_dfs,
                    day_count=DayCountConvention.ACT_365_FIXED, interpolation=interpolation,
                )
                df_start = temp_curve.df(start_date)
            elif start_date == reference_date:
                df_start = 1.0
            else:
                # Fix T3.18 (symmetric to FRA path): refuse to anchor
                # df(start) = 1 silently when start ≠ ref.
                raise ValueError(
                    f"Future start_date {start_date} is after reference_date "
                    f"{reference_date} but no deposits provided to anchor "
                    f"df(start_date).  Add a deposit covering start_date."
                )
            df_end = df_start / (1 + fwd_rate * tau)
            # Pin df(start) too — same rationale as the FRA path (W3).
            if start_date != reference_date and start_date not in pillar_dates:
                pillar_dates.append(start_date)
                pillar_dfs.append(df_start)
            pillar_dates.append(end_date)
            pillar_dfs.append(df_end)

    # --- Long end: swap par rates, solved iteratively ---
    for mat, par_rate in swaps:
        fixed_sched = generate_schedule(
            reference_date, mat, fixed_frequency,
            calendar, convention, StubType.SHORT_FRONT, True,
        )
        float_sched = generate_schedule(
            reference_date, mat, float_frequency,
            calendar, convention, StubType.SHORT_FRONT, True,
        )

        def objective(df_guess: float, _mat=mat, _par=par_rate,
                      _fsched=fixed_sched, _flsched=float_sched) -> float:
            """Build trial curve and price the swap — PV should be 0 at par."""
            trial_dates = pillar_dates + [_mat]
            trial_dfs = pillar_dfs + [df_guess]
            trial_curve = DiscountCurve(
                reference_date, trial_dates, trial_dfs,
                day_count=DayCountConvention.ACT_365_FIXED,
                interpolation=interpolation,
            )

            # PV of fixed leg: par_rate * sum(yf_i * df_i)
            pv_fixed = 0.0
            for i in range(1, len(_fsched)):
                yf = year_fraction(_fsched[i - 1], _fsched[i], fixed_day_count)
                pv_fixed += _par * yf * trial_curve.df(_fsched[i])

            # PV of floating leg: sum(forward_rate_i * yf_i * df_i)
            # Compute forward rate using float_day_count for consistency:
            #   F = (df1/df2 - 1) / tau_float
            pv_float = 0.0
            for i in range(1, len(_flsched)):
                d1, d2 = _flsched[i - 1], _flsched[i]
                df1 = trial_curve.df(d1)
                df2 = trial_curve.df(d2)
                yf = year_fraction(d1, d2, float_day_count)
                fwd = (df1 - df2) / (yf * df2)
                pv_float += fwd * yf * df2

            return pv_fixed - pv_float

        # Bracket: df must be between 0 and 1 (or slightly above 1 for negative rates)
        df_solved = brentq(objective, 1e-6, 3.0)  # wider bracket: handles negative rates to -3%

        pillar_dates.append(mat)
        pillar_dfs.append(df_solved)

    curve = DiscountCurve(
        reference_date, pillar_dates, pillar_dfs,
        day_count=DayCountConvention.ACT_365_FIXED,
        interpolation=interpolation,
    )

    # --- Round-trip verification: reprice all inputs ---
    _verify_round_trip(curve, reference_date, deposits, swaps, fras, futures,
                       deposit_day_count, fixed_day_count, float_day_count,
                       fixed_frequency, float_frequency, calendar, convention,
                       hw_convexity_a, hw_convexity_sigma, turn_of_year_spread)

    # --- Attach the canonical CalibrationResult (G1 P1 Slice 5; G1 P2 Slice 2) ---
    curve.calibration_result = _build_bootstrap_calibration_result(
        curve, reference_date, deposits, swaps, fras, futures,
        deposit_day_count, fixed_day_count, float_day_count,
        fixed_frequency, float_frequency, calendar, convention,
        hw_convexity_a, hw_convexity_sigma, turn_of_year_spread,
        interpolation,
        market_snapshot_id=market_snapshot.id if market_snapshot is not None else None,
    )

    return curve


def _build_bootstrap_calibration_result(
    curve, reference_date, deposits, swaps, fras, futures,
    deposit_day_count, fixed_day_count, float_day_count,
    fixed_frequency, float_frequency, calendar, convention,
    hw_convexity_a, hw_convexity_sigma, turn_of_year_spread,
    interpolation,
    market_snapshot_id=None,
) -> CalibrationResult:
    """Build the CalibrationResult artefact for a bootstrapped discount curve.

    Residuals are model-minus-market in rate units (deposits/FRAs/futures) or
    in PV units of par (swaps — model PV_fixed - PV_float, which should be ~0
    by construction). Parameters are the pillar discount factors keyed by
    pillar date.
    """
    import math as _math

    quotes: list[str] = []
    residuals: list[float] = []

    for mat, rate in deposits:
        tau = year_fraction(reference_date, mat, deposit_day_count)
        if tau > 0:
            model_rate = (1.0 / curve.df(mat) - 1.0) / tau
            quotes.append(f"deposit_{mat.isoformat()}")
            residuals.append(model_rate - rate)

    for mat, par_rate in swaps:
        fixed_sched = generate_schedule(
            reference_date, mat, fixed_frequency,
            calendar, convention, StubType.SHORT_FRONT, True,
        )
        float_sched = generate_schedule(
            reference_date, mat, float_frequency,
            calendar, convention, StubType.SHORT_FRONT, True,
        )
        pv_fixed = 0.0
        for i in range(1, len(fixed_sched)):
            yf = year_fraction(fixed_sched[i-1], fixed_sched[i], fixed_day_count)
            pv_fixed += par_rate * yf * curve.df(fixed_sched[i])
        pv_float = 0.0
        for i in range(1, len(float_sched)):
            d1, d2 = float_sched[i - 1], float_sched[i]
            df1 = curve.df(d1)
            df2 = curve.df(d2)
            yf = year_fraction(d1, d2, float_day_count)
            fwd = (df1 - df2) / (yf * df2)
            pv_float += fwd * yf * df2
        quotes.append(f"swap_{mat.isoformat()}")
        residuals.append(pv_fixed - pv_float)  # ~0 by construction

    if fras:
        for start, end, fra_rate in fras:
            tau = year_fraction(start, end, deposit_day_count)
            if tau > 0:
                df_s, df_e = curve.df(start), curve.df(end)
                model_fwd = (df_s - df_e) / (tau * df_e)
                quotes.append(f"fra_{start.isoformat()}_{end.isoformat()}")
                residuals.append(model_fwd - fra_rate)

    if futures:
        for start_date, end_date, fut_rate in futures:
            tau = year_fraction(start_date, end_date, deposit_day_count)
            if tau > 0:
                conv_adj = 0.0
                if hw_convexity_a > 0 and hw_convexity_sigma > 0:
                    t_start = year_fraction(reference_date, start_date, deposit_day_count)
                    t_end = year_fraction(reference_date, end_date, deposit_day_count)
                    def _B(s, t):
                        return (1 - _math.exp(-hw_convexity_a * (t - s))) / hw_convexity_a
                    # Must match the bootstrap's own convexity formula (Fix
                    # T3.17, line 150) — earlier this site held the
                    # pre-T3.17 formula, so residuals reported wrong values
                    # whenever σ>0 (W3).
                    conv_adj = 0.5 * hw_convexity_sigma**2 * _B(0, t_start) * _B(t_start, t_end)
                expected_fwd = fut_rate - conv_adj
                if turn_of_year_spread > 0 and start_date.year != end_date.year:
                    expected_fwd += turn_of_year_spread
                df_s, df_e = curve.df(start_date), curve.df(end_date)
                model_fwd = (df_s - df_e) / (tau * df_e)
                quotes.append(f"future_{start_date.isoformat()}_{end_date.isoformat()}")
                residuals.append(model_fwd - expected_fwd)

    # Parameters: pillar discount factors keyed by pillar date.
    parameters = {
        f"df({d.isoformat()})": float(df)
        for d, df in zip(curve.pillar_dates, [curve.df(d) for d in curve.pillar_dates])
    }

    return CalibrationResult.new(
        model_class="discount_curve_bootstrap",
        parameters=parameters,
        residuals=residuals,
        objective=ObjectiveKind.SSE,           # per-pillar exact fit; residuals ~0
        optimiser=OptimiserSpec(
            algorithm="brentq-sequential",
            tolerance=1e-12,                   # brentq xtol default
            max_iterations=len(parameters),
            extra={
                "interpolation": str(interpolation.value),
                "deposit_day_count": str(deposit_day_count.value),
                "hw_convexity_a": float(hw_convexity_a),
                "hw_convexity_sigma": float(hw_convexity_sigma),
                "turn_of_year_spread": float(turn_of_year_spread),
            },
        ),
        iterations=len(parameters),
        converged=True,                         # bootstrap is exact-fit by construction
        quotes_fitted=quotes,
        diagnostics=CalibrationDiagnostics(
            extra={
                "n_deposits": len(deposits),
                "n_swaps": len(swaps),
                "n_fras": len(fras) if fras else 0,
                "n_futures": len(futures) if futures else 0,
            },
        ),
        market_snapshot_id=market_snapshot_id,
    )


def _verify_round_trip(
    curve, reference_date, deposits, swaps, fras, futures,
    deposit_day_count, fixed_day_count, float_day_count,
    fixed_frequency, float_frequency, calendar, convention,
    hw_convexity_a, hw_convexity_sigma, turn_of_year_spread,
    tol=1e-6,
):
    """Verify the bootstrapped curve reprices all input instruments.

    Raises RuntimeWarning if any instrument reprices with error > tol.
    """
    import warnings

    errors = []

    # Check deposits
    for mat, rate in deposits:
        tau = year_fraction(reference_date, mat, deposit_day_count)
        if tau > 0:
            model_rate = (1.0 / curve.df(mat) - 1.0) / tau
            err = abs(model_rate - rate)
            if err > tol:
                errors.append(f"Deposit {mat}: input={rate:.6f}, model={model_rate:.6f}, err={err:.2e}")

    # Check swaps: reprice using the same fixed/float PV logic as the bootstrap
    for mat, par_rate in swaps:
        fixed_sched = generate_schedule(
            reference_date, mat, fixed_frequency,
            calendar, convention, StubType.SHORT_FRONT, True,
        )
        float_sched = generate_schedule(
            reference_date, mat, float_frequency,
            calendar, convention, StubType.SHORT_FRONT, True,
        )
        pv_fixed = 0.0
        for i in range(1, len(fixed_sched)):
            yf = year_fraction(fixed_sched[i-1], fixed_sched[i], fixed_day_count)
            pv_fixed += par_rate * yf * curve.df(fixed_sched[i])
        pv_float = 0.0
        for i in range(1, len(float_sched)):
            d1, d2 = float_sched[i - 1], float_sched[i]
            df1 = curve.df(d1)
            df2 = curve.df(d2)
            yf = year_fraction(d1, d2, float_day_count)
            fwd = (df1 - df2) / (yf * df2)
            pv_float += fwd * yf * df2
        err = abs(pv_fixed - pv_float)
        if err > tol:
            errors.append(f"Swap {mat}: PV_fixed-PV_float={err:.2e} (tol={tol:.2e})")

    # Check FRAs
    if fras:
        for start, end, fra_rate in fras:
            tau = year_fraction(start, end, deposit_day_count)
            if tau > 0:
                df_s, df_e = curve.df(start), curve.df(end)
                model_fwd = (df_s - df_e) / (tau * df_e)
                err = abs(model_fwd - fra_rate)
                if err > tol:
                    errors.append(f"FRA {start}-{end}: input={fra_rate:.6f}, model={model_fwd:.6f}, err={err:.2e}")

    # Check futures (convexity-adjusted forward rates)
    if futures:
        import math as _math
        for start_date, end_date, fut_rate in futures:
            tau = year_fraction(start_date, end_date, deposit_day_count)
            if tau > 0:
                # Apply same convexity adjustment as in bootstrap
                conv_adj = 0.0
                if hw_convexity_a > 0 and hw_convexity_sigma > 0:
                    t_start = year_fraction(reference_date, start_date, deposit_day_count)
                    t_end = year_fraction(reference_date, end_date, deposit_day_count)
                    def _B(s, t):
                        return (1 - _math.exp(-hw_convexity_a * (t - s))) / hw_convexity_a
                    # Mirror the bootstrap's own convexity formula (line 150,
                    # Fix T3.17); the pre-W3 verifier held the pre-T3.17
                    # formula so it reported wrong errors whenever σ>0.
                    conv_adj = 0.5 * hw_convexity_sigma**2 * _B(0, t_start) * _B(t_start, t_end)
                expected_fwd = fut_rate - conv_adj
                if turn_of_year_spread > 0 and start_date.year != end_date.year:
                    expected_fwd += turn_of_year_spread
                df_s, df_e = curve.df(start_date), curve.df(end_date)
                model_fwd = (df_s - df_e) / (tau * df_e)
                err = abs(model_fwd - expected_fwd)
                if err > tol:
                    errors.append(f"Future {start_date}-{end_date}: expected_fwd={expected_fwd:.6f}, model={model_fwd:.6f}, err={err:.2e}")

    if errors:
        msg = "Bootstrap round-trip failures:\n" + "\n".join(errors)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def bootstrap_forward_curve(
    reference_date: date,
    swaps: list[tuple[date, float]],
    discount_curve: DiscountCurve,
    deposits: list[tuple[date, float]] | None = None,
    fras: list[tuple[date, date, float]] | None = None,
    futures: list[tuple[date, date, float]] | None = None,
    deposit_day_count: DayCountConvention = DayCountConvention.ACT_360,
    fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360,
    float_day_count: DayCountConvention = DayCountConvention.ACT_360,
    fixed_frequency: Frequency = Frequency.SEMI_ANNUAL,
    float_frequency: Frequency = Frequency.QUARTERLY,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
    calendar: Calendar | None = None,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
    hw_convexity_a: float = 0.0,
    hw_convexity_sigma: float = 0.0,
    turn_of_year_spread: float = 0.0,
) -> DiscountCurve:
    """
    Bootstrap a forward/projection curve from IRS par rates, discounting off
    an externally provided discount curve (typically OIS).

    This is the second stage of dual-curve construction:
        1. Bootstrap OIS curve from OIS rates (discount curve)
        2. Bootstrap forward curve from IRS rates using OIS for discounting

    The forward curve is used to project floating rates. The discount curve
    is used to discount all cashflows.

    Args:
        reference_date: Curve reference date.
        swaps: List of (maturity_date, par_rate) for IRS, sorted by maturity.
        discount_curve: The OIS/risk-free curve for discounting.
        deposits: Optional short-end deposit rates for the forward curve.
        deposit_day_count: Day count for deposits.
        fixed_day_count: Day count for the fixed leg.
        float_day_count: Day count for the floating leg.
        fixed_frequency: Fixed leg payment frequency.
        float_frequency: Floating leg payment frequency.
        interpolation: Interpolation method.

    Returns:
        A DiscountCurve representing the forward/projection curve.
    """
    for i in range(1, len(swaps)):
        if swaps[i][0] <= swaps[i - 1][0]:
            raise ValueError("swaps must be sorted by maturity")

    pillar_dates: list[date] = []
    pillar_dfs: list[float] = []

    # Short end: deposits imply forward curve discount factors
    if deposits:
        for i in range(1, len(deposits)):
            if deposits[i][0] <= deposits[i - 1][0]:
                raise ValueError("deposits must be sorted by maturity")
        if swaps and deposits[-1][0] >= swaps[0][0]:
            raise ValueError("deposit maturities must come before swap maturities")
        for mat, rate in deposits:
            yf = year_fraction(reference_date, mat, deposit_day_count)
            pillar_dates.append(mat)
            pillar_dfs.append(1.0 / (1.0 + rate * yf))

    # Long end: solve for forward curve df at each swap maturity
    for mat, par_rate in swaps:
        fixed_sched = generate_schedule(
            reference_date, mat, fixed_frequency,
            calendar, convention, StubType.SHORT_FRONT, True,
        )
        float_sched = generate_schedule(
            reference_date, mat, float_frequency,
            calendar, convention, StubType.SHORT_FRONT, True,
        )

        def objective(df_guess: float, _mat=mat, _par=par_rate,
                      _fsched=fixed_sched, _flsched=float_sched) -> float:
            # Build trial forward curve
            trial_dates = pillar_dates + [_mat]
            trial_dfs = pillar_dfs + [df_guess]
            fwd_curve = DiscountCurve(
                reference_date, trial_dates, trial_dfs,
                day_count=DayCountConvention.ACT_365_FIXED,
                interpolation=interpolation,
            )

            # PV of fixed leg: discounted off the OIS curve
            pv_fixed = 0.0
            for i in range(1, len(_fsched)):
                yf = year_fraction(_fsched[i - 1], _fsched[i], fixed_day_count)
                pv_fixed += _par * yf * discount_curve.df(_fsched[i])

            # PV of floating leg: forwards from forward curve, discounted off OIS
            pv_float = 0.0
            for i in range(1, len(_flsched)):
                d1, d2 = _flsched[i - 1], _flsched[i]
                fdf1 = fwd_curve.df(d1)
                fdf2 = fwd_curve.df(d2)
                yf = year_fraction(d1, d2, float_day_count)
                fwd = (fdf1 - fdf2) / (yf * fdf2)
                pv_float += fwd * yf * discount_curve.df(d2)

            return pv_fixed - pv_float

        df_solved = brentq(objective, 1e-6, 3.0)  # wider bracket: handles negative rates to -3%
        pillar_dates.append(mat)
        pillar_dfs.append(df_solved)

    fwd_curve = DiscountCurve(
        reference_date, pillar_dates, pillar_dfs,
        day_count=DayCountConvention.ACT_365_FIXED,
        interpolation=interpolation,
    )

    # Round-trip verification: each swap should reprice at par
    _verify_forward_curve_round_trip(
        fwd_curve, discount_curve, reference_date, swaps, deposits,
        fixed_day_count, float_day_count, deposit_day_count,
        fixed_frequency, float_frequency, calendar, convention,
    )

    return fwd_curve


def _verify_forward_curve_round_trip(
    fwd_curve, discount_curve, reference_date, swaps, deposits,
    fixed_day_count, float_day_count, deposit_day_count,
    fixed_frequency, float_frequency, calendar, convention,
    tol=1e-6,
):
    """Verify forward curve reprices all input instruments."""
    import warnings
    errors = []

    # Check deposits
    if deposits:
        for mat, rate in deposits:
            tau = year_fraction(reference_date, mat, deposit_day_count)
            if tau > 0:
                model_rate = (1.0 / fwd_curve.df(mat) - 1.0) / tau
                err = abs(model_rate - rate)
                if err > tol:
                    errors.append(f"Fwd deposit {mat}: input={rate:.6f}, model={model_rate:.6f}, err={err:.2e}")

    # Check swaps: PV_fixed (on OIS) == PV_float (fwd rates on fwd curve, discounted on OIS)
    for mat, par_rate in swaps:
        fixed_sched = generate_schedule(
            reference_date, mat, fixed_frequency,
            calendar, convention, StubType.SHORT_FRONT, True,
        )
        float_sched = generate_schedule(
            reference_date, mat, float_frequency,
            calendar, convention, StubType.SHORT_FRONT, True,
        )
        pv_fixed = 0.0
        for i in range(1, len(fixed_sched)):
            yf = year_fraction(fixed_sched[i-1], fixed_sched[i], fixed_day_count)
            pv_fixed += par_rate * yf * discount_curve.df(fixed_sched[i])
        pv_float = 0.0
        for i in range(1, len(float_sched)):
            d1, d2 = float_sched[i - 1], float_sched[i]
            fdf1 = fwd_curve.df(d1)
            fdf2 = fwd_curve.df(d2)
            yf = year_fraction(d1, d2, float_day_count)
            fwd = (fdf1 / fdf2 - 1.0) / yf
            pv_float += fwd * yf * discount_curve.df(d2)
        err = abs(pv_fixed - pv_float)
        if err > tol:
            errors.append(f"Fwd swap {mat}: PV_fixed-PV_float={err:.2e} (tol={tol:.2e})")

    if errors:
        msg = "Forward curve round-trip failures:\n" + "\n".join(errors)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
