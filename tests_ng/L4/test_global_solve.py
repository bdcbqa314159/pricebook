"""L4 oracle — the simultaneous (global) orchestration + its Jacobian (doc 22 Q4, doc 18 §3/§8).

`calibrate` now forks on `spec.solve.method`: the existing sequential bootstrap, or a global
N-D solve of the whole `CurveSet` at once. Both compose the SAME `CalibrationInstrument`
residuals. Oracles: sequential == simultaneous (degenerate), reprice-to-par through the L4
engine under the globally-solved curves, the Jacobian validated vs finite-difference, and
non-convergence returned as a value.
"""

from dataclasses import replace
from datetime import date

from pricebook_ng.calibration.calibrate import (
    CalibrationFailure,
    CalibrationMethod,
    CalibrationSpec,
    CurveBuild,
    ParSwapQuote,
    SolveConfig,
    calibrate,
    curve_set_residuals,
    product,
)
from pricebook_ng.engine import price
from pricebook_ng.foundation import (
    Currency,
    DayCountConvention,
    Frequency,
    PricingResult,
    Tenor,
    TenorUnit,
    get_rate_index,
    next_imm,
)
from pricebook_ng.market.curve import DiscountCurve
from pricebook_ng.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook_ng.market.quotes import DepositQuote, FRAQuote, FutureQuote

VAL = date(2026, 1, 15)
ESTR = get_rate_index("ESTR")
EURIBOR_3M = get_rate_index("EURIBOR_3M")
DC = DayCountConvention.ACT_365_FIXED
Y, M = TenorUnit.YEAR, TenorUnit.MONTH
IMM1 = next_imm(VAL + Tenor(1, Y))
IMM2 = next_imm(IMM1 + Tenor(1, M))

DISCOUNT = CurveBuild(
    index=ESTR, frequency=Frequency.ANNUAL, day_count=DC,
    quotes=(
        DepositQuote(Tenor(3, M), 0.0290), DepositQuote(Tenor(6, M), 0.0295),
        ParSwapQuote(Tenor(1, Y), 0.0300), ParSwapQuote(Tenor(2, Y), 0.0320),
        ParSwapQuote(Tenor(3, Y), 0.0340), ParSwapQuote(Tenor(5, Y), 0.0360),
    ),
)
PROJECTION = CurveBuild(
    index=EURIBOR_3M, frequency=Frequency.ANNUAL, day_count=DC,
    quotes=(
        FRAQuote(None, Tenor(3, M), 0.0312), FRAQuote(Tenor(3, M), Tenor(6, M), 0.0316),
        FRAQuote(Tenor(6, M), Tenor(9, M), 0.0320), FRAQuote(Tenor(9, M), Tenor(12, M), 0.0324),
        FutureQuote(IMM1, 0.9660), FutureQuote(IMM2, 0.9655),
        ParSwapQuote(Tenor(2, Y), 0.0332), ParSwapQuote(Tenor(3, Y), 0.0352),
        ParSwapQuote(Tenor(5, Y), 0.0372),
    ),
)
SEQUENTIAL = CalibrationSpec.single_currency(valuation_date=VAL, currency=Currency.EUR, discount=DISCOUNT, projection=PROJECTION)
GLOBAL = replace(SEQUENTIAL, solve=SolveConfig(method=CalibrationMethod.SIMULTANEOUS))


def _ok(spec: CalibrationSpec):
    result = calibrate(spec)
    assert not isinstance(result, CalibrationFailure), result
    return result


def test_sequential_equals_simultaneous_degenerate() -> None:
    seq_model, _ = _ok(SEQUENTIAL)
    glob_model, glob_result = _ok(GLOBAL)
    assert glob_result.converged
    for key in (
        (seq_model.market.curves.discount(Currency.EUR), glob_model.market.curves.discount(Currency.EUR)),
        (seq_model.market.curves.projection(EURIBOR_3M), glob_model.market.curves.projection(EURIBOR_3M)),
    ):
        seq_c, glob_c = key
        assert max(abs(a - b) for a, b in zip(seq_c.dfs, glob_c.dfs)) < 1e-10


def test_globally_solved_curves_reprice_every_instrument_to_zero_through_engine() -> None:
    model, _ = _ok(GLOBAL)
    for build in (DISCOUNT, PROJECTION):
        for quote in build.quotes:
            result_pv = price(product(GLOBAL, build, quote), model)
            assert isinstance(result_pv, PricingResult), f"{quote!r} -> {result_pv!r}"
            assert abs(result_pv.pv.amount) < 1e-9


def test_jacobian_predicts_residual_change_vs_finite_difference() -> None:
    model, result = _ok(GLOBAL)
    jac = result.jacobian
    assert jac is not None
    curves = model.market.curves
    disc = curves.discount(Currency.EUR)
    proj = curves.projection(EURIBOR_3M)
    n = len(jac.matrix)
    assert len(jac.matrix[0]) == n == (len(disc.dfs) - 1) + (len(proj.dfs) - 1)

    base = curve_set_residuals(GLOBAL, curves)
    h = 1e-6
    disc2 = DiscountCurve(disc.time_measure, disc.times, tuple([disc.dfs[0]] + [d + h for d in disc.dfs[1:]]), disc.interpolation)
    proj2 = DiscountCurve(proj.time_measure, proj.times, tuple([proj.dfs[0]] + [d + h for d in proj.dfs[1:]]), proj.interpolation)
    curves2 = CurveSet({
        CurveKey(CurveRole.DISCOUNT, Currency.EUR): disc2,
        CurveKey(CurveRole.PROJECTION, ESTR): disc2,
        CurveKey(CurveRole.PROJECTION, EURIBOR_3M): proj2,
    })
    bumped = curve_set_residuals(GLOBAL, curves2)
    for i in range(n):
        predicted = sum(jac.matrix[i][j] * h for j in range(n))  # jac · dx, dx = [h]*n
        actual = bumped[i] - base[i]
        assert abs(actual - predicted) < 1e-6, f"row {i}: actual={actual}, predicted={predicted}"


def test_non_convergence_is_a_value() -> None:
    starved = replace(GLOBAL, solve=SolveConfig(method=CalibrationMethod.SIMULTANEOUS, max_iterations=1))
    assert isinstance(calibrate(starved), CalibrationFailure)


def test_global_reprice_is_byte_identical() -> None:
    model, _ = _ok(GLOBAL)
    swap = product(GLOBAL, PROJECTION, PROJECTION.quotes[-1])
    assert price(swap, model) == price(swap, model)
