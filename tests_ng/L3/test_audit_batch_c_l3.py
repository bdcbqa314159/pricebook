"""L3 audit Batch C — #6: HAGAN_WEST + SEQUENTIAL is rejected (not a non-converged tuple). repro_H/N.

Hagan–West is non-local — a pillar-by-pillar bootstrap can't converge — so the unsupported combination
returns a `CalibrationFailure` naming SIMULTANEOUS, rather than a `(model, result)` with `converged=False`.
"""

from datetime import date

from pricebook_ng.calibration.calibrate import (
    CalibrationFailure,
    CalibrationMethod,
    CalibrationSpec,
    CurveBuild,
    ParSwapQuote,
    SolveConfig,
    calibrate,
)
from pricebook_ng.foundation import (
    Currency,
    DayCountConvention,
    Frequency,
    Interpolation,
    Tenor,
    TenorUnit,
    get_rate_index,
)

VAL = date(2026, 1, 15)
ESTR = get_rate_index("ESTR")
EURIBOR_3M = get_rate_index("EURIBOR_3M")
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR
HW = Interpolation.HAGAN_WEST
_OIS = tuple(ParSwapQuote(Tenor(y, Y), r) for y, r in [(1, 0.030), (2, 0.032), (3, 0.034), (5, 0.036)])
_IBOR = tuple(ParSwapQuote(Tenor(y, Y), r) for y, r in [(1, 0.0312), (2, 0.0332), (3, 0.0352), (5, 0.0372)])


def _spec(method: CalibrationMethod, interp: Interpolation) -> CalibrationSpec:
    return CalibrationSpec.single_currency(
        valuation_date=VAL,
        currency=Currency.EUR,
        discount=CurveBuild(ESTR, Frequency.ANNUAL, DC, _OIS, interpolation=interp),
        projection=CurveBuild(EURIBOR_3M, Frequency.ANNUAL, DC, _IBOR, interpolation=interp),
        solve=SolveConfig(method=method),
    )


def test_hagan_west_sequential_is_rejected() -> None:  # repro_H — was converged=False tuple
    out = calibrate(_spec(CalibrationMethod.SEQUENTIAL, HW))
    assert isinstance(out, CalibrationFailure)
    assert "simultaneous" in out.reason.lower()


def test_hagan_west_simultaneous_still_converges() -> None:
    out = calibrate(_spec(CalibrationMethod.SIMULTANEOUS, HW))
    assert not isinstance(out, CalibrationFailure)
    assert out[1].converged


def test_log_linear_sequential_unchanged() -> None:  # byte-identical for the supported combo
    out = calibrate(_spec(CalibrationMethod.SEQUENTIAL, Interpolation.LOG_LINEAR))
    assert not isinstance(out, CalibrationFailure)
    assert out[1].converged
