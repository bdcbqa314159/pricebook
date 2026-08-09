"""L3 audit-response oracles (slice 6d) — findings #1 (negative-rate calibration) and #7
(convergence gate from configured tolerance).

Each test EXPOSES an audit finding: it fails on v0.92.0 and passes after the fix.
"""

from datetime import date

from pricebook_ng.calibration.calibrate import (
    CalibrationFailure,
    CalibrationSpec,
    CurveBuild,
    ParSwapQuote,
    calibrate,
    curve_set_residuals,
)
from pricebook_ng.foundation import (
    Currency,
    DayCountConvention,
    Frequency,
    Tenor,
    TenorUnit,
    get_rate_index,
)
from pricebook_ng.market.quotes import DepositQuote

VAL = date(2026, 1, 15)
ESTR = get_rate_index("ESTR")
EURIBOR_3M = get_rate_index("EURIBOR_3M")
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR
A = Frequency.ANNUAL


def _q(pairs):
    return tuple(ParSwapQuote(Tenor(t, Y), r) for t, r in pairs)


# finding #1 — a EUR/JPY-style negative-rate OIS + swap curve
NEG_DISC = CurveBuild(ESTR, A, DC, _q([(1, -0.005), (2, -0.004), (3, -0.003), (5, -0.001)]))
NEG_PROJ = CurveBuild(EURIBOR_3M, A, DC, _q([(1, -0.004), (2, -0.003), (3, -0.002), (5, 0.0)]))
NEG_SPEC = CalibrationSpec.single_currency(
    valuation_date=VAL, currency=Currency.EUR, discount=NEG_DISC, projection=NEG_PROJ
)


def test_negative_rate_curve_calibrates_and_reprices_to_par() -> None:
    out = calibrate(NEG_SPEC)  # SEQUENTIAL default — must not raise on DF > 1 pillars
    assert not isinstance(out, CalibrationFailure), out
    model, result = out
    assert result.converged
    residuals = curve_set_residuals(NEG_SPEC, model.market.curves)
    assert max(abs(r) for r in residuals) < 1e-9


def test_unsolvable_input_returns_failure_not_raise() -> None:
    # a deposit rate below -1/τ makes deposit_df non-positive → no positive-DF root exists
    spec = CalibrationSpec.single_currency(
        valuation_date=VAL,
        currency=Currency.EUR,
        discount=CurveBuild(ESTR, A, DC, (DepositQuote(Tenor(1, Y), -1.5),)),
        projection=CurveBuild(EURIBOR_3M, A, DC, (DepositQuote(Tenor(1, Y), -1.5),)),
    )
    out = calibrate(spec)  # invariant 4: failure is a VALUE, never an escaped exception
    assert isinstance(out, CalibrationFailure)
