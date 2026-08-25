"""L3 audit Batch D — #13 (lower-bracket expansion), #17 (SolveConfig honoured on the sequential path).

#13: `_solve_pillar_df` expands the upper bracket for negative rates but not the lower — a high-rate
short end (ARS is in scope) needs a DF below the old `1e-9` floor. #17: the sequential Brent must honour
`SolveConfig.tolerance`/`max_iterations` (invariant 5), not the solver's own defaults.
"""

from datetime import date

from pricebook_ng.calibration.calibrate import (
    CalibrationFailure,
    CalibrationMethod,
    CalibrationSpec,
    CurveBuild,
    ParSwapQuote,
    SolveConfig,
    _solve_pillar_df,
    calibrate,
)
from pricebook_ng.foundation import (
    Currency,
    DayCountConvention,
    Frequency,
    Tenor,
    TenorUnit,
    get_rate_index,
)

VAL = date(2026, 1, 15)
ESTR = get_rate_index("ESTR")
EURIBOR_3M = get_rate_index("EURIBOR_3M")
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR


def _q(pairs):
    return tuple(ParSwapQuote(Tenor(t, Y), r) for t, r in pairs)


def _spec(rates, solve=None):
    disc = CurveBuild(ESTR, Frequency.ANNUAL, DC, _q(rates))
    proj = CurveBuild(EURIBOR_3M, Frequency.ANNUAL, DC, _q([(t, r + 0.001) for t, r in rates]))
    return CalibrationSpec.single_currency(
        valuation_date=VAL, currency=Currency.EUR, discount=disc, projection=proj, solve=solve or SolveConfig()
    )


def test_solve_pillar_df_brackets_a_root_below_the_old_floor() -> None:  # #13 (ARS-style hyperinflation)
    # a pillar DF at 1e-12 (below the old 1e-9 lower bound); the solver must expand `lo` DOWN to it
    root = 1e-12
    got = _solve_pillar_df(lambda df: df - root, SolveConfig())
    assert abs(got - root) < 1e-14


def test_solve_pillar_df_unbracketable_raises_for_the_caller() -> None:  # #13 — failure is a value upstream
    # a residual with no positive-DF root anywhere → raises (the caller turns it into CalibrationFailure)
    try:
        _solve_pillar_df(lambda df: df * df + 1.0, SolveConfig())  # always > 0, no sign change
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_sequential_tolerance_is_live() -> None:  # #17 — the knob was dead on the sequential path
    rates = [(1, 0.03), (2, 0.032), (5, 0.036)]
    loose = calibrate(_spec(rates, SolveConfig(method=CalibrationMethod.SEQUENTIAL, tolerance=1e-1)))
    tight = calibrate(_spec(rates, SolveConfig(method=CalibrationMethod.SEQUENTIAL, tolerance=1e-14)))
    assert not isinstance(loose, CalibrationFailure) and not isinstance(tight, CalibrationFailure)
    # a 1e-1 xtol brackets far more coarsely than 1e-14 → the solved pillars measurably differ
    loose_df = loose[0].market.curves.discount(Currency.EUR).dfs
    tight_df = tight[0].market.curves.discount(Currency.EUR).dfs
    assert any(abs(a - b) > 1e-9 for a, b in zip(loose_df, tight_df))
