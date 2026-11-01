"""The floor -> floorlet strip and the implied-vol surface. BERHAN-2012 §5.2.

Under the thesis's **Simplified Jarrow-Yildirim** -- which §5.2 proves (Prop 2.3.1) is JY at
`sigma_n = sigma_r = 0` -- the convexity `C` vanishes and the log-ratio variance collapses to
`V^2 = sigma_I^2 * (T_i - T_{i-1})`. Both are verified by tranche 4/5 oracles, and the second is the
reason the **variance horizon is the PERIOD LENGTH, not the time to expiry**: the floorlet on
`[2y, 3y]` has expiry 3y and a one-year variance horizon.

So a node is inverted as:

    black_value_i = floorlet_i / (N * psi_i * Pn(0,T_i))
    m_i           = [Pn(0,T_{i-1})/Pn(0,T_i)] * [Pr(0,T_i)/Pr(0,T_{i-1})]      (C = 0)
    sigma_I(i)    = black_implied_vol(black_value_i, m_i, 1 + kappa, T_i - T_{i-1}, FLOOR)

**On oracle power -- the count is the floor, the per-node values are the discriminator.** A count
(N expiries -> N floorlets) proves none was *dropped*; it cannot catch a **mis-indexed or reversed**
difference, where N nodes still appear with the right count and the wrong values. The discriminator
is the per-node match, and it only has power when **consecutive floorlets differ** -- so the fixture
uses a SLOPED sigma_I across four periods, the distinctness is asserted
(`test_the_fixture_floorlets_are_DISTINCT...`), and a deliberately reversed strip is shown to produce
different numbers (`test_a_REVERSED_difference_would_NOT_pass`). On a flat surface every node holds
the same value and a reordering is invisible -- the fifth appearance of the fixture-power lesson,
applied before the build rather than after a failure.
"""

import math
from datetime import date, timedelta

import pytest
from scipy.stats import norm

from pricebook.foundation import (
    CapFloorKind,
    CouponSchedule,
    Currency,
    DayCountConvention,
    IndexInterpolation,
    InflationIndex,
    Money,
)
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import (
    CurveKey,
    CurveRole,
    CurveSet,
    DiscountCurve,
    FloorQuote,
    ImpliedVolSurface,
    MarketSnapshot,
)
from pricebook.models.implied_vol_surface import implied_vol_surface

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
MEASURE = TimeMeasure(VAL, DC)
HICP = InflationIndex("HICPxt", publication_lag=3, interpolation=IndexInterpolation.LINEAR)
N = 1_000_000.0
KAPPA = 0.015
DATES = tuple(VAL + timedelta(days=365 * k) for k in range(5))   # anchor + 4 annual periods

# SLOPED, so consecutive floorlets differ and a reordering is detectable. Flat would make the
# per-node discriminator powerless -- see the module docstring.
SIGMA_BY_PERIOD = (0.008, 0.012, 0.018, 0.025)

# SLOPED curves too, so the forward ratios are not all equal either.
NOMINAL = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 4.0), (0.026, 0.030, 0.034, 0.038))
REAL = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 4.0), (0.008, 0.011, 0.013, 0.016))
MARKET = MarketSnapshot(
    valuation_date=VAL,
    curves=CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): NOMINAL,
                     CurveKey(CurveRole.REAL, EUR): REAL}),
)


def _schedule(n_periods: int) -> CouponSchedule:
    return CouponSchedule(DATES[: n_periods + 1], DC)


def _reference_black(forward, strike, variance, kind):
    """Black-76 transcribed here, independent of `models/black.py`."""
    w = 1.0 if kind is CapFloorKind.CAP else -1.0
    if variance <= 0.0:
        return max(w * (forward - strike), 0.0)
    v = math.sqrt(variance)
    d1 = (math.log(forward / strike) + 0.5 * variance) / v
    d2 = d1 - v
    return w * (forward * norm.cdf(w * d1) - strike * norm.cdf(w * d2))


def _reference_floorlet(i: int, sigma: float, kappa: float = KAPPA) -> float:
    """Floorlet `i` (1-based) priced HERE from eq 2.6.5 with `C = 0`, so the test owns the forward
    map and oracle 1 does not telescope."""
    start, pay, psi = list(_schedule(i).periods())[i - 1]
    tau = MEASURE.year_fraction(pay) - MEASURE.year_fraction(start)
    m = (NOMINAL.df(start) / NOMINAL.df(pay)) * (REAL.df(pay) / REAL.df(start))
    value = _reference_black(m, 1.0 + kappa, sigma * sigma * tau, CapFloorKind.FLOOR)
    return N * psi * NOMINAL.df(pay) * value


def _quotes(sigmas=SIGMA_BY_PERIOD, kappa: float = KAPPA) -> list[FloorQuote]:
    """Nested floor quotes: the floor to `T_n` is the sum of floorlets 1..n."""
    out = []
    for n in range(1, len(sigmas) + 1):
        total = sum(_reference_floorlet(i, sigmas[i - 1], kappa) for i in range(1, n + 1))
        out.append(FloorQuote(_schedule(n), kappa, Money(total, EUR), Money(N, EUR)))
    return out


# ============================================================================================
# ORACLE 1 -- the anchor. The decomposition, per-node, on a discriminating fixture.
# ============================================================================================

def test_the_fixture_floorlets_are_DISTINCT_so_the_per_node_check_has_power():
    """The premise oracle 1 rests on, asserted rather than assumed: if consecutive floorlets were
    equal, a mis-indexed or reversed difference would be invisible."""
    floorlets = [_reference_floorlet(i, SIGMA_BY_PERIOD[i - 1]) for i in range(1, 5)]
    assert len(set(round(f, 6) for f in floorlets)) == 4, floorlets
    assert all(f > 0.0 for f in floorlets)
    gaps = [abs(b - a) for a, b in zip(floorlets, floorlets[1:])]
    assert all(g > 1.0 for g in gaps), f"consecutive floorlets are nearly equal: {gaps}"


def test_the_surface_recovers_the_GENERATING_vol_at_every_node():
    surface = implied_vol_surface(MARKET, _quotes())
    for i, sigma in enumerate(SIGMA_BY_PERIOD, start=1):
        assert surface.vol(DATES[i], KAPPA) == pytest.approx(sigma, rel=1e-9), f"node {i}"


def test_the_node_COUNT_equals_the_expiry_count_population_floor():
    """The floor, not the discriminator: it proves no node was dropped."""
    surface = implied_vol_surface(MARKET, _quotes())
    assert len(surface.nodes) == len(SIGMA_BY_PERIOD)


def test_a_REVERSED_difference_would_NOT_pass():
    """Proves the per-node check actually discriminates a mis-indexed strip. Reverse the generating
    vols, rebuild the quotes, and require the recovered surface to differ at a middle node -- which
    is where a reversal shows and a count never would."""
    forward_surface = implied_vol_surface(MARKET, _quotes(SIGMA_BY_PERIOD))
    reversed_surface = implied_vol_surface(MARKET, _quotes(tuple(reversed(SIGMA_BY_PERIOD))))
    middle = DATES[2]
    assert forward_surface.vol(middle, KAPPA) != pytest.approx(
        reversed_surface.vol(middle, KAPPA), rel=1e-6
    ), "a reversed strip produces the same middle node -- the fixture cannot detect a reorder"


def test_the_FIRST_floor_quote_IS_the_first_floorlet():
    """`floor(T_0) = 0`, so the difference at `i = 1` is the quote itself. Asserted because an
    off-by-one in the differencing direction lands precisely here."""
    quotes = _quotes()
    surface = implied_vol_surface(MARKET, quotes[:1])
    assert surface.vol(DATES[1], KAPPA) == pytest.approx(SIGMA_BY_PERIOD[0], rel=1e-9)


# ============================================================================================
# ORACLE 4 -- reduction, and its mandatory companion.
# ============================================================================================

def test_a_FLAT_vol_input_recovers_a_FLAT_surface():
    flat = (0.014,) * 4
    surface = implied_vol_surface(MARKET, _quotes(flat))
    recovered = [surface.vol(DATES[i], KAPPA) for i in range(1, 5)]
    assert all(v == pytest.approx(0.014, rel=1e-9) for v in recovered), recovered


def test_a_SLOPED_vol_input_recovers_the_SLOPE_not_a_constant():
    """The companion. Without it, the flat reduction above is satisfied by an inverter that returns
    a single number regardless of input."""
    surface = implied_vol_surface(MARKET, _quotes())
    recovered = [surface.vol(DATES[i], KAPPA) for i in range(1, 5)]
    assert recovered == sorted(recovered), recovered
    assert len(set(round(v, 10) for v in recovered)) == 4, f"the surface came out flat: {recovered}"
    assert recovered[-1] / recovered[0] > 2.5, "the recovered slope is far shallower than the input"


# ============================================================================================
# Several strikes -> a genuine surface, not a curve.
# ============================================================================================

def test_two_STRIKES_build_two_independent_columns():
    low, high = 0.010, 0.025
    quotes = _quotes(SIGMA_BY_PERIOD, low) + _quotes(tuple(s * 1.3 for s in SIGMA_BY_PERIOD), high)
    surface = implied_vol_surface(MARKET, quotes)
    assert len(surface.nodes) == 8
    for i, sigma in enumerate(SIGMA_BY_PERIOD, start=1):
        assert surface.vol(DATES[i], low) == pytest.approx(sigma, rel=1e-9)
        assert surface.vol(DATES[i], high) == pytest.approx(sigma * 1.3, rel=1e-9)


def test_strikes_are_stripped_INDEPENDENTLY_not_pooled():
    """A builder that differenced across strike instead of across maturity would still produce the
    right node count. Bump only the high-strike quotes and require the low column not to move."""
    low, high = 0.010, 0.025
    base = _quotes(SIGMA_BY_PERIOD, low)
    surface_a = implied_vol_surface(MARKET, base + _quotes(SIGMA_BY_PERIOD, high))
    surface_b = implied_vol_surface(
        MARKET, base + _quotes(tuple(s * 1.5 for s in SIGMA_BY_PERIOD), high)
    )
    for i in range(1, 5):
        assert surface_a.vol(DATES[i], low) == pytest.approx(surface_b.vol(DATES[i], low), rel=1e-15)
        assert surface_a.vol(DATES[i], high) != pytest.approx(surface_b.vol(DATES[i], high), rel=1e-6)


# ============================================================================================
# ORACLE 5 -- refusal, never a guessed vol.
# ============================================================================================

def test_a_DECREASING_floor_strip_is_REFUSED():
    """A floor to `T_2` worth less than the floor to `T_1` implies a negative floorlet, which no
    volatility produces. Refuse rather than hand a negative price to the inverter."""
    quotes = _quotes()
    broken = [quotes[0], FloorQuote(quotes[1].schedule, KAPPA,
                                    Money(quotes[0].price.amount * 0.5, EUR), Money(N, EUR))]
    with pytest.raises(ValueError, match="negative floorlet"):
        implied_vol_surface(MARKET, broken)


def test_quotes_NOT_in_ascending_maturity_are_REFUSED():
    quotes = _quotes()
    with pytest.raises(ValueError, match="ascending maturity"):
        implied_vol_surface(MARKET, [quotes[2], quotes[0], quotes[1]])


def test_a_schedule_that_does_not_EXTEND_the_previous_one_is_REFUSED():
    """Nesting is a precondition of differencing, as it was for the tranche 8 bootstrap."""
    offset = CouponSchedule((VAL, VAL + timedelta(days=200), VAL + timedelta(days=730)), DC)
    quotes = _quotes()
    with pytest.raises(ValueError, match="extend"):
        implied_vol_surface(
            MARKET, [quotes[0], FloorQuote(offset, KAPPA, quotes[1].price, Money(N, EUR))]
        )


def test_no_quotes_is_REFUSED():
    with pytest.raises(ValueError, match="at least one"):
        implied_vol_surface(MARKET, [])


def test_a_missing_REAL_curve_is_REFUSED():
    nominal_only = MarketSnapshot(
        valuation_date=VAL, curves=CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): NOMINAL})
    )
    with pytest.raises(KeyError):
        implied_vol_surface(nominal_only, _quotes())


def test_mixed_CURRENCIES_between_price_and_notional_are_REFUSED():
    quotes = _quotes()
    with pytest.raises(ValueError, match="currency"):
        implied_vol_surface(
            MARKET,
            [FloorQuote(quotes[0].schedule, KAPPA, quotes[0].price, Money(N, Currency("USD")))],
        )


# ============================================================================================
# The surface value type (L1).
# ============================================================================================

def test_the_surface_refuses_a_node_it_does_not_hold():
    surface = implied_vol_surface(MARKET, _quotes())
    with pytest.raises(KeyError, match="no implied volatility"):
        surface.vol(DATES[1], 0.99)


def test_the_surface_does_NOT_interpolate_between_nodes():
    """Deliberately not built (§3.4): interpolation arrives with a consumer that needs an off-node
    vol. Asserted so the absence is a decision rather than an oversight."""
    surface = implied_vol_surface(MARKET, _quotes())
    midway = DATES[1] + timedelta(days=180)
    with pytest.raises(KeyError):
        surface.vol(midway, KAPPA)


def test_the_surface_is_frozen():
    surface = implied_vol_surface(MARKET, _quotes())
    assert isinstance(surface, ImpliedVolSurface)
    assert surface.anchor == VAL
