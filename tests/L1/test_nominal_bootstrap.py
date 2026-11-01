"""The nominal curve bootstrapped from par swap quotes (L1). BERHAN-2012 §5.1.

    S(Ti)  = (1 - P(0,Ti)) / SUM_{j=1..i} tau_j P(0,Tj)                    (5.1.1, corrected)
    P(0,Ti) = (1 - S(Ti) * SUM_{j=1..i-1} tau_j P(0,Tj)) / (1 + S(Ti) tau_i)  (5.1.2, corrected)

**Both equations are MISPRINTED in the thesis, and the two misprints are different.** Read in
`-raw` and `-layout`, which agree with each other, so the dual-rendering rule does not catch these
-- the source itself is wrong:

  * 5.1.1 carries a **spurious leading minus**, which would make every par rate negative;
  * 5.1.2 sums **`j = 0 .. i-1`** where the inversion of 5.1.1 requires `j = 1 .. i-1`. With
    `P(0,T0) := 1` the printed form adds a spurious unit to the annuity.

What catches them is the **requirement** that the two be mutual inverses, which is oracle 1:
starting from a known flat 3% curve, `j = 1..i-1` recovers it to the last digit (0.97044553) and
the printed `j = 0..i-1` does not (0.94089107). So the formula implemented here is the standard
par-swap convention, which the corrected equations are.

No identifiability oracle, and that is a reason rather than an omission: a sequential bootstrap is
**exactly determined** -- pillar `i` is fixed by quote `i` given pillars `< i`, a triangular
system -- so there is no valley for a tranche-6-style multi-start discriminator to explore. Oracle
1's pointwise recovery is the whole proof.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.foundation import CouponSchedule, DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import (
    DiscountCurve,
    ParSwapQuote,
    nominal_curve_from_par_swaps,
    swap_annuity,
)

ANCHOR = date(2026, 1, 15)
DC = DayCountConvention.ACT_365_FIXED
MEASURE = TimeMeasure(ANCHOR, DC)
DATES = tuple(ANCHOR + timedelta(days=365 * k) for k in range(6))   # anchor + 5 annual payments


def _schedule(n_periods: int) -> CouponSchedule:
    return CouponSchedule(DATES[: n_periods + 1], DC)


def _known_curve(pillars: tuple[tuple[float, float], ...]) -> DiscountCurve:
    return DiscountCurve(MEASURE, tuple(t for t, _ in pillars), tuple(r for _, r in pillars))


# A SLOPED known curve. Flat would make oracle 2 powerless and would hide any dependence on the
# earlier pillars, which is exactly what the annuity term carries.
SLOPED = _known_curve(((0.0, 0.020), (1.0, 0.025), (2.0, 0.030), (3.0, 0.034), (4.0, 0.037),
                       (5.0, 0.039)))


def _par_rate_of(curve: DiscountCurve, n_periods: int) -> float:
    """Eq 5.1.1 (corrected), computed HERE from the curve -- never read back from the bootstrap.

    This is what makes oracle 1 non-telescoping: the test owns the forward map, the code owns the
    inverse, and they are written from the formula independently."""
    schedule = _schedule(n_periods)
    annuity = sum(psi * curve.df(pay) for _, pay, psi in schedule.periods())
    return (1.0 - curve.df(DATES[n_periods])) / annuity


def _quotes_from(curve: DiscountCurve, upto: int = 5) -> list[ParSwapQuote]:
    return [ParSwapQuote(_schedule(n), _par_rate_of(curve, n)) for n in range(1, upto + 1)]


# ============================================================================================
# ORACLE 1 -- the anchor: a known curve round-trips through its own par rates, POINTWISE.
# ============================================================================================

def test_the_bootstrap_recovers_a_known_curve_at_every_pillar():
    built = nominal_curve_from_par_swaps(MEASURE, _quotes_from(SLOPED))
    for n in range(1, 6):
        assert built.df(DATES[n]) == pytest.approx(SLOPED.df(DATES[n]), rel=1e-12), f"{n}y pillar"


def test_the_printed_5_1_2_SUMMATION_would_NOT_recover_the_curve():
    """The misprint, as a test. Re-derive the first pillar with the printed `j = 0..i-1` sum --
    which includes `P(0,T0) = 1` -- and show it misses the known curve by ~3%, while the
    implemented form hits it. Without this, "we transcribed it correctly" is an assertion."""
    quotes = _quotes_from(SLOPED, upto=1)
    s, psi = quotes[0].par_rate, next(iter(_schedule(1).periods()))[2]
    as_printed = (1.0 - s * 1.0) / (1.0 + s)          # the spurious P(0,T0)=1 in the annuity
    corrected = (1.0 - s * 0.0) / (1.0 + s * psi)     # no j=0 term; i=1 has an empty prior sum
    truth = SLOPED.df(DATES[1])
    assert corrected == pytest.approx(truth, rel=1e-12)
    assert abs(as_printed - truth) > 0.02, "the printed form is not actually different -- no power"


def test_the_printed_5_1_1_SIGN_would_make_every_par_rate_negative():
    """The other misprint. Every recovered par rate must be positive on an upward curve; the
    printed leading minus would flip them all."""
    rates = [_par_rate_of(SLOPED, n) for n in range(1, 6)]
    assert all(r > 0.0 for r in rates), rates
    assert all(-r < 0.0 for r in rates)


# ============================================================================================
# ORACLE 2 -- companion: a sloped input gives a NON-flat curve (oracle 1 is not vacuous).
# ============================================================================================

def test_a_sloped_par_curve_bootstraps_to_a_sloped_discount_curve():
    built = nominal_curve_from_par_swaps(MEASURE, _quotes_from(SLOPED))
    zeros = [built.zero_rate(float(n)) for n in range(1, 6)]
    assert len(set(round(z, 10) for z in zeros)) == len(zeros), f"the curve came out flat: {zeros}"
    assert zeros == sorted(zeros), f"an upward par curve must give upward zeros: {zeros}"


def test_the_LATER_quotes_actually_move_the_LATER_pillars():
    """A bootstrap that ignored every quote after the first would still pass a flat-input test.
    Perturb only the 5y quote and require that the 5y pillar moves and the 1y does not."""
    quotes = _quotes_from(SLOPED)
    base = nominal_curve_from_par_swaps(MEASURE, quotes)
    bumped = list(quotes)
    bumped[-1] = ParSwapQuote(bumped[-1].schedule, bumped[-1].par_rate + 0.0050)
    moved = nominal_curve_from_par_swaps(MEASURE, bumped)
    assert moved.df(DATES[5]) != base.df(DATES[5])
    assert moved.df(DATES[1]) == pytest.approx(base.df(DATES[1]), rel=1e-15)
    assert moved.df(DATES[5]) < base.df(DATES[5]), "a higher par rate must discount harder"


# ============================================================================================
# ORACLE 3 -- reduction: one flat par rate, closed form.
# ============================================================================================

def test_a_single_quote_gives_the_closed_form_discount_factor():
    """One annual swap at `S`: `P = 1/(1 + S*psi)`, directly from 5.1.2 with an empty prior sum."""
    psi = next(iter(_schedule(1).periods()))[2]
    built = nominal_curve_from_par_swaps(MEASURE, [ParSwapQuote(_schedule(1), 0.03)])
    assert built.df(DATES[1]) == pytest.approx(1.0 / (1.0 + 0.03 * psi), rel=1e-14)


# ============================================================================================
# ORACLE 4 -- the consistency guard. TELESCOPES, so it is a guard and not the anchor.
# ============================================================================================

def test_the_bootstrapped_curve_reprices_every_input_quote_to_par():
    """Satisfied by construction when the bootstrap and `swap_annuity` share one convention --
    even a wrong one -- so this is kept as a guard that the shared atom IS shared, and oracle 1
    carries the correctness."""
    quotes = _quotes_from(SLOPED)
    built = nominal_curve_from_par_swaps(MEASURE, quotes)
    for q in quotes:
        annuity = swap_annuity(built, q.schedule)
        floating = 1.0 - built.df(q.schedule.dates[-1])
        assert floating / annuity == pytest.approx(q.par_rate, rel=1e-12)


# ============================================================================================
# ORACLE 5 -- negative rates work; the unsolvable is refused, never crashed.
# ============================================================================================

def test_NEGATIVE_par_rates_bootstrap_to_discount_factors_ABOVE_one():
    """The stock's lesson, kept: a negative rate is ordinary market data in EUR, and must produce
    `df > 1` rather than hitting a bracket. Closed-form per pillar, so there is no bracket to hit
    at all — which is the strongest version of this guarantee."""
    negative = _known_curve(((0.0, -0.006), (1.0, -0.004), (2.0, -0.002), (3.0, 0.001),
                             (4.0, 0.004), (5.0, 0.007)))
    quotes = _quotes_from(negative)
    assert quotes[0].par_rate < 0.0, "the fixture is not actually negative -- no power"
    built = nominal_curve_from_par_swaps(MEASURE, quotes)
    assert built.df(DATES[1]) > 1.0
    for n in range(1, 6):
        assert built.df(DATES[n]) == pytest.approx(negative.df(DATES[n]), rel=1e-12)


def test_a_par_rate_that_implies_a_NON_POSITIVE_discount_factor_is_REFUSED():
    """An absurd quote (a 5000% par rate against a long annuity) drives the implied `df` to zero
    or below. That is unsolvable, not a number to return."""
    quotes = _quotes_from(SLOPED)[:2]
    quotes[1] = ParSwapQuote(quotes[1].schedule, 50.0)
    with pytest.raises(ValueError, match="non-positive discount factor"):
        nominal_curve_from_par_swaps(MEASURE, quotes)


# ============================================================================================
# ORACLE 6 -- ill-posed input is refused.
# ============================================================================================

def test_no_quotes_is_refused():
    with pytest.raises(ValueError, match="at least one par swap quote"):
        nominal_curve_from_par_swaps(MEASURE, [])


def test_quotes_out_of_MATURITY_ORDER_are_refused():
    quotes = _quotes_from(SLOPED)[:3]
    with pytest.raises(ValueError, match="ascending maturity"):
        nominal_curve_from_par_swaps(MEASURE, [quotes[2], quotes[0], quotes[1]])


def test_DUPLICATE_maturities_are_refused():
    q = _quotes_from(SLOPED)[:1][0]
    with pytest.raises(ValueError, match="ascending maturity"):
        nominal_curve_from_par_swaps(MEASURE, [q, q])


def test_a_quote_whose_schedule_does_not_EXTEND_the_previous_one_is_refused():
    """The thesis's "swaps roll on a yearly basis" is a *precondition*, not a coincidence: each
    quote's payment dates must extend the previous quote's, or an earlier pillar the annuity needs
    was never solved. Stated and checked rather than assumed to hold."""
    offset = CouponSchedule(
        (ANCHOR, ANCHOR + timedelta(days=200), ANCHOR + timedelta(days=730)), DC
    )
    quotes = [_quotes_from(SLOPED)[0], ParSwapQuote(offset, 0.03)]
    with pytest.raises(ValueError, match="extend"):
        nominal_curve_from_par_swaps(MEASURE, quotes)


def test_a_schedule_not_anchored_at_the_curve_anchor_is_refused():
    late = CouponSchedule(
        (ANCHOR + timedelta(days=30), ANCHOR + timedelta(days=395)), DC
    )
    with pytest.raises(ValueError, match="anchor"):
        nominal_curve_from_par_swaps(MEASURE, [ParSwapQuote(late, 0.03)])


# ============================================================================================
# The shared annuity atom (§3d) -- one definition, so a future IRS pricer composes THIS one.
# ============================================================================================

def test_the_annuity_is_the_accrual_weighted_sum_of_discount_factors():
    schedule = _schedule(3)
    expected = sum(psi * SLOPED.df(pay) for _, pay, psi in schedule.periods())
    assert swap_annuity(SLOPED, schedule) == pytest.approx(expected, rel=1e-15)


def test_the_annuity_READS_the_schedules_own_day_count():
    """`psi` comes from the schedule's convention, not a hardcoded 1.0 — the carried-field lesson
    from tranche 3, applied to the atom.

    **The fixture must have non-unit accruals or this test cannot fail**, and the first version of
    it did not: the annual `_schedule` has exactly-365-day periods, so on ACT/365F `psi` is exactly
    1.0 and the accrual-weighted sum IS the plain sum. It passed `swap_annuity == Σ psi·df` while
    asserting `!= Σ df` against the same number. A SEMI-ANNUAL schedule makes `psi ≈ 0.49` and the
    two genuinely differ — the same fixture-power trap as a flat curve, found here by the test
    failing rather than by review."""
    semi = CouponSchedule(
        (ANCHOR, ANCHOR + timedelta(days=180), ANCHOR + timedelta(days=360)), DC
    )
    accruals = [psi for _, _, psi in semi.periods()]
    assert all(abs(psi - 1.0) > 0.4 for psi in accruals), f"the fixture has unit accruals: {accruals}"
    plain = sum(SLOPED.df(pay) for _, pay, _ in semi.periods())
    weighted = sum(psi * SLOPED.df(pay) for _, pay, psi in semi.periods())
    assert swap_annuity(SLOPED, semi) == pytest.approx(weighted, rel=1e-15)
    assert abs(swap_annuity(SLOPED, semi) - plain) > 0.5


def test_the_annuity_of_a_one_period_schedule_is_psi_times_one_df():
    schedule = _schedule(1)
    _, pay, psi = next(iter(schedule.periods()))
    assert swap_annuity(SLOPED, schedule) == pytest.approx(psi * SLOPED.df(pay), rel=1e-15)
