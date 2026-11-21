"""The two smile models' parameter types (L3). MCINERNEY-ZASTAWNIAK §8.2, §8.4.

Frozen data and the refusals; the pricing is oracled in `tests/L4/test_smile_caplet.py`. The one
piece of mathematics here is eq (8.9), the at-the-money matching rule the chapter's own numerical
example uses, and it has a printed reference value: `sigma = 0.2`, `F = 0.05`, `alpha = 0.05` gives
`sigma_D = 0.1`.

**On the naming.** The source calls the displacement `alpha` (eq 8.8) and reserves `beta` for the
CEV exponent it is derived from. The field is `displacement` rather than either letter: `alpha` is
already this arc's most overloaded symbol (Vasicek's mean reversion, G2++'s first factor) and
`beta` is about to mean the exponent when tranche 72 brings CEV in. Naming it after what it IS is
the tranche-66 `MertonParams.drift` precedent.
"""

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from pricebook.foundation import Currency, DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models.smile import (
    BachelierModel,
    DisplacedDiffusionModel,
    atm_matched_displaced_vol,
)

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
MEASURE = TimeMeasure(VAL, DayCountConvention.ACT_365_FIXED)
CURVE = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 4.0), (0.026, 0.030, 0.034, 0.038))
SNAPSHOT = MarketSnapshot(
    valuation_date=VAL, curves=CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): CURVE})
)


def test_eq_8_9_reproduces_the_chapters_OWN_printed_matching_vol():
    # Example 8.1: "if alpha = 0.05, then sigma_D(t) = 0.1". Transcribed from p.152.
    assert atm_matched_displaced_vol(0.2, 0.05, 0.05) == pytest.approx(0.1, abs=5e-16)


def test_a_zero_displacement_leaves_the_vol_UNTOUCHED_and_that_is_eq_8_9_exactly():
    # `sigma*F/(F+0) = sigma`. The shift is what the rule corrects for, so no shift is no correction.
    assert atm_matched_displaced_vol(0.2, 0.05, 0.0) == 0.2


def test_the_matching_vol_FALLS_as_the_displacement_grows_because_the_shift_adds_variance():
    vols = [atm_matched_displaced_vol(0.2, 0.05, a) for a in (0.0, 0.01, 0.05, 0.2)]
    assert vols == sorted(vols, reverse=True), vols
    # and materially so, not a rounding-scale drift the ordering would pass on noise
    assert vols[-1] < 0.5 * vols[0]


def test_a_displacement_that_drives_the_shifted_forward_non_positive_is_REFUSED():
    with pytest.raises(ValueError, match="shifted forward"):
        atm_matched_displaced_vol(0.2, 0.05, -0.05)


@pytest.mark.parametrize("vol", (0.0, -0.01))
def test_a_non_positive_vol_is_REFUSED_by_both_models(vol):
    with pytest.raises(ValueError, match="volatility"):
        BachelierModel(SNAPSHOT, vol)
    with pytest.raises(ValueError, match="volatility"):
        DisplacedDiffusionModel(SNAPSHOT, vol, 0.01)


def test_the_models_are_FROZEN():
    # §3's no-mutation clause. `setattr` rather than an assignment so the refusal is checked at
    # run time without a type suppression, which §7 forbids outright.
    model = BachelierModel(SNAPSHOT, 0.0085)
    with pytest.raises(FrozenInstanceError):
        setattr(model, "normal_vol", 0.01)


def test_they_are_DISTINCT_types_and_neither_subclasses_the_other():
    # A normal forward and a shifted-lognormal forward are different ideas (§6): they share the
    # word "vol" and not its units -- Bachelier's is a RATE per root-year, the displaced-diffusion's
    # is dimensionless. A base class would invite one to be passed where the other is meant.
    assert not isinstance(BachelierModel(SNAPSHOT, 0.0085), DisplacedDiffusionModel)
    assert not issubclass(BachelierModel, DisplacedDiffusionModel)
    assert not issubclass(DisplacedDiffusionModel, BachelierModel)
