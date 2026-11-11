"""The equity underlying as market data (L1): spot, repo, dividends, and the dividend forward.

Quote-side equity data has the same status as a discount curve - it is what the market said, not a
model of it - so it lives at L1 beside the curves. The one computed thing here is the **dividend
forward**, which is market data composed from market data and involves no model parameter:

    F(T) = (S0 - sum_{t_i <= T} D_i*df(t_i)) * prod_{t_i <= T}(1 - d_i) * exp(-(q + b)T) / df(T)

**Three dividend conventions, because the market quotes all three** (Brockhaus ch 5.1-5.3), and they
are not interchangeable:

* **discrete cash** `D_i` - a fixed amount on a date. Subtracted at **present value**, because the
  forward is what a spot-plus-financing strategy delivers and the cash leaves the stock at `t_i`, not
  at `T`. Summing the raw amounts is the commonest slip and it overstates the deduction.
* **proportional** `d_i` - a fraction of spot on a date, so it scales the whole forward rather than
  shifting it, and multiple dividends **compound** `prod(1 - d_i)` rather than adding.
* **continuous yield** `q` - the smooth-limit idealisation, `exp(-qT)`. A fine strip of proportional
  dividends converges to it, which is the cross-check between two of the three forms.

**The repo/borrow rate `b` enters exactly like a yield.** Holding the stock to deliver the forward
earns the repo on the lent shares, so it reduces the forward the way a dividend does. It is a
separate field rather than folded into `q` because they are separately quoted and separately
stressed, and adding them at construction would lose which is which.

**`t_i <= T` is load-bearing and is a *date* boundary, not a value.** A dividend after expiry is
never received by a holder at `T`, so it cannot enter `F(T)` - and every value-based oracle would be
blind to dropping the restriction if a fixture happened to put all dividends before expiry. The same
shape as invariant 6's `pay > valuation_date`, where the analogous slip was a 41% mispricing.

Provenance:
  stock:  none - the archived predecessor's equity layer was never purified into this tree
  source: BROCKHAUS-2016 (docs/references.md) ch 5.1-5.3 - the dividend forward under discrete,
          proportional and continuous dividend conventions, read in `-raw` and `-layout`
  oracle: tests/L4/test_equity_vanilla.py - the forward against an independently composed reference;
          the continuous form against `S0*exp((r-q)T)`; discrete cash proved PV-discounted rather
          than raw-summed; a fine proportional strip converging to the continuous yield (a
          cross-form oracle, different formula each side); the post-expiry dividend proved unable to
          move `F(T)` with a companion proving it material at a later expiry; monotonicity in every
          dividend and in the repo
  crosscheck: quantlib BlackScholesMertonProcess / dividend-adjusted forward - deferred to the
          optional QuantLib tier
  slice:  tranche 30 - the equity underlying: the equity & hybrids arc opens
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from pricebook.market.curve import CurveHandle


@dataclass(frozen=True)
class EquityAsset:
    """What the market quotes on - the key a snapshot's equity data is held under."""

    ticker: str

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("an equity asset needs a ticker")


@dataclass(frozen=True)
class DividendSchedule:
    """The three quoted dividend conventions. Empty is a non-dividend-paying stock.

    `cash` and `proportional` are `(date, amount)` pairs; `continuous_yield` is the smooth
    idealisation. All three may be present at once - a name can have announced cash dividends near
    term, a proportional assumption further out, and a yield beyond the quoted horizon - which is why
    this is one object rather than three alternative types.
    """

    cash: tuple[tuple[date, float], ...] = ()
    proportional: tuple[tuple[date, float], ...] = ()
    continuous_yield: float = 0.0

    def __post_init__(self) -> None:
        if any(rate >= 1.0 for _, rate in self.proportional):
            raise ValueError("a proportional dividend of 100% or more would wipe out the forward")
        if not math.isfinite(self.continuous_yield):
            raise ValueError(f"the continuous yield must be finite, got {self.continuous_yield}")


@dataclass(frozen=True)
class EquityUnderlying:
    """Spot, the repo/borrow rate, and the dividend schedule - the quote side of one name."""

    spot: float
    repo_rate: float
    dividends: DividendSchedule = field(default_factory=DividendSchedule)

    def __post_init__(self) -> None:
        if not math.isfinite(self.spot) or self.spot <= 0.0:
            raise ValueError(f"the spot must be finite and positive, got {self.spot}")
        if not math.isfinite(self.repo_rate):
            raise ValueError(f"the repo rate must be finite, got {self.repo_rate}")


def dividend_forward(underlying: EquityUnderlying, expiry: date, discount: CurveHandle) -> float:
    """`F(T)` - the forward a spot-plus-financing strategy delivers at `expiry`.

    Dividends strictly after `expiry` are excluded: a holder at `T` never receives them. The three
    conventions compose in the order the finance dictates - cash is *subtracted* at present value
    (it leaves the stock, so it reduces the amount that can be financed), proportional dividends and
    the yield *scale* what is left, and the whole is grown to `T` by dividing by `df(T)`.
    """
    years = discount.time_measure.year_fraction(expiry)
    schedule = underlying.dividends
    present_value_of_cash = math.fsum(
        amount * discount.df(when) for when, amount in schedule.cash if when <= expiry
    )
    proportional_growth = math.prod(
        1.0 - rate for when, rate in schedule.proportional if when <= expiry
    )
    carry = math.exp(-(schedule.continuous_yield + underlying.repo_rate) * years)
    return (underlying.spot - present_value_of_cash) * proportional_growth * carry / discount.df(expiry)
