"""Black by simulation — the Monte Carlo spine's first priced number. T15.

Simulates the forward as a driftless lognormal martingale under its own forward measure,

    F_T = F_0 * exp(-V^2/2 + V*z),   z ~ N(0,1)

and averages `max(omega*(F_T - K), 0)`. That is exactly the expectation `black_price` evaluates in
closed form, so the closed form is the oracle — which is the point of starting here.

**Why this and not an `engine/mc_european.py` on a European-option product, as the spec's §2 says.**
The library has no vanilla-option product and nothing pulls one: adding an L2 product plus an L4
engine to host a simulation would be building a product with no consumer (charter §6), and the
tranche's purpose is the MC machinery, not a new instrument. The simulated counterpart of
`black_price` has a real consumer — these oracles — and sits beside it at L3. Raised as an `Ask:`
rather than taken silently.

**The oracle shape is new to this library and ruled honest up front.** An MC price never equals the
closed form, so the anchor is `|MC - Black| < k*SE` with the standard error **measured** (sigma/sqrt(N)),
never a tolerance chosen to make a test pass. The seed is fixed, so a fixed-seed MC is a *pure
function* and this is a hard assertion rather than a flaky one. And oracle 2 is what makes "it's
Monte Carlo" testable rather than an excuse: the error must fall at the **-1/2 rate**, which a biased
estimator passing at one N cannot fake.
"""

import math

import pytest

from pricebook.foundation import CapFloorKind
from pricebook.models.black import black_price
from pricebook.models.black_mc import MonteCarloConfig, black_mc_price

F, K = 1.025, 1.020
VARIANCE = 0.04          # V^2; V = 20%


@pytest.mark.parametrize("kind", list(CapFloorKind))
def test_the_MC_price_is_within_FOUR_measured_standard_errors_of_black(kind):
    """**The anchor.** `k = 4` is stated and justified: a two-sided 4-sigma band is ~99.994%, so at a
    FIXED seed this is a deterministic assertion that would need a real bias to fail — not a
    tolerance tuned until green."""
    closed = black_price(F, K, VARIANCE, kind)
    out = black_mc_price(F, K, VARIANCE, kind, MonteCarloConfig(paths=200_000, seed=20260101))
    assert out.standard_error > 0.0
    assert abs(out.mean - closed) < 4.0 * out.standard_error, (
        f"MC {out.mean} vs Black {closed}, SE {out.standard_error}, "
        f"gap {abs(out.mean - closed) / out.standard_error:.2f} SE"
    )


def test_the_standard_error_is_NOT_so_wide_that_the_anchor_is_vacuous():
    """The companion the anchor needs: a 4-SE band around a huge SE would accept anything. At 200k
    paths the band must be small relative to the price itself."""
    closed = black_price(F, K, VARIANCE, CapFloorKind.CAP)
    out = black_mc_price(F, K, VARIANCE, CapFloorKind.CAP,
                         MonteCarloConfig(paths=200_000, seed=20260101))
    assert 4.0 * out.standard_error < 0.05 * closed, (out.standard_error, closed)


def test_the_error_falls_at_the_MINUS_ONE_HALF_rate():
    """**The oracle that makes "it's Monte Carlo" a testable claim.** A biased estimator can sit
    inside 4 SE at one `N` by luck; it cannot reproduce the `-1/2` slope of `log SE` against
    `log N`. Fitted by least squares over four decades-ish of path counts."""
    counts = (2_000, 8_000, 32_000, 128_000)
    ses = []
    for n in counts:
        out = black_mc_price(F, K, VARIANCE, CapFloorKind.CAP,
                             MonteCarloConfig(paths=n, seed=5150))
        ses.append(out.standard_error)

    xs = [math.log(n) for n in counts]
    ys = [math.log(s) for s in ses]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum(
        (x - mean_x) ** 2 for x in xs
    )
    assert slope == pytest.approx(-0.5, abs=0.03), (slope, ses)


def test_the_estimator_is_UNBIASED_across_seeds():
    """A construction bias — `exp` of a mis-centred normal is the classic — is invisible at one seed
    and shows as a systematic offset across many. The band is the standard error of the MEAN of the
    seed estimates, so it tightens with the seed count rather than being chosen."""
    closed = black_price(F, K, VARIANCE, CapFloorKind.CAP)
    seeds = range(1, 41)
    means = [
        black_mc_price(F, K, VARIANCE, CapFloorKind.CAP,
                       MonteCarloConfig(paths=20_000, seed=s)).mean
        for s in seeds
    ]
    n = len(means)
    grand = math.fsum(means) / n
    spread = math.sqrt(math.fsum((m - grand) ** 2 for m in means) / (n - 1))
    standard_error_of_mean = spread / math.sqrt(n)
    assert abs(grand - closed) < 4.0 * standard_error_of_mean, (
        f"grand mean {grand} vs Black {closed}, SE of mean {standard_error_of_mean}"
    )


def test_the_same_seed_gives_a_BYTE_IDENTICAL_price():
    """§3 determinism and §9's byte-identical check: a fixed-seed MC is a pure function."""
    config = MonteCarloConfig(paths=5_000, seed=424242)
    a = black_mc_price(F, K, VARIANCE, CapFloorKind.CAP, config)
    b = black_mc_price(F, K, VARIANCE, CapFloorKind.CAP, config)
    assert repr(a.mean) == repr(b.mean)
    assert repr(a.standard_error) == repr(b.standard_error)


def test_DIFFERENT_seeds_give_different_prices():
    a = black_mc_price(F, K, VARIANCE, CapFloorKind.CAP, MonteCarloConfig(paths=5_000, seed=1))
    b = black_mc_price(F, K, VARIANCE, CapFloorKind.CAP, MonteCarloConfig(paths=5_000, seed=2))
    assert a.mean != b.mean


def test_at_ZERO_variance_the_simulation_IS_the_intrinsic_exactly():
    """An exact reduction, not a statistical one: with no variance every path lands on the forward,
    so the payoff is deterministic and the standard error is exactly zero. The one case where MC and
    the closed form must agree to the bit."""
    for kind in CapFloorKind:
        out = black_mc_price(F, K, 0.0, kind, MonteCarloConfig(paths=1_000, seed=9))
        intrinsic = black_price(F, K, 0.0, kind)
        assert out.mean == pytest.approx(intrinsic, rel=1e-14)
        assert out.standard_error == pytest.approx(0.0, abs=1e-15)


def test_the_simulated_forward_is_a_MARTINGALE():
    """A requirement independent of the payoff: `E[F_T] = F_0`. It catches a missing `-V^2/2`
    convexity correction, which would bias every option price upward while still producing a
    plausible number."""
    out = black_mc_price(F, 0.0, VARIANCE, CapFloorKind.CAP,
                         MonteCarloConfig(paths=200_000, seed=77))
    # With a zero strike a call pays the forward itself, so the estimate IS E[F_T].
    assert abs(out.mean - F) < 4.0 * out.standard_error, (out.mean, F, out.standard_error)


def test_a_NON_POSITIVE_path_count_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        MonteCarloConfig(paths=0, seed=1)
    with pytest.raises(ValueError, match="at least two"):
        MonteCarloConfig(paths=1, seed=1)


def test_a_NEGATIVE_seed_is_refused():
    with pytest.raises(ValueError, match="seed"):
        MonteCarloConfig(paths=100, seed=-1)


def test_an_ODD_path_count_is_accepted_and_still_deterministic():
    """Box-Muller works in pairs, so an odd path count needs handling — the engine must not refuse a
    perfectly reasonable request, nor silently price a different number of paths than asked."""
    out = black_mc_price(F, K, VARIANCE, CapFloorKind.CAP, MonteCarloConfig(paths=1_001, seed=3))
    assert out.n_samples == 1_001
    again = black_mc_price(F, K, VARIANCE, CapFloorKind.CAP, MonteCarloConfig(paths=1_001, seed=3))
    assert repr(out.mean) == repr(again.mean)
