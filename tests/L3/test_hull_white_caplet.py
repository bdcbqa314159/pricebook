"""The Hull-White caplet, as a put on a zero-coupon bond. MCINERNEY-ZASTAWNIAK §3.5.

    BC(0;S,T,K) = B(0,T)*N(d+) - K*B(0,S)*N(d-)                              (3.16)
    BP(0;S,T,K) = K*B(0,S)*N(-d-) - B(0,T)*N(-d+)                            (3.17)
    d+ = [ln(B(0,T)/(B(0,S)*K)) + nu/2] / sqrt(nu),   d- = d+ - sqrt(nu)     (3.18)
    nu(0,S) = sigma^2/(2*a^3) * (1 - e^{-a(T-S)})^2 * (1 - e^{-2aS})         (3.20)
    Cpl_i(0) = (1 + tau_i*K) * BP(0; T_{i-1}, T_i, 1/(1 + tau_i*K))          (3.21)
    Flr_i(0) = (1 + tau_i*K) * BC(0; T_{i-1}, T_i, 1/(1 + tau_i*K))

**Why this source and not BERHAN-2012.** The thesis performs the §5.3.1 fit with a Hull-White cap
formula it declines to publish -- "the implementation of nominal volatility structure calibration is
a subject in itself and is beyond the scope of this thesis" -- so the forward model comes from
McInerney-Zastawniak §3.5 and only the *calibration target* comes from BERHAN §5.3.1.

**The source's `d-` drops its radical in BOTH renderings**, printing `d- = d+ - nu(0,S)` while
`nu` is defined as a variance. The §5.1 shape again, so transcription independence cannot settle it.
What settles it: the un-radicalled reading makes the **call price negative** (-0.0049 at
B(0,S)=0.97, B(0,T)=0.94, K=0.985, nu=0.0036), which no option can be. Noted here because
`test_bond_option_PARITY_...` below CANNOT settle it -- parity is an algebraic identity in `d+/d-`
whatever their relation, so on this question it is a check that cannot fail.

**The formulas are transcribed longhand here while the code COMPOSES `black_price`** -- a
Hull-White bond option is Black on the forward bond price `B(0,T)/B(0,S)` with total variance `nu`,
discounted by `B(0,S)`. Different code on each side, which is what makes this an independent check
rather than a restatement.
"""

import math

import pytest
from scipy.stats import norm

from pricebook.foundation import CapFloorKind
from pricebook.models.hull_white import (
    CapletPeriod,
    HullWhiteParams,
    bond_option_variance,
    hw_caplet,
    zero_bond_option,
)

A, SIGMA = 0.05, 0.010
PARAMS = HullWhiteParams(A, SIGMA)
FLAT_RATE = 0.03


def _df(t: float) -> float:
    return math.exp(-FLAT_RATE * t)


def _ref_variance(a, sigma, S, T):
    """(3.20), transcribed."""
    return sigma**2 / (2.0 * a**3) * (1.0 - math.exp(-a * (T - S))) ** 2 * (
        1.0 - math.exp(-2.0 * a * S)
    )


def _ref_bond_option(df_S, df_T, strike, nu, kind):
    """(3.16)/(3.17)/(3.18), transcribed -- with the radical the source omits."""
    if nu <= 0.0:
        intrinsic = df_T - strike * df_S
        return max(intrinsic, 0.0) if kind is CapFloorKind.CAP else max(-intrinsic, 0.0)
    v = math.sqrt(nu)
    dp = (math.log(df_T / (df_S * strike)) + 0.5 * nu) / v
    dm = dp - v
    if kind is CapFloorKind.CAP:
        return df_T * norm.cdf(dp) - strike * df_S * norm.cdf(dm)
    return strike * df_S * norm.cdf(-dm) - df_T * norm.cdf(-dp)


# ---- oracle 1: the independent transcription ----

@pytest.mark.parametrize("a", [0.01, 0.05, 0.30])
@pytest.mark.parametrize("sigma", [0.002, 0.010, 0.040])
@pytest.mark.parametrize(("S", "T"), [(1.0, 2.0), (2.0, 3.0), (0.5, 5.0), (7.0, 8.0)])
def test_the_variance_matches_eq_3_20_across_a_sweep(a, sigma, S, T):
    assert bond_option_variance(HullWhiteParams(a, sigma), S, T) == pytest.approx(
        _ref_variance(a, sigma, S, T), rel=1e-13
    )


@pytest.mark.parametrize("kind", list(CapFloorKind))
@pytest.mark.parametrize(("S", "T"), [(1.0, 2.0), (2.0, 3.0), (0.5, 5.0)])
@pytest.mark.parametrize("strike", [0.93, 0.97, 1.00])
def test_the_bond_option_matches_eqs_3_16_to_3_18(kind, S, T, strike):
    nu = _ref_variance(A, SIGMA, S, T)
    assert zero_bond_option(_df(S), _df(T), strike, nu, kind) == pytest.approx(
        _ref_bond_option(_df(S), _df(T), strike, nu, kind), rel=1e-13
    )


@pytest.mark.parametrize("kind", list(CapFloorKind))
@pytest.mark.parametrize("strike_rate", [0.01, 0.03, 0.06])
def test_the_caplet_matches_eq_3_21(kind, strike_rate):
    """A caplet is `(1+tau*K)` PUTS on the bond; a floorlet is `(1+tau*K)` CALLS. The direction is
    the thing an implementation silently reverses, so it is transcribed here, not inferred."""
    S, T, tau = 2.0, 3.0, 1.0
    nu = _ref_variance(A, SIGMA, S, T)
    scale = 1.0 + tau * strike_rate
    bond_kind = CapFloorKind.FLOOR if kind is CapFloorKind.CAP else CapFloorKind.CAP
    expected = scale * _ref_bond_option(_df(S), _df(T), 1.0 / scale, nu, bond_kind)
    period = CapletPeriod(df_start=_df(S), df_pay=_df(T), accrual=tau)
    assert hw_caplet(period, strike_rate, nu, kind) == pytest.approx(expected, rel=1e-13)


# ---- oracle 2: the anchor reduction, and its mandatory companion ----

@pytest.mark.parametrize("kind", list(CapFloorKind))
def test_at_ZERO_vol_the_caplet_IS_its_intrinsic(kind):
    """`sigma_n = 0` kills the variance exactly (eq 3.20 has it as an overall factor), so the
    option is its intrinsic on the deterministic forward rate. An exact reduction, not a limit."""
    S, T, tau, K = 2.0, 3.0, 1.0, 0.03
    period = CapletPeriod(df_start=_df(S), df_pay=_df(T), accrual=tau)
    nu = bond_option_variance(HullWhiteParams(A, 0.0), S, T)
    assert nu == 0.0
    forward = (_df(S) / _df(T) - 1.0) / tau
    omega = kind.omega
    expected = _df(T) * tau * max(omega * (forward - K), 0.0)
    assert hw_caplet(period, K, nu, kind) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("kind", list(CapFloorKind))
def test_sigma_MOVES_the_caplet_so_the_reduction_is_not_vacuous(kind):
    """The companion the charter requires: an implementation returning the intrinsic always would
    pass the reduction above."""
    S, T, tau, K = 2.0, 3.0, 1.0, 0.03
    period = CapletPeriod(df_start=_df(S), df_pay=_df(T), accrual=tau)
    prices = [
        hw_caplet(period, K, bond_option_variance(HullWhiteParams(A, s), S, T), kind)
        for s in (0.0, 0.004, 0.010, 0.025)
    ]
    assert prices == sorted(prices), prices
    assert len(set(round(p, 14) for p in prices)) == 4, prices
    assert prices[-1] > prices[0] * 1.05


# ---- oracle 3: parity. A guard -- and NOT a discriminator for `d-`, which is why that is said. ----

@pytest.mark.parametrize(("S", "T"), [(1.0, 2.0), (2.0, 5.0)])
def test_bond_option_PARITY_holds_but_CANNOT_settle_the_radical(S, T):
    """`BC - BP = B(0,T) - K*B(0,S)`, from the payoff. It guards a swapped `B(0,S)`/`B(0,T)` or a
    sign slip -- but it is satisfied **identically** for any `d+`/`d-`, because
    `N(d) + N(-d) = 1` collapses both terms. So it is kept as a guard and explicitly NOT relied on
    for the source's dropped radical; the positivity check below is what excludes that."""
    nu = _ref_variance(A, SIGMA, S, T)
    strike = 0.95
    call = zero_bond_option(_df(S), _df(T), strike, nu, CapFloorKind.CAP)
    put = zero_bond_option(_df(S), _df(T), strike, nu, CapFloorKind.FLOOR)
    assert call - put == pytest.approx(_df(T) - strike * _df(S), abs=1e-15)


def test_the_UNRADICALLED_reading_of_d_minus_gives_a_NEGATIVE_call():
    """The discriminator parity is not. The source prints `d- = d+ - nu`; read literally it prices
    a call below zero, which no option can be — so the radical is implied, and this test pins the
    reasoning rather than leaving it in a comment."""
    df_S, df_T, strike, nu = 0.97, 0.94, 0.985, 0.0036
    dp = (math.log(df_T / (df_S * strike)) + 0.5 * nu) / math.sqrt(nu)
    literal = df_T * norm.cdf(dp) - strike * df_S * norm.cdf(dp - nu)   # no radical
    assert literal < 0.0, literal
    assert zero_bond_option(df_S, df_T, strike, nu, CapFloorKind.CAP) > 0.0


# ---- oracle 4: requirements that survive a misprint the transcription might share ----

@pytest.mark.parametrize("kind", list(CapFloorKind))
def test_the_caplet_is_never_negative(kind):
    S, T, tau = 2.0, 3.0, 1.0
    period = CapletPeriod(df_start=_df(S), df_pay=_df(T), accrual=tau)
    for s in (0.0, 0.001, 0.01, 0.05, 0.20):
        nu = bond_option_variance(HullWhiteParams(A, s), S, T)
        for K in (-0.01, 0.0, 0.02, 0.05, 0.5):
            assert hw_caplet(period, K, nu, kind) >= 0.0, (s, K)


def test_a_DEEP_out_of_the_money_caplet_tends_to_zero():
    S, T, tau = 2.0, 3.0, 1.0
    period = CapletPeriod(df_start=_df(S), df_pay=_df(T), accrual=tau)
    nu = bond_option_variance(PARAMS, S, T)
    assert hw_caplet(period, 10.0, nu, CapFloorKind.CAP) < 1e-12


def test_the_variance_vanishes_when_the_BOND_MATURES_AT_EXPIRY():
    """`T = S` makes `D(S,T) = 0`, so there is nothing to be uncertain about — the bond is worth 1
    at its own maturity. A requirement, and it catches a swapped `(T-S)`."""
    assert bond_option_variance(PARAMS, 2.0, 2.0) == 0.0


def test_the_variance_RISES_then_SATURATES_in_option_expiry():
    """The `(1 - e^{-2aS})` factor: the caplet's variance grows with expiry and flattens. A wrong
    sign on `a` would make it fall, which no magnitude check notices."""
    nus = [bond_option_variance(PARAMS, S, S + 1.0) for S in (0.5, 1.0, 2.0, 5.0, 20.0, 60.0)]
    assert nus == sorted(nus), nus
    assert nus[-1] - nus[-2] < nus[1] - nus[0], f"it never saturates: {nus}"


def test_the_variance_GROWS_with_the_bond_tenor():
    nus = [bond_option_variance(PARAMS, 2.0, 2.0 + d) for d in (0.25, 0.5, 1.0, 5.0)]
    assert nus == sorted(nus), nus
    assert len(set(round(n, 18) for n in nus)) == 4


def test_a_NEGATIVE_mean_reversion_is_refused_at_construction():
    with pytest.raises(ValueError, match="mean reversion"):
        HullWhiteParams(-0.01, 0.01)
