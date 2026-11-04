"""The Monte Carlo path engine: LMM forwards evolved and averaged (L4). JAECKEL-2002 eq 13.2.

Per step, in logarithmic coordinates (Jaeckel calls eq 12.34 *"the Euler scheme in logarithmic
coordinates"*, with the `sqrt(dt)` and `dt` scalings absorbed into `A` and `C`):

    f_i(t+dt) = f_i(t) * exp( mu_i(f(t)) * dt  -  0.5 * c_ii  +  SUM_j a_ij z_j )        (13.2)

with `c_ij = sigma_i sigma_j rho_ij dt` the step covariance and `A` its pseudo-square-root (12.7).
The drift is recomputed from the CURRENT state each step, because eq 12.8 is state-dependent.

**The anchor is a closed form of ours, and it is clean for a reason worth stating.** Choose the
numeraire so that forward `i` has exactly zero drift (`N = i+1`, eq 12.8's middle branch). Then its
log-Euler step is `exp(-c_ii/2 + sum_j a_ij z_j)` — an **exact** lognormal martingale whatever the
step count, since the per-step variances sum to `sigma_i^2 T_i` and the `-c/2` terms sum to match. So
there is **no discretisation bias in that forward at all**, and the only gap to `black_price` is Monte
Carlo error. That is what makes oracle 1 a sharp `|MC - Black| < k*SE` rather than a fuzzy one — and
`test_the_step_count_does_NOT_move_the_martingale_forward` asserts the exactness instead of assuming it.
"""

import math
from functools import lru_cache

import pytest

from pricebook.engine.mc_path_engine import PathGrid, mc_caplet_price, simulate_terminal_forwards
from pricebook.foundation import CapFloorKind
from pricebook.models.black import black_price
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.libor_market_model import LiborMarketModel, LmmTenor

TENOR = LmmTenor(
    accruals=(0.5, 0.5, 0.5, 0.5),
    forwards=(0.030, 0.034, 0.038, 0.042),
    volatilities=(0.22, 0.20, 0.18, 0.16),
)
RHO = 0.80
# Forward 2 spans [1.0, 1.5]; with the numeraire at 3 it is driftless, so it is the anchor's subject.
TARGET = 2
EXPIRY = 1.0            # T_2, when f_2 fixes
GRID = PathGrid(horizon=EXPIRY, steps=8)
# 20,000 paths, not 60,000: the `slow` gate's ceiling is 10s and CI measured ~4x slower than this
# machine in tranche 11, so the file needs real margin. The band companion below asserts the SE is
# still small enough for the anchor to mean something at this count.
CONFIG = MonteCarloConfig(paths=20_000, seed=20261105)


def _driftless_model() -> LiborMarketModel:
    return LiborMarketModel(tenor=TENOR, correlation=RHO, numeraire_index=TARGET + 1)


# **Simulations are CACHED, and the `slow` gate is why.** A simulation is a deterministic function
# of (numeraire, steps, paths, seed), so caching changes nothing about what is asserted — while
# several oracles ask about the same one. Vectorising took the file 8.4s -> 1.7s; caching takes it
# further, which matters because CI measured ~4x slower than this machine in tranche 11 and a time
# ceiling has to hold on the slowest runner.
@lru_cache(maxsize=None)
def _paths(numeraire: int, steps: int, paths: int, seed: int):
    model = LiborMarketModel(tenor=TENOR, correlation=RHO, numeraire_index=numeraire)
    return simulate_terminal_forwards(model, PathGrid(EXPIRY, steps), MonteCarloConfig(paths, seed))


def _caplet_from_paths(numeraire: int, strike: float, config: MonteCarloConfig):
    """The caplet estimate from cached paths — identical arithmetic to `mc_caplet_price`, which the
    anchor still calls directly at least once so the public entry point is itself exercised."""
    from pricebook.foundation.mc_estimator import estimate

    return estimate(
        [max(p[TARGET] - strike, 0.0) for p in _paths(numeraire, GRID.steps, config.paths, config.seed)]
    )


def _total_variance() -> float:
    return TENOR.volatilities[TARGET] ** 2 * EXPIRY


# ============================================================================================
# ORACLE 1 — THE ANCHOR: a caplet reprices to our Black within measured standard errors.
# ============================================================================================

@pytest.mark.parametrize("strike", [0.030, 0.038, 0.046])
def test_O1_a_caplet_reprices_to_our_BLACK_within_four_standard_errors(strike):
    """`k = 4` stated and justified (a two-sided 4-sigma band is ~99.994%), the seed fixed so this
    is a hard assertion, and the SE **measured** rather than chosen."""
    model = _driftless_model()
    closed = black_price(TENOR.forwards[TARGET], strike, _total_variance(), CapFloorKind.CAP)
    out = (
        mc_caplet_price(model, TARGET, strike, GRID, CONFIG)
        if strike == 0.038
        else _caplet_from_paths(TARGET + 1, strike, CONFIG)
    )
    assert out.standard_error > 0.0
    assert abs(out.mean - closed) < 4.0 * out.standard_error, (
        f"MC {out.mean} vs Black {closed}, gap {abs(out.mean - closed) / out.standard_error:.2f} SE"
    )


def test_O1_the_band_is_not_vacuously_wide():
    """The companion the anchor needs: four SEs around a huge SE would accept anything."""
    model = _driftless_model()
    closed = black_price(TENOR.forwards[TARGET], 0.038, _total_variance(), CapFloorKind.CAP)
    out = _caplet_from_paths(TARGET + 1, 0.038, CONFIG)
    assert 4.0 * out.standard_error < 0.08 * closed, (out.standard_error, closed)


def test_O1_the_caplet_is_NOT_independent_of_the_strike():
    """A pricer returning a constant would sit inside the band at one strike by construction."""
    model = _driftless_model()
    prices = [_caplet_from_paths(TARGET + 1, k, CONFIG).mean for k in (0.02, 0.038, 0.06)]
    assert prices[0] > prices[1] > prices[2], prices


# ============================================================================================
# ORACLE 2 — the martingale property, measure-explicit. This is what pins the DRIFT.
# ============================================================================================

def test_O2_the_target_forward_is_a_MARTINGALE_under_its_own_measure():
    """`E[f_i(T_i)] = f_i(0)` with the numeraire at `i+1`. The drift transcription (eq 12.8) must
    satisfy this for ALL parameters, so it survives an error oracle 1 might not see at one strike."""
    model = _driftless_model()
    terminals = [p[TARGET] for p in _paths(TARGET + 1, GRID.steps, CONFIG.paths, CONFIG.seed)]
    n = len(terminals)
    mean = math.fsum(terminals) / n
    spread = math.sqrt(math.fsum((x - mean) ** 2 for x in terminals) / (n - 1))
    standard_error = spread / math.sqrt(n)
    assert abs(mean - TENOR.forwards[TARGET]) < 4.0 * standard_error, (
        mean, TENOR.forwards[TARGET], standard_error
    )


def test_O2_the_drift_IS_present_measured_by_PAIRING_two_numeraires():
    """**The companion to the martingale anchor — and my first version of it could not work.**

    I asserted that a non-target forward's mean differs from its initial value by more than 4 SE.
    Measured, the LMM drift is **economically tiny**: `mu_0 = -1.2e-3` moves `f_0` by ~3.5e-5 against
    a standard error of ~4.7e-5, so the signal sits BELOW one SE and 4 SE would need ~170,000 paths
    for one forward and far more for the others. The test was unpassable, not the code wrong.

    The fix uses the structure instead of brute force: simulate the **same seed under two
    numeraires**. The random draws are then identical, so the difference in each forward's terminal
    mean is almost pure drift and the Monte Carlo noise cancels. That turns a 0.4-1.4 SE signal into
    **297-372 SE at 8,000 paths** — and it tests eq 12.8's *summation bounds* as well as its
    presence, because moving the numeraire changes how many terms each sum carries."""
    paired = MonteCarloConfig(paths=8_000, seed=4242)
    under_1 = _paths(1, GRID.steps, paired.paths, paired.seed)
    under_3 = _paths(3, GRID.steps, paired.paths, paired.seed)
    for index in range(len(TENOR)):
        differences = [b[index] - a[index] for a, b in zip(under_1, under_3)]
        n = len(differences)
        mean = math.fsum(differences) / n
        spread = math.sqrt(math.fsum((d - mean) ** 2 for d in differences) / (n - 1))
        in_se = abs(mean) / (spread / math.sqrt(n))
        assert in_se > 20.0, (index, mean, in_se)
        # Moving the numeraire from 1 to 3 reduces every forward's drift (eq 12.8: fewer positive
        # terms above the numeraire, more negative ones below), so every difference is negative.
        assert mean < 0.0, (index, mean)


def test_O2_the_drift_SIGN_matches_eq_12_8_either_side_of_the_numeraire():
    """Eq 12.8's sign flips: a forward **below** the numeraire drifts negatively, one **at or above**
    it positively. Measured by pairing each forward against the numeraire that makes *it* driftless
    (`N = i+1`), so the comparison isolates its own drift with the draws held identical.

    **Unpaired this is not measurable**, which my first version missed: `mu_3 = +5.3e-4` moves `f_3`
    by ~2.2e-5 against an SE of ~4.7e-5, so the sign sits below the noise. Pairing removes the noise
    rather than demanding more paths."""
    paired = MonteCarloConfig(paths=8_000, seed=4242)

    def mean_terminal(numeraire: int, index: int) -> float:
        paths = _paths(numeraire, GRID.steps, paired.paths, paired.seed)
        return math.fsum(p[index] for p in paths) / len(paths)

    # Forward 0 is driftless at N=1; at N=3 eq 12.8 gives it a NEGATIVE drift.
    assert mean_terminal(3, 0) < mean_terminal(1, 0)
    assert LiborMarketModel(TENOR, RHO, numeraire_index=3).drift(0) < 0.0

    # Forward 3 is driftless at N=4; at N=3 eq 12.8 gives it a POSITIVE drift.
    assert mean_terminal(3, 3) > mean_terminal(4, 3)
    assert LiborMarketModel(TENOR, RHO, numeraire_index=3).drift(3) > 0.0


# ============================================================================================
# ORACLE 3 — the -1/2 rate, and the exactness that makes the anchor sharp.
# ============================================================================================

def test_O3_the_error_falls_at_the_MINUS_ONE_HALF_rate():
    model = _driftless_model()
    counts = (3_000, 12_000, 48_000)
    ses = [
        mc_caplet_price(model, TARGET, 0.038, GRID, MonteCarloConfig(paths=n, seed=909)).standard_error
        for n in counts
    ]
    xs = [math.log(n) for n in counts]
    ys = [math.log(s) for s in ses]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    assert slope == pytest.approx(-0.5, abs=0.05), (slope, ses)


def test_the_step_count_does_NOT_BIAS_the_martingale_forward():
    """**The exactness claim, with the right yardstick.** A driftless forward's log-Euler step is
    exact, so its terminal distribution must not depend on the step count.

    My first version compared the means to `1e-4` relative — far tighter than Monte Carlo error, and
    wrong in principle: different step counts consume *different numbers of draws*, so the samples
    are independent and the means differ by MC noise whatever the scheme. Measured at 20,000 paths:
    0.03796434 / 0.03796264 / 0.03798866 for 1 / 8 / 32 steps, against an SE of 4.9e-5 — agreement
    to ~0.5 SE. The assertion is therefore in standard errors, which is what "no bias" can mean here.
    A scheme with a genuine discretisation bias would show a TREND with the step count, not scatter."""
    model = _driftless_model()
    config = MonteCarloConfig(paths=12_000, seed=515)
    summary = []
    for steps in (1, 8, 32):
        terminals = [p[TARGET] for p in _paths(TARGET + 1, steps, config.paths, config.seed)]
        n = len(terminals)
        mean = math.fsum(terminals) / n
        spread = math.sqrt(math.fsum((x - mean) ** 2 for x in terminals) / (n - 1))
        summary.append((mean, spread / math.sqrt(n)))

    reference = TENOR.forwards[TARGET]
    for mean, standard_error in summary:
        assert abs(mean - reference) < 4.0 * standard_error, (mean, reference, standard_error)
    # And no trend: the one-step and thirty-two-step means agree within their combined error.
    (m1, s1), _, (m32, s32) = summary
    assert abs(m1 - m32) < 4.0 * math.sqrt(s1 * s1 + s32 * s32), (m1, m32, s1, s32)


# ============================================================================================
# ORACLE 6 — determinism and failure-as-a-value.
# ============================================================================================

def test_O6_the_same_seed_gives_BYTE_IDENTICAL_paths():
    model = _driftless_model()
    a = simulate_terminal_forwards(model, GRID, MonteCarloConfig(paths=200, seed=31337))
    b = simulate_terminal_forwards(model, GRID, MonteCarloConfig(paths=200, seed=31337))
    assert [repr(x) for p in a for x in p] == [repr(x) for p in b for x in p]


def test_O6_different_seeds_give_different_paths():
    model = _driftless_model()
    a = simulate_terminal_forwards(model, GRID, MonteCarloConfig(paths=200, seed=1))
    b = simulate_terminal_forwards(model, GRID, MonteCarloConfig(paths=200, seed=2))
    assert a != b


def test_O6_the_forwards_stay_POSITIVE_which_lognormality_guarantees():
    """A log-Euler scheme cannot produce a negative rate — unlike the plain Euler scheme Jaeckel
    contrasts it with. Asserted because a sign error in the exponent would show here first."""
    paths = _paths(TARGET + 1, GRID.steps, CONFIG.paths, CONFIG.seed)
    assert all(f > 0.0 for p in paths for f in p)


def test_O6_a_BAD_grid_is_refused():
    with pytest.raises(ValueError, match="positive"):
        PathGrid(horizon=0.0, steps=4)
    with pytest.raises(ValueError, match="at least one step"):
        PathGrid(horizon=1.0, steps=0)


def test_O6_a_BAD_forward_index_is_refused():
    with pytest.raises(IndexError):
        mc_caplet_price(_driftless_model(), 9, 0.038, GRID, CONFIG)
