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
CONFIG = MonteCarloConfig(paths=60_000, seed=20261105)


def _driftless_model() -> LiborMarketModel:
    return LiborMarketModel(tenor=TENOR, correlation=RHO, numeraire_index=TARGET + 1)


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
    out = mc_caplet_price(model, TARGET, strike, GRID, CONFIG)
    assert out.standard_error > 0.0
    assert abs(out.mean - closed) < 4.0 * out.standard_error, (
        f"MC {out.mean} vs Black {closed}, gap {abs(out.mean - closed) / out.standard_error:.2f} SE"
    )


def test_O1_the_band_is_not_vacuously_wide():
    """The companion the anchor needs: four SEs around a huge SE would accept anything."""
    model = _driftless_model()
    closed = black_price(TENOR.forwards[TARGET], 0.038, _total_variance(), CapFloorKind.CAP)
    out = mc_caplet_price(model, TARGET, 0.038, GRID, CONFIG)
    assert 4.0 * out.standard_error < 0.08 * closed, (out.standard_error, closed)


def test_O1_the_caplet_is_NOT_independent_of_the_strike():
    """A pricer returning a constant would sit inside the band at one strike by construction."""
    model = _driftless_model()
    prices = [mc_caplet_price(model, TARGET, k, GRID, CONFIG).mean for k in (0.02, 0.038, 0.06)]
    assert prices[0] > prices[1] > prices[2], prices


# ============================================================================================
# ORACLE 2 — the martingale property, measure-explicit. This is what pins the DRIFT.
# ============================================================================================

def test_O2_the_target_forward_is_a_MARTINGALE_under_its_own_measure():
    """`E[f_i(T_i)] = f_i(0)` with the numeraire at `i+1`. The drift transcription (eq 12.8) must
    satisfy this for ALL parameters, so it survives an error oracle 1 might not see at one strike."""
    model = _driftless_model()
    paths = simulate_terminal_forwards(model, GRID, CONFIG)
    terminals = [p[TARGET] for p in paths]
    n = len(terminals)
    mean = math.fsum(terminals) / n
    spread = math.sqrt(math.fsum((x - mean) ** 2 for x in terminals) / (n - 1))
    standard_error = spread / math.sqrt(n)
    assert abs(mean - TENOR.forwards[TARGET]) < 4.0 * standard_error, (
        mean, TENOR.forwards[TARGET], standard_error
    )


def test_O2_a_NON_target_forward_is_NOT_a_martingale_under_this_measure():
    """The companion: under the `P_3` measure only forward 2 is driftless. If every forward came out
    a martingale, the drift would be a no-op and oracle 2 would be vacuous."""
    model = _driftless_model()
    paths = simulate_terminal_forwards(model, GRID, CONFIG)
    for index in (0, 3):
        terminals = [p[index] for p in paths]
        n = len(terminals)
        mean = math.fsum(terminals) / n
        spread = math.sqrt(math.fsum((x - mean) ** 2 for x in terminals) / (n - 1))
        drift_in_se = abs(mean - TENOR.forwards[index]) / (spread / math.sqrt(n))
        assert drift_in_se > 4.0, (index, mean, TENOR.forwards[index], drift_in_se)


def test_O2_the_drift_DIRECTION_matches_eq_12_8s_sign():
    """Forward 0 sits below the numeraire (negative drift) and forward 3 above it (positive) — so
    the simulated means must straddle their initial values in opposite directions."""
    paths = simulate_terminal_forwards(_driftless_model(), GRID, CONFIG)
    mean_0 = math.fsum(p[0] for p in paths) / len(paths)
    mean_3 = math.fsum(p[3] for p in paths) / len(paths)
    assert mean_0 < TENOR.forwards[0], (mean_0, TENOR.forwards[0])
    assert mean_3 > TENOR.forwards[3], (mean_3, TENOR.forwards[3])


# ============================================================================================
# ORACLE 3 — the -1/2 rate, and the exactness that makes the anchor sharp.
# ============================================================================================

def test_O3_the_error_falls_at_the_MINUS_ONE_HALF_rate():
    model = _driftless_model()
    counts = (4_000, 16_000, 64_000)
    ses = [
        mc_caplet_price(model, TARGET, 0.038, GRID, MonteCarloConfig(paths=n, seed=909)).standard_error
        for n in counts
    ]
    xs = [math.log(n) for n in counts]
    ys = [math.log(s) for s in ses]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    assert slope == pytest.approx(-0.5, abs=0.05), (slope, ses)


def test_the_step_count_does_NOT_move_the_martingale_forward():
    """**The exactness claim, asserted not assumed.** A driftless forward's log-Euler step is exact,
    so the terminal distribution of forward 2 must not depend on the step count — one step and
    thirty-two must give the same mean to within MC error. A scheme with a discretisation bias in
    that forward would show a drift with the step count."""
    model = _driftless_model()
    means = []
    for steps in (1, 8, 32):
        paths = simulate_terminal_forwards(model, PathGrid(EXPIRY, steps), CONFIG)
        means.append(math.fsum(p[TARGET] for p in paths) / len(paths))
    assert max(means) - min(means) < 1e-4 * TENOR.forwards[TARGET], means


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
    paths = simulate_terminal_forwards(_driftless_model(), GRID, CONFIG)
    assert all(f > 0.0 for p in paths for f in p)


def test_O6_a_BAD_grid_is_refused():
    with pytest.raises(ValueError, match="positive"):
        PathGrid(horizon=0.0, steps=4)
    with pytest.raises(ValueError, match="at least one step"):
        PathGrid(horizon=1.0, steps=0)


def test_O6_a_BAD_forward_index_is_refused():
    with pytest.raises(IndexError):
        mc_caplet_price(_driftless_model(), 9, 0.038, GRID, CONFIG)
