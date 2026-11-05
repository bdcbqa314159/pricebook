"""The Bermudan swaption under the LMM by Monte Carlo (L4). JAECKEL-2002 §12.6-12.8.

Jäckel's algorithm: simulate the yield curve out to the last exercise date, train a parametric
exercise boundary backwards through the exercise dates by maximising the pathwise exercise-dependent
value (eq 12.41), then **re-simulate with fresh paths** and price the trained policy.

    V(0) = notional · B(0,T_N) · E[ I_{τ}  ]        with τ the first date the policy exercises

`I_i`, the residual intrinsic value in the numéraire, is `(A_i/B_N)(T_i) · ω·(S_i(T_i) − K)` — the
swap's value at `T_i` divided by the numéraire bond. Expressed as that ratio because it is what the
expectation is taken of under the `P_N`-forward measure the simulation lives in.

**Train on one path set, price on a fresh one — a correctness requirement, not an optimisation.**
Jäckel: *"The final value `U_0(p_0)` gives then an estimate of the value of the Bermudan swaption with
a slight upward bias. Therefore, we finally re-run the simulation with a new set of `N_Sampling` yield
curve evolutions using the established exercise strategy parametrisation."* The policy is fitted to
maximise the average payoff **on the training paths**, so scoring it there lets it profit from the
realisations it was shaped by. The oracle asserts the bias is real rather than trusting the warning.

**The price is a LOWER bound, and that is the honest framing.** Any achievable exercise strategy
under-prices a Bermudan, whose value is the optimum over strategies — so a trained policy is a
genuine lower bound, and `perfect_foresight_price` (exercise at the ex-post best date, which no real
holder can do) is a genuine upper bound. The two bracket the price, and the bracket's width is the
measure of how good the parametrisation is for this trade. That diagnostic is the job Jäckel's
non-recombining tree did for him; the tranche-18 spec rules the tree out of scope (it is his tool for
*discovering* the coordinates, and he states the ones he settles on), so the sandwich carries it.

Provenance:
  stock:  parked/ng/.../swaption_mc.py and the legacy options/bermudan_* — read for SHAPE and edge
          cases only via docs/stock/mc_base_inventory.md; never imported (§5)
  source: JAECKEL-2002 (docs/references.md) §12.6 (the exercise-domain geometry and the coordinate
          pair), §12.7 eqs 12.38/12.39 via models/exercise_boundary.py, §12.8 eqs 12.40/12.41 (the
          training algorithm, the backward induction and the fresh-path re-simulation), §13.2 eq 13.2
          via engine/mc_path_engine.py for the evolution; MCINERNEY-ZASTAWNIAK §5.9 eqs 5.31/5.32 via
          models/swap_rate_vol.py for the residual swap rate and annuity
  oracle: tests/L4/test_bermudan_swaption_mc.py — the exact reduction of a single-exercise-date
          Bermudan to the tranche-17 European (which carries the caplet->Black anchor beneath it);
          the early-exercise premium banded at the resolution the combined standard errors allow;
          the perfect-foresight sandwich with its width reported; training monotonicity; the
          in-sample bias measured; determinism; invariant 7; and the measure/date refusals
  crosscheck: quantlib BermudanSwaptionEngine / Gaussian1dSwaptionEngine — deferred to the optional
          QuantLib tier
  slice:  tranche 18 — the Bermudan swaption under the LMM
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.engine.mc_path_engine import ObservationSchedule, simulate_forward_paths
from pricebook.foundation.mc_estimator import estimate
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.exercise_boundary import (
    BoundaryParams,
    BoundaryState,
    TrainingSlice,
    exercise_now,
    roll_continuations,
    train_slice,
)
from pricebook.models.libor_market_model import LiborMarketModel
from pricebook.models.swap_rate_vol import tenor_discount_factors
from pricebook.products.bermudan_swaption import BermudanSwaption


@dataclass(frozen=True)
class BermudanMcConfig:
    """The two path sets the algorithm needs, and the stepping between exercise dates.

    **Two `MonteCarloConfig`s rather than one, because the train/price split is part of the method.**
    Naming them separately in the type is what makes it impossible to price on the training paths by
    accident; Jäckel suggests `N_Sampling ≈ 2 N_Training`, and the oracle that proves the split is
    doing work deliberately violates it by passing the same config twice."""

    training: MonteCarloConfig
    pricing: MonteCarloConfig
    steps_per_segment: int = 2

    def __post_init__(self) -> None:
        if self.steps_per_segment < 1:
            raise ValueError(
                f"a Bermudan simulation needs at least one step between exercise dates, got "
                f"{self.steps_per_segment}"
            )


@dataclass(frozen=True)
class ExerciseSlices:
    """Per-path, per-exercise-date intrinsics and boundary states — eq 12.40's stored evolutions.

    Public because it is a named step of the algorithm (*"for each evolution of the yield curve, the
    residual intrinsic value `I_jk` ... is pre-calculated and stored"*) with three consumers: the
    pricer, the perfect-foresight bound, and the oracle that pins the **coordinate choice** — that
    the second coordinate is the NEXT coterminal swap rate `s_{i+1}`, which is §12.6's whole finding
    and which a mutation substituting `s_i` otherwise survives."""

    intrinsics: tuple[tuple[float, ...], ...]
    states: tuple[tuple[BoundaryState, ...], ...]
    count: int


def price_bermudan_swaption(
    bermudan: BermudanSwaption, model: LiborMarketModel, config: BermudanMcConfig
) -> PricingResult | PricingFailure:
    """Train the exercise boundary on one path set, then price the policy on a fresh one."""
    checked = _refuse(bermudan, model)
    if checked is not None:
        return checked

    times = _exercise_times(bermudan, model)
    schedule = ObservationSchedule(times=times, steps_per_segment=config.steps_per_segment)

    trained = train_exercise_policy(bermudan, model, schedule, config.training)
    priced = exercise_slices(bermudan, model, schedule, config.pricing)

    payoffs = [
        _policy_payoff(trained, priced.states[j], priced.intrinsics[j], bermudan.kind.omega)
        for j in range(priced.count)
    ]
    return _result(estimate(payoffs), bermudan, model)


def perfect_foresight_price(
    bermudan: BermudanSwaption, model: LiborMarketModel, config: BermudanMcConfig
) -> PricingResult:
    """The UPPER bound: exercise at the ex-post best date on each path.

    A biased-high estimator and a genuine upper bound, because it uses information no holder has —
    `max_i I_i` per path rather than the first `i` a policy fires at. It is computed on the **pricing**
    paths so the bracket is a statement about the same sample, not about two different ones: comparing
    across path sets would let Monte Carlo noise invert the inequality."""
    times = _exercise_times(bermudan, model)
    schedule = ObservationSchedule(times=times, steps_per_segment=config.steps_per_segment)
    sliced = exercise_slices(bermudan, model, schedule, config.pricing)
    payoffs = [max(max(row), 0.0) for row in sliced.intrinsics]
    return _result(estimate(payoffs), bermudan, model)


def _refuse(
    bermudan: BermudanSwaption, model: LiborMarketModel
) -> PricingFailure | None:
    """§3.4 — failure is a value, and the measure check is the one that matters silently."""
    span = bermudan.span
    if span.end > len(model.tenor):
        return PricingFailure(
            f"the swap span [{span.first}, {span.end}) does not fit the tenor's "
            f"{len(model.tenor)} forwards"
        )
    if model.numeraire_index != span.end:
        return PricingFailure(
            f"the model's numeraire index is {model.numeraire_index}, but pricing this Bermudan "
            f"needs the span's end bond {span.end} as numeraire"
        )
    if bermudan.first_exercise == 0:
        return PricingFailure(
            "the first exercise opportunity falls at t = 0, so the Bermudan has no optionality "
            "left to value; a trade exercisable today is the shell's business"
        )
    return None


def _exercise_times(bermudan: BermudanSwaption, model: LiborMarketModel) -> tuple[float, ...]:
    """`T_i` for each exercise index — the cumulative accruals, which is what the tenor's dates are."""
    accruals = model.tenor.accruals
    return tuple(math.fsum(accruals[:i]) for i in bermudan.exercise_indices)


def _residual_swap(forwards, accruals, first: int, end: int) -> tuple[float, float]:
    """`(S, A/B_end)` for the swap spanning `first … end−1`, from the forwards seen at `T_first`.

    The same construction tranche 17's engine uses, evaluated at a mid-life observation instead of at
    expiry — `B(T_first,T_first) = 1` because the span's own start is the spot date for the residual
    swap, whichever exercise date that happens to be."""
    bonds = [1.0]
    for k in range(first, end):
        bonds.append(bonds[-1] / (1.0 + forwards[k] * accruals[k]))
    annuity = math.fsum(accruals[first + m] * bonds[m + 1] for m in range(end - first))
    swap_rate = (bonds[0] - bonds[-1]) / annuity
    return swap_rate, annuity / bonds[-1]


def exercise_slices(
    bermudan: BermudanSwaption,
    model: LiborMarketModel,
    schedule: ObservationSchedule,
    config: MonteCarloConfig,
) -> ExerciseSlices:
    """Simulate, then reduce each path to what eqs 12.38/12.41 read — eq 12.40's precalculation.

    *"Also, for each evolution of the yield curve, the residual intrinsic value `I_jk` in the chosen
    numéraire as seen at time `t_k` is pre-calculated and stored."* The boundary's two coordinates are
    stored with it: `f_i(t_i)`, and the **next** coterminal swap rate `s_{i+1}(t_i)` — the one starting
    at the reset of `f_{i+1}` and running to terminal maturity, which is §12.6's choice."""
    indices = bermudan.exercise_indices
    end = bermudan.span.end
    accruals = model.tenor.accruals
    omega = bermudan.kind.omega
    initial = {
        i: _residual_swap(model.tenor.forwards, accruals, i + 1, end)[0]
        for i in indices
        if i + 1 < end
    }

    paths = simulate_forward_paths(model, schedule, config)
    intrinsics: list[tuple[float, ...]] = []
    states: list[tuple[BoundaryState, ...]] = []
    for path in paths:
        row_intrinsic: list[float] = []
        row_state: list[BoundaryState] = []
        for slot, i in enumerate(indices):
            forwards = path[slot]
            swap_rate, annuity_ratio = _residual_swap(forwards, accruals, i, end)
            row_intrinsic.append(annuity_ratio * omega * (swap_rate - bermudan.strike))
            # At the FINAL exercise date there is no `s_{i+1}`: the swap beyond terminal maturity
            # does not exist. Eq 12.39 sets `p_i1 = 0` there, so the coordinate is never read — the
            # zero below is unreachable arithmetic, not a stand-in value.
            next_rate = (
                _residual_swap(forwards, accruals, i + 1, end)[0] if i + 1 < end else 0.0
            )
            row_state.append(
                BoundaryState(
                    forward=forwards[i],
                    swap_rate=next_rate,
                    initial_swap_rate=initial.get(i, 0.0),
                )
            )
        intrinsics.append(tuple(row_intrinsic))
        states.append(tuple(row_state))
    return ExerciseSlices(tuple(intrinsics), tuple(states), len(paths))


def train_exercise_policy(
    bermudan: BermudanSwaption,
    model: LiborMarketModel,
    schedule: ObservationSchedule,
    config: MonteCarloConfig,
) -> tuple[BoundaryParams, ...]:
    """§12.8's backward induction: one optimisation per exercise date except the last.

    Public because the policy is a result in its own right — Jäckel's algorithm *produces an exercise
    strategy*, and pricing it is a separate step — and because the oracle must pin the policy itself,
    not only the bounds it implies. A forward-running induction also yields a valid lower bound, so
    bounds alone cannot see the direction; the oracle checks that the earliest slice was trained
    against continuations the LATER slices had already contributed to.

    *"Prior to each optimisation, we assign a path-value `v_j` to each evolution path which represents
    the value of the Bermudan swaption on this path if no exercise occurs up until and including
    `t_i`. This path-value vector is initialised to be zero."* The loop counts **down**, and step 1's
    reassignment `v_j := I_j(i+1)` overwrites rather than maximises — which is correct precisely
    because the iteration is backwards: the last write wins, and going backwards the last write is the
    *earliest* date the policy exercises at."""
    sliced = exercise_slices(bermudan, model, schedule, config)
    slots = len(bermudan.exercise_indices)
    omega = bermudan.kind.omega

    # The final slice is PINNED by eq 12.39, not fitted: exercise iff the residual swap is in the
    # money, which is known exactly. *"Whence no optimisation is required for them."*
    params: list[BoundaryParams] = [BoundaryParams.final(bermudan.strike)] * slots
    values: tuple[float, ...] = (0.0,) * sliced.count

    for slot in range(slots - 2, -1, -1):
        nxt = slot + 1
        # Step 1: carry the values back through the NEXT date, whose policy is already fitted.
        values = roll_continuations(
            params[nxt], _slice_at(sliced, nxt, values), omega
        )
        # Step 2: fit this date's boundary against those continuations (eq 12.41).
        params[slot] = train_slice(
            _slice_at(sliced, slot, values), omega, bermudan.strike
        )
    return tuple(params)


def _slice_at(
    sliced: ExerciseSlices, slot: int, continuations: tuple[float, ...]
) -> TrainingSlice:
    """One exercise date's column of the stored evolutions, paired with the carried values."""
    return TrainingSlice(
        states=tuple(sliced.states[j][slot] for j in range(sliced.count)),
        intrinsics=tuple(sliced.intrinsics[j][slot] for j in range(sliced.count)),
        continuations=continuations,
    )


def _policy_payoff(
    params: tuple[BoundaryParams, ...],
    states: tuple[BoundaryState, ...],
    intrinsics: tuple[float, ...],
    omega: float,
) -> float:
    """The payoff on one path under the trained policy: exercise at the FIRST date it fires.

    Walked forwards, which is the mirror of the training's backward overwrite. A holder exercises
    only into a positive value, so a policy that fires out of the money is ignored rather than
    obeyed — Jäckel's step 1 carries the same `I > 0` condition."""
    for params_i, state, intrinsic in zip(params, states, intrinsics, strict=True):
        if exercise_now(params_i, state, omega) and intrinsic > 0.0:
            return intrinsic
    return 0.0


def _result(sampled, bermudan: BermudanSwaption, model: LiborMarketModel) -> PricingResult:
    """Project the internal estimate into a `PricingResult` — §3 invariant 7.

    `scale` is deterministic, so it multiplies the mean and the standard error alike."""
    scale = tenor_discount_factors(model.tenor)[bermudan.span.end] * bermudan.notional.amount
    return PricingResult(
        pv=Money(sampled.mean * scale, bermudan.notional.currency),
        standard_error=sampled.standard_error * scale,
    )
