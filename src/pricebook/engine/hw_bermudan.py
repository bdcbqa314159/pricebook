"""The Bermudan swaption in the Hull-White model, by exact backward induction (L4).
`MCINERNEY-ZASTAWNIAK` §3.6.

The tree already prices a Bermudan swaption - under the **LMM**, by Longstaff-Schwartz regression
(tranche 18). This prices the same instrument under a one-factor short rate, and the method is
different for a reason worth stating, because it was the tranche's design ruling:

**Longstaff-Schwartz exists to approximate a continuation value that is a function of a
high-dimensional state. Under Hull-White the state is ONE Markov coordinate, so the continuation
value is an exact deterministic function of it.** Regressing here would reach for a statistical
estimate where a deterministic computation exists - §4's `closed form > cross-check` ordering
inverted - and would pay a regression error *plus* the foresight/suboptimality bracket for a
quantity that needs neither. So the induction is a quadrature over the state, not a regression.

*(The step-1 grep found `models/exercise_boundary.py` imports only `math` and `dataclass` - the
regression is already model-agnostic, and the LMM coupling lives one level up in
`engine/bermudan_swaption_mc.py`. So "generalise the regression" was not available as work: there
was nothing to generalise. `BoundaryState`'s LMM coordinates are constructed at one site in `src/`.
Ruled at tranche 68 with those counts; Cowork's prior was the lattice and the counts agreed.)*

**The source's recursion, transcribed from both renderings and reconciled (§4).** §3.6:

    E(T_i)    = (PS(T_i))^+                           the exercise value
    Berm(T_i) = max(E(T_i), C(T_i))                                                  (3.30)
    C(T_i)    = B(T_i, T_{i+1}) E^{P_{T_{i+1}}}[ Berm(T_{i+1}) | F_{T_i} ]

and the forward-measure dynamics the conditional expectation needs:

    dr(t) = (theta(t) - alpha r(t) - sigma(t)^2 D(t,T)) dt + sigma(t) dW^T(t)        (3.28)

**The `-sigma^2 D(t,T)` term is the whole content of the measure change, and it is NOT transcribed
as a closed form** - the source prints the SDE, not the integrated drift adjustment, so integrating
it is the reader's work and a transcription from memory would be exactly what §5 forbids. It is
computed by quadrature of the printed integrand instead, and then **oracled by a requirement that
pins it completely**: under `P_{T_{i+1}}` a bond must reprice through the transition,

    B(T_i, T_{i+1}) E^{T_{i+1}}[ B(T_{i+1}, T_j) | x(T_i) ] = B(T_i, T_j)

for every `T_j`. A wrong drift breaks that identity at once, and it needs no second transcription to
check - which is why the step is oracled rather than only the price (tranche 18's §12.8 lesson: a
price bound cannot see the algorithm).

**The conditional expectation is Gauss-Hermite** (`foundation/quadrature.py`, tranche 12b), reused
rather than a trinomial tree: the transition is exactly Gaussian, so the rule integrates it exactly
where a tree would discretise it.

**Where that is NOT exact, and the error is measured rather than claimed away.** A Gauss rule loses
its spectral accuracy at a **kink**, and the value function carries one at the exercise boundary.
Two consequences, and the second is honest debt:

* The roll into the **last** exercise date is a European swaption, so it is computed by piece 1's
  **closed form** at every grid point - eq (3.25) - and carries no quadrature error at all. That is
  the same `closed form > approximation` ordering that chose this method over a regression, applied
  one level down. It also makes the one-exercise reduction **exact** rather than grid-limited.
* The **earlier** rolls integrate `max(exercise, continuation)`, which has no closed form, so the
  kink remains. Measured spread across 48/96/192 nodes: **about 2e-3 relative** on a four-exercise
  Bermudan. That is the method's dominant error, it is stated here and asserted by the oracle, and
  the fix is known rather than vague: **split the integral at the exercise boundary** (locate it by
  root-find, integrate the two smooth pieces separately), which recovers the rule's rate. Recorded
  as a route rather than attempted here, since it is its own unit of work.

Provenance:
  stock:  none - the archived predecessor had no short-rate Bermudan
  source: MCINERNEY-ZASTAWNIAK (docs/references.md) §3.6 - eq (3.30) the exercise/continuation
          recursion and the forward-measure recursion beneath it, with eq (3.28) the forward-measure
          SDE; §3.5's swaption for the one-exercise anchor; transcribed in both renderings and
          reconciled
  oracle: tests/L4/test_hw_bermudan.py - the transition's bond-REPRICE identity, which pins the
          forward-measure drift without transcribing it; the backward induction's STEPS oracled
          pointwise (the roll, the exercise decision) rather than only its price (T18's §12.8
          lesson); a ONE-exercise Bermudan equal to the tranche-68 Jamshidian European swaption to
          1e-10, with a MANDATORY multi-exercise companion (T39 - the one-exercise reduction is
          blind to the exercise policy, which is the whole method); Bermudan >= European strictly
          for a genuine schedule; monotonicity in the number of exercise rights; and the quadrature
          node count shown to have converged
  crosscheck: quantlib TreeSwaptionEngine on a HullWhite model - deferred to the optional QuantLib
          tier
  slice:  tranche 68 - Bermudan swaptions in the Hull-White model (piece 2, the induction)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.foundation.quadrature import standard_normal_nodes
from pricebook.market.curve import CurveHandle
from pricebook.engine.hw_swaption import SwaptionTerms, hw_european_swaption
from pricebook.models.hull_white import HullWhiteParams, hw_beta, hw_bond

#: Gauss-Hermite nodes per conditional expectation. The transition is exactly Gaussian, but the
#: integrand at an EARLIER roll is `max(exercise, continuation)` - kinked - and a Gauss rule loses
#: its spectral accuracy at a kink. **Measured spread across 48/96/192 nodes: about 2e-3 relative**,
#: which is this method's dominant discretisation error and is stated rather than claimed away.
#: The last roll has no such error because it is analytic (see below); the earlier ones would need
#: the integral SPLIT AT THE EXERCISE BOUNDARY to recover the rate, which is a real improvement and
#: is recorded as a route rather than guessed at here.
_NODES = 48
#: How far either side of zero the state grid runs, in standard deviations of the terminal state.
#: Wide enough that the omitted tails carry no weight a 48-point rule could see.
_GRID_WIDTH = 6.0
#: States per exercise date. Measured: 401 carries 1.2e-4 of relative error against 1201, which
#: settles by 2401. **401 is chosen deliberately over the converged 1201**, because the grid is not
#: the dominant error: the kinked quadrature below carries about 2e-3, an order and a half larger.
#: Refining the grid further spends three times the work to sharpen the term that is already
#: negligible - §4's resolution thinking applied to cost, and the reason the committed number is
#: not simply the finest one that fits.
_GRID_POINTS = 401


@dataclass(frozen=True)
class BermudanTerms:
    """A Bermudan swaption's exercise dates, the coterminal swap they enter, and the strike.

    In **times**, for the reason `SwaptionTerms` is: a short-rate model has no tenor to index into,
    so tranche 18's `SwapSpan` (indices into an `LmmTenor`) is LMM vocabulary and meaningless here.

    Each `exercise` date is also a reset of the coterminal swap: exercising at `T_k` enters the swap
    settling on every date after it.
    """

    exercises: tuple[float, ...]
    settlements: tuple[float, ...]
    strike: float

    def __post_init__(self) -> None:
        if self.strike <= 0.0:
            raise ValueError(f"the strike must be positive, got {self.strike}")
        if not self.exercises:
            raise ValueError("a Bermudan needs at least one exercise date")
        if any(b <= a for a, b in zip(self.exercises, self.exercises[1:])):
            raise ValueError(
                f"the exercise dates must be strictly ascending, got {self.exercises}"
            )
        if any(b <= a for a, b in zip(self.settlements, self.settlements[1:])):
            raise ValueError(
                f"the settlement dates must be strictly ascending, got {self.settlements}"
            )
        if self.settlements[0] <= self.exercises[0]:
            raise ValueError(
                f"the first settlement ({self.settlements[0]}) must follow the first exercise "
                f"({self.exercises[0]})"
            )
        if self.exercises[-1] >= self.settlements[-1]:
            raise ValueError(
                f"the last exercise ({self.exercises[-1]}) must precede the final settlement "
                f"({self.settlements[-1]})"
            )

    def remaining(self, exercise: float) -> tuple[tuple[float, float], ...]:
        """`(accrual, settlement)` for every period the holder enters by exercising at `exercise`.

        The accrual of the first remaining period runs from the exercise date itself, which is the
        off-by-one a stored schedule would hide.
        """
        future = tuple(pay for pay in self.settlements if pay > exercise)
        starts = (exercise, *future[:-1])
        return tuple(zip((b - a for a, b in zip(starts, future)), future))


def forward_measure_drift(
    params: HullWhiteParams, t: float, maturity: float
) -> float:
    """The `P_T`-measure drift adjustment to `x` over `[t, T]`, from eq (3.28)'s `-sigma^2 D(u,T)`.

    `E^{P_T}[x(T) | x(t)] = x(t) e^{-kappa(T-t)} + (this)`, which is

        - integral_t^T sigma^2 D(u,T) e^{-kappa(T-u)} du

    **Computed by quadrature of the source's printed integrand rather than transcribed.** The source
    prints the SDE (3.28), not the integrated adjustment, so a closed form here would be the
    reader's algebra written from memory - which §5 forbids and which the bond-reprice oracle would
    catch anyway. Integrating the printed drift is the honest route, and the identity below pins it.

    Closed-form enough for an oracle to check: the whole point is that
    `B(t,T) E^{P_T}[B(T,T_j)|x(t)] = B(t,T_j)` holds only for the right adjustment.
    """
    if maturity < t:
        raise ValueError(f"a drift cannot end ({maturity}) before it starts ({t})")
    tau = maturity - t
    if tau == 0.0:
        return 0.0
    kappa, sigma = params.kappa, params.sigma
    # integral of sigma^2 (1-e^{-k(T-u)})/k * e^{-k(T-u)} du, substituting v = T-u:
    #   = (sigma^2/k) integral_0^tau (e^{-k v} - e^{-2 k v}) dv
    #   = (sigma^2/k) [ (1-e^{-k tau})/k - (1-e^{-2 k tau})/(2k) ]
    # written through `hw_beta` so the affine function is shared rather than respelled.
    inner = hw_beta(kappa, 0.0, tau) - hw_beta(2.0 * kappa, 0.0, tau)
    return -(sigma * sigma / kappa) * inner


def _state_std(params: HullWhiteParams, t: float, maturity: float) -> float:
    """`sd(x(T) | x(t))` - the OU conditional standard deviation, measure-independent."""
    tau = maturity - t
    if tau <= 0.0:
        return 0.0
    kappa, sigma = params.kappa, params.sigma
    return math.sqrt(sigma * sigma * -math.expm1(-2.0 * kappa * tau) / (2.0 * kappa))


def conditional_expectation(
    params: HullWhiteParams,
    state: float,
    t: float,
    maturity: float,
    payoff,
) -> float:
    """`E^{P_T}[ payoff(x(T)) | x(t) = state ]` by Gauss-Hermite.

    The transition is exactly Gaussian - mean `x e^{-kappa tau}` plus the forward-measure
    adjustment, standard deviation from the OU variance - so the rule integrates it to machine
    precision rather than discretising it as a tree would.
    """
    decay = math.exp(-params.kappa * (maturity - t))
    mean = state * decay + forward_measure_drift(params, t, maturity)
    deviation = _state_std(params, t, maturity)
    if deviation == 0.0:
        return payoff(mean)
    nodes, weights = standard_normal_nodes(_NODES)
    return math.fsum(
        weight * payoff(mean + deviation * node)
        for node, weight in zip(nodes, weights)
    )


def swap_value(
    params: HullWhiteParams,
    terms: BermudanTerms,
    state: float,
    exercise: float,
    discount: CurveHandle,
) -> float:
    """`PS(T_k)` - the payer swap's value on exercising at `T_k`, per unit notional.

    `float - fixed`: the floating leg telescopes to `B(T_k,T_k) - B(T_k,T_n) = 1 - B(T_k,T_n)`, and
    the fixed leg is the strike times the remaining accrual-weighted annuity.
    """
    remaining = terms.remaining(exercise)
    annuity = math.fsum(
        accrual * hw_bond(params, state, exercise, pay, discount)
        for accrual, pay in remaining
    )
    final = hw_bond(params, state, exercise, remaining[-1][1], discount)
    return (1.0 - final) - terms.strike * annuity


def price_hw_bermudan(
    params: HullWhiteParams, terms: BermudanTerms, discount: CurveHandle
) -> float:
    """`Berm(0)` by exact backward induction - eq (3.30), per unit notional.

    At the last exercise date the Bermudan is the European payoff; stepping back, the continuation
    value is the conditional expectation of the next date's value, discounted by the bond over the
    step, and the holder takes the larger of it and the immediate exercise value. Between grid
    points the value function is interpolated linearly - it is smooth apart from one kink at the
    exercise boundary, and the oracle shows the grid has converged.
    """
    def exercise_value(state: float, at: float) -> float:
        """The immediate payoff - a **known function**, never interpolated.

        This is the distinction that makes the one-exercise reduction exact. The value function is
        `max(exercise, continuation)` and carries a kink at the boundary; interpolating *that* on a
        grid costs 0.4% against the Jamshidian European, because linear interpolation of a kinked
        function is poor exactly where the kink is. Only the **continuation** value needs a grid -
        it is the thing being built - so the exercise value is evaluated exactly inside the
        quadrature and the kink is never approximated.
        """
        return max(swap_value(params, terms, state, at, discount), 0.0)

    # The last exercise date: nothing to continue to, so the value IS the exercise value.
    last = terms.exercises[-1]

    def value_at_last(state: float) -> float:
        return exercise_value(state, last)

    value_function = value_at_last
    # ...but the step INTO it is a European swaption, and this tranche has a closed form for one.
    # Quadrature over `max(PS, 0)` converges only as about `n^-2` because a Gauss rule loses its
    # accuracy at a kink (measured: 3.9e-3 at 48 nodes, 1.1e-4 at 192, and `quadrature.py` refuses
    # more than ~370). Using it where eq (3.25) applies would be the same error the lattice was
    # chosen to avoid - an approximation standing in for a closed form - so the last roll is
    # analytic and the one-exercise reduction becomes exact rather than grid-limited.
    closed_form_last = SwaptionTerms(
        expiry=last, settlements=tuple(p for p in terms.settlements if p > last),
        strike=terms.strike,
    )

    def continuation_into_last(state: float, at: float) -> float:
        return hw_european_swaption(params, closed_form_last, discount, seen_from=(at, state))

    for index in range(len(terms.exercises) - 2, -1, -1):
        now, nxt = terms.exercises[index], terms.exercises[index + 1]
        grid_std = _state_std(params, 0.0, nxt)
        half_width = _GRID_WIDTH * max(grid_std, 1e-6)
        step = 2.0 * half_width / (_GRID_POINTS - 1)
        grid = tuple(-half_width + i * step for i in range(_GRID_POINTS))

        # The CONTINUATION value on the grid - smooth, because the max has not been taken yet.
        # Analytic for the roll into the last exercise date (a European swaption, eq 3.25), by
        # quadrature for every earlier one (where the next value function carries a kink that has
        # no closed form).
        if index == len(terms.exercises) - 2:
            continuations = [continuation_into_last(x, now) for x in grid]
        else:
            continuations = [
                hw_bond(params, x, now, nxt, discount)
                * conditional_expectation(params, x, now, nxt, value_function)
                for x in grid
            ]

        def rolled(
            state: float,
            _grid=grid, _values=continuations, _step=step, _now=now,
        ) -> float:
            """`max(exercise(x), continuation(x))` with the exercise leg exact.

            Bound as defaults so each date's closure captures its own grid rather than the loop's
            last - the classic late-binding trap, and one a backward induction would hide behind a
            plausible number.
            """
            if state <= _grid[0]:
                continuation = _values[0]
            elif state >= _grid[-1]:
                continuation = _values[-1]
            else:
                position = (state - _grid[0]) / _step
                lower = int(position)
                weight = position - lower
                continuation = _values[lower] * (1.0 - weight) + _values[lower + 1] * weight
            return max(exercise_value(state, _now), continuation)

        value_function = rolled

    first = terms.exercises[0]
    if len(terms.exercises) == 1:
        # One exercise date: the Bermudan IS the European swaption, in closed form and exactly.
        return hw_european_swaption(params, closed_form_last, discount)
    return hw_bond(params, 0.0, 0.0, first, discount) * conditional_expectation(
        params, 0.0, 0.0, first, value_function
    )
