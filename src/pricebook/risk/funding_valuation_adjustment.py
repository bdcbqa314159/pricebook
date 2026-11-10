"""FVA - the funding valuation adjustment (L5): a fixed point with a closed form.

Crepey defines the funding adjustment implicitly, as a **semilinear fixed point** (eq 19), and then
proves it has a unique explicit solution (Prop 1, eq 17):

    FVA = gamma * (P - CVA - FVA)^+                     (19)  the definition
    FVA = gamma/(1+gamma) * (P - CVA)^+                 (17)  the solution

`P` is the clean (risk-free) mark of the netting set, `CVA` the counterparty adjustment of tranche
27, and `gamma` the bank's funding spread over the horizon. The definition is circular on purpose:
funding is needed for the *net* position after adjustments, and the adjustment is part of what has to
be funded. The `^+` is the economics: a **net receiver** is not funding anything, so its FVA is zero.

**Why the closed form is production and the iteration is only a guard.** The algebra is immediate -
with the positive part open, eq 19 reads `FVA = gamma(P-CVA) - gamma*FVA`, so `FVA(1+gamma) =
gamma(P-CVA)` - and the result holds for **every** `gamma >= 0`. The obvious alternative, solving eq
19 by Picard iteration from `FVA_0 = 0`, is *strictly weaker*: its error falls by a factor of exactly
`gamma` per step, so it converges only for `gamma < 1`, oscillates at `gamma = 1`, and diverges above
it - while the closed form keeps satisfying the fixed point throughout. A `gamma >= 1` funding spread
is extreme, but "the iteration happens to converge for realistic inputs" is not a reason to compute
with it when an exact solution exists. (The tranche spec reasoned the iteration contracts at
`gamma/(1+gamma) < 1`; measured, the factor is `gamma`. `gamma/(1+gamma)` is the *solution's*
coefficient, not the map's. Reported, and it only reinforces the ruling.)

**`CA = CVA` here, and that is an assumption of the uncollateralized spine, not an identity.** Crepey
writes eq 17's first form with `CA` (the total credit/funding asset), which equals `CVA + FVA` in
general (eq 11); the substitution to `CVA` in the second form is his own, and it holds because this
spine has **no collateral and no initial margin**. Margin valuation adjustment and the collateral
terms that make `CA != CVA` are deferred (T29 / `to_go_deeper`), so a caller who later adds collateral
must revisit this, which is why the assumption is named here rather than assumed silently.

Provenance:
  stock:  none - the archived predecessor's `risk/xva.py` was a phantom (tranche 24 pruned its rows)
  source: CREPEY-2026 (docs/references.md) Prop 1 - eq (17) the closed form, eq (19) the semilinear
          fixed point it solves. The `gamma/(1+gamma)` coefficient is printed as a stacked fraction,
          so it was read in both `-raw` and `-layout` and reconciled (the eq-25 reflow hazard), and
          independently *derived* from eq 19, which §4 rates above a transcription
  oracle: tests/L5/test_funding_valuation_adjustment.py - the closed form against an independent
          root-find on the defining equation across the spread range; the **anchor**, that the closed
          form satisfies its own fixed point to machine precision, with a companion proving the
          natural reflow misreading `gamma*(P-CVA)^+` does NOT satisfy it; both regimes of the `^+`;
          the iteration's geometric rate measured at `gamma` and its domain limit characterised
  crosscheck: none - FVA as a balance-sheet adjustment is a modelling frame rather than a numerical
          convention a library implements independently; QuantLib has no FVA functional to compare
  slice:  tranche 28 - FVA, the last core unit of the XVA spine
"""

from __future__ import annotations

import math


def _checked_net_mark(clean_mark: float, cva: float, funding_spread: float) -> float:
    """`P - CVA`, with the inputs validated once for every entry point here."""
    if not math.isfinite(clean_mark) or not math.isfinite(cva) or not math.isfinite(funding_spread):
        raise ValueError(
            f"the mark, CVA and funding spread must all be finite, got "
            f"{clean_mark}, {cva}, {funding_spread}"
        )
    if funding_spread < 0.0:
        # Crepey: "as gamma (a probability) is nonnegative, this equation has the unique solution".
        raise ValueError(f"the funding spread may not be negative, got {funding_spread}")
    if cva < 0.0:
        raise ValueError(f"a CVA may not be negative, got {cva}")
    return clean_mark - cva


def funding_valuation_adjustment(clean_mark: float, cva: float, funding_spread: float) -> float:
    """`gamma/(1+gamma) * (P - CVA)^+` - eq 17, the unique solution of eq 19's fixed point.

    Exact for every `gamma >= 0`, which the Picard iteration is not (see the module docstring).
    """
    net = _checked_net_mark(clean_mark, cva, funding_spread)
    if net <= 0.0:
        # A net receiver funds nothing. Returned as an exact zero, not a small number: this is the
        # `^+` binding, which is a statement about the economics and not a rounding outcome.
        return 0.0
    return funding_spread / (1.0 + funding_spread) * net


def fva_fixed_point_residual(
    fva: float,
    clean_mark: float,
    cva: float,
    funding_spread: float,
) -> float:
    """`FVA - gamma*(P - CVA - FVA)^+` - eq 19's defining equation, as a residual.

    Named surface rather than a step inside a test, because this is the tranche's **correctness
    anchor** (§4: a formula and its stated inverse must actually invert) and §4 also says to extract
    the thing an oracle needs to reach. It is zero at the true FVA, and materially non-zero for the
    reflow misreading `gamma*(P-CVA)^+` - which is what gives the anchor its power rather than
    leaving it a property a whole family of formulas would satisfy.
    """
    net = _checked_net_mark(clean_mark, cva, funding_spread)
    return fva - funding_spread * max(net - fva, 0.0)


def iterate_fva(
    clean_mark: float,
    cva: float,
    funding_spread: float,
    iterations: int,
) -> tuple[float, ...]:
    """The Picard iteration `FVA_{n+1} = gamma*(P - CVA - FVA_n)^+` from `FVA_0 = 0`.

    Returns the whole sequence including `FVA_0`, so a test can read the **rate** and not merely the
    limit. A consistency guard only, and a *telescoping* one: it shares `gamma` and `P - CVA` with
    the closed form, so agreement is a weaker kind of evidence than the fixed-point residual.

    **Its domain is `gamma < 1`.** The error obeys `e_{n+1} = -gamma*e_n` exactly while the positive
    part stays open, so the sequence converges geometrically at `gamma`, oscillates without
    converging at `gamma = 1`, and diverges above it. Not refused, because characterising that
    boundary is more useful than hiding it - and the contrast is the argument for computing with the
    closed form instead.
    """
    if iterations < 1:
        raise ValueError(f"the iteration needs at least one step, got {iterations}")
    net = _checked_net_mark(clean_mark, cva, funding_spread)
    fva = 0.0
    path = [fva]
    for _ in range(iterations):
        fva = funding_spread * max(net - fva, 0.0)
        path.append(fva)
    return tuple(path)
