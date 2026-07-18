# Cowork → Build ruling — spine-conformance correction (do BEFORE the CP-5 retire sweep)

Raised by Bernardo: `foundation/black.py` is mis-homed, and more broadly the build has been
"moving in all directions, re-introducing volatility in the code." **Both parts are correct.**
This ruling fixes the file *and* the process gap that allowed it.

## 1. The violation (confirmed)
`foundation/black.py` — Black-76 — sits in **L0**, whose ratified definition is *"time &
conventions · value types · **finance-free** numerics."*
- **18 finance-term hits** in that module (strike, vol, call/put, discount, moneyness). Not
  finance-free. Its own docstring self-declares "L0 numerical toolkit" — that claim is wrong.
- **Quarry provenance is `pricing/`**, not `core/`.
- **A4.3** (ratified): a closed-form analytic block of a *dynamics* belongs at **L3 models**;
  the L4 engine composes it. Black-76 is exactly that.
- Consumers are `engine/fx_option.py` + `engine/spot_option.py` — both **L4**, so nothing forces
  L0 placement. It is a semantic choice, and the wrong one.

*(`foundation/distributions.py`'s single finance hit is a docstring reference to the ZCB oracle —
benign. `black.py` is the only real violation found.)*

## 2. Why it slipped — the process gap (Cowork's)
`acyclic` **passed**: foundation imports nothing, engines import downward, no cycle. The
*dependency* rule was satisfied while the *semantic* rule was violated. And the four mandatory
checkpoint review inputs (oracle / drawdown / challenge / smell-debt) contained **nothing checking
layer conformance**. Nobody was ever asked *"does this module belong in the layer it sits in?"*

Compounding it: the drawdown-first push has the retire-reads pulling the build opportunistically
across the tree, with no structural gate. That is the mechanism behind the "volatility" concern —
the spine was enforced only where mechanically checkable, and placement is not.

## 3. The correction — full scope, and it runs FIRST (before CP-5's retire sweep)

```
BRANCH slice/spine-conformance

1. refactor: move foundation/black.py -> models/black.py (behaviour-preserving; update the two
   L4 engine imports; fix the docstring's false "L0 numerical toolkit" claim; provenance header
   -> quarry pricing/ (black), layer L3).
2. AUDIT all ~50 ng modules for placement drift against each layer's definition:
   L0 finance-free · L1 immutable market data · L2 products = pure data (no pricing) ·
   L3 dynamics/calibration (may expose analytic blocks) · L4 engines compose to price ·
   L5 risk on Priceable · L6 state/lifecycle only. Report every module moved + every one
   consciously confirmed in place.
3. feat(verify): `verify.py layers` now FAILS on semantic misplacement — mechanised rule:
   **no finance vocabulary in `foundation/` code** (strike|vol|volatility|payoff|call|put|
   discount|moneyness|option), docstrings/comments excluded. Wire into the merge gate + CI.
4. Oracles: pure refactor — every existing FX/equity/commodity option oracle must stay green
   (§8 carve-out: no new RED required, but run them). verify all green both OSes.
5. CHANGELOG (Changed), provenance updated, version PATCH.
```

## 4. Standing process change (now canonical)
- **`CLAUDE.md §1`** — "Layer conformance is SEMANTIC, not just acyclic," with the `black.py`
  precedent recorded so the failure mode is remembered.
- **`redesign/11` §2** — a **5th mandatory checkpoint review input: spine-conformance audit**.
  Every module created or moved since the last checkpoint is justified against its layer's
  definition, not merely against `acyclic`.
- **`redesign/09`** — `verify.py layers` upgraded from *print* to *enforce*.

## 5. Then CP-5
Only after this lands does the CP-5 retire-read sweep begin — so the sweep runs across a clean,
gated structure rather than compounding drift. Retirement pace must not buy structural erosion:
**a tick that costs spine conformance is not a win.**
```
