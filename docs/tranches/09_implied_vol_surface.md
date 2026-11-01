# Tranche 9 — the implied-vol surface inversion (§5.2)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.9.0`.
**Source:** `BERHAN-2012` §5.2 (`worked-from`) — strip quoted floor prices into floorlets, imply a
Black vol per (expiry, strike), assemble the surface of Figures 5.3/5.4.

## 0. What this is, and what it is NOT

§5.2's model — the "simplified Jarrow–Yildirim" — is **already ours**: it is T5's Black reduction
(`σn=σr=0`), which the thesis itself proves equivalent (coverage map). What is missing is the
**inversion**: given market floor quotes, recover the Black implied vol per node and build the
surface. This is the **market's** view of vol; it is NOT a claim that JY reproduces it — the thesis
is explicit that the model fits ATM and not the smile (§5.3.2). T9 builds the market-data side; the
skew the surface shows is exactly what the model cannot fit, and saying so is the honest scope.

## 1. Pieces and layers

```
L3  models/black.py        — the Black formula + black_implied_vol (its inverse)
                             EXTRACTED from T5's inline caplet and SHARED (see §3.1); the inverse new
L1  market/vol_surface.py  — ImpliedVolSurface: a frozen (expiry, strike) → σI value, market data
L3  the surface BUILDER    — strip floor quotes → floorlets, invert Black per node → the L1 surface
```

**Black is L3, not L0.** It is a finance analytic (strike, vol, call/put), and L0 is finance-free —
the migration's own `foundation/black.py` is the named precedent for this exact drift. It lives in
`models/`.

**The builder is L3, though the surface it returns is an L1 type.** It composes the L3 Black inverse,
so it cannot sit at L1 (`acyclic`). This refines §2's "produces market data → L1": that holds for a
bootstrap using only L0+L1; a market-data object built *through a model analytic* is built at the
analytic's layer and returns the L1 type. **Code: confirm `acyclic`/`layers` accept an L3 builder
constructing an L1 value; if they object, raise it — I may have the split wrong and will re-rule.**

## 2. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **ANCHOR — the strip is a decomposition, tested as one.** A floor to `Tₙ` is `Σ` floorlets to `T₁..Tₙ`; differencing recovers each floorlet, and the stripped floorlet prices equal floorlets **independently priced in the test from a known σI**. A *count* too: an N-expiry floor strips to N floorlets | **1 — structural** | tests the structure, not a scalar off it (the T3/T8 strip pattern); a mis-indexed difference or a dropped node shows here |
| 2 | **Black inversion round trip (consistency guard — telescopes, leans on T5):** price a floorlet at known σ through Black, imply it back, recover σ to 1e-10. It cannot catch a shared Black error — **but T5 already oracled the forward Black against the thesis independently**, so with that prior it proves the root-find inverter is correct | self-consistency | the root-find converging to the wrong root |
| 3 | **monotonicity:** implied σI strictly increases with floorlet price | **structural** | a wrong root or a sign error, invisible to a single round trip |
| 4 | **reduction + companion:** a FLAT σI input → prices → strip → invert → flat σI recovered (reduction); a **sloped** σI input recovers the slope, not a constant (companion — the inverter is not returning a fixed number) | exact / structural | the charter's reduction clause; the companion is mandatory |
| 5 | **no implied vol is a RAISE, not a guess.** A price below intrinsic or above the no-arbitrage max has no Black vol → `ValueError` (the L1/L3 constructor convention, T8); `PricingFailure` stays the engine path | structural | failure-is-a-value at the right boundary — a constructor raises, it does not invent a vol |

Oracle 1 is the anchor because it is non-telescoping; oracle 2 is a guard that only earns its keep
sitting on T5's independent forward-Black oracle — state that dependency in the test so it is not
mistaken for the anchor.

## 3. Design decisions (ruled)

### 3.1 Black is extracted to L3 and shared — rule-of-two fires here, correctly
T5 computed Black inline in `engine/capfloor.py` (right then — one consumer). T9 is the **second**
consumer (the inverter needs the same Black), so §3d + rule-of-two say extract it now: one
`models/black.py` the caplet pricer and the inverter both compose, so a Black convention cannot
diverge between pricing and implying. The caplet refactor is a **pure refactor guarded by T5's
already-green oracles** — no new RED, but run those oracles (the §8/§9 carve-out), and the caplet's
numbers must not move by a bit.

### 3.2 `black_implied_vol` uses the solver seam, with a bracket that cannot miss
Black is monotincreasing in σ on `[0, ∞)`, so the inverse is a 1-D root-find with a guaranteed
bracket (price at σ=0 is intrinsic, price → the no-arb max as σ → ∞). Use `foundation/solvers.py`;
the bracket is structural, not a magic constant — oracle 5 is the out-of-bracket case.

### 3.3 Cross-check dispositions for the new modules (§8 gate)
- `models/black.py` → `crosscheck: quantlib blackFormula / blackFormulaImpliedStdDev`
- the builder → `crosscheck: quantlib (floorlet stripping + implied vol)`
- `market/vol_surface.py` → `crosscheck: none — a frozen (expiry,strike)→σ container, no numerics`

### 3.4 Not building
Smile *modelling* (a SABR/other fit to the surface — the thesis does not, and §5.3.2 says JY cannot);
vol interpolation *between* nodes (the surface stores the implied nodes; interpolation arrives with
a consumer that needs an off-node vol); Market Model II's vols (ch 4).

## 4. Split if the oracle will not check in one pass (§6)
This is larger than a typical rung (a Black extraction + the inverse + the strip + the surface). If
oracle 1 and oracle 2 cannot both be checked in one pass, split:
- **(a)** extract Black to L3 + `black_implied_vol` with oracles 2, 3, 5 (the single-node inverse);
- **(b)** the floor→floorlet strip + the surface assembly with oracles 1, 4.
Build (a) first; (b) composes it. Either lands as its own PR.

## 5. Stock
**Mine for content:** `_stock/.../options/inflation_smile.py` and `inflation_vol.py` for the
stripping shape and the implied-vol solve — **the formulas and the strip method, not the structure**,
and read the Black inverse from the source/standard Black-76, not from the stock's code (it may be a
different parameterisation). Build fresh on `models/black.py`.

## 6. Process
Branch `tranche/09-implied-vol-surface` (or `-a`/`-b` if split). `test` (RED) → `feat` → `docs` →
`chore(release): v0.9.0`. `source:` cites `BERHAN-2012` §5.2 + standard Black-76. The Black
extraction's provenance notes it supersedes the inline caplet Black, and T5's oracles are re-run
green to prove the refactor moved nothing.
