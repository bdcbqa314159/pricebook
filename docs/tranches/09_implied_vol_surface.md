# Tranche 9 — the implied-vol surface inversion (§5.2)

**Status:** SPLIT per §4, both halves BUILT. **(a)** v0.9.0 — Black at L3 + the inverse.
**(b)** v0.10.0 — the strip and the surface. Decided by Cowork, built by Code. **Results in §7, §10.**
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

**Parameterised by TOTAL VARIANCE, not `(σ, T)` — ruled, confirming (a)'s build.** `black_price`
takes `V²` because JY's variance (eq 2.6.4) is an eight-term total variance, *not* `σ²·T`; a
`(σ, T)` signature would force the caplet pricer to invent a `σ` the model does not have. **The
surface node stores `σI` (annualised)**, matching Figures 5.3/5.4's axis: `black_implied_vol`
annualises at its boundary, `σI = √(V²_implied / τ)` with `τ` on the **vol clock (ACT/365F, the C2
convention)** — the one place annualisation happens. Internally the strip works in variance; the
round trip (oracle 2) must convert back, `V² = σI²·τ`, or it will look wrong for a right reason.

## 2. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **ANCHOR — the strip is a decomposition, tested as one, on a DISCRIMINATING fixture.** A floor to `Tₙ` is `Σ` floorlets to `T₁..Tₙ`; `floorletᵢ = floorᵢ − floorᵢ₋₁` with `floor(T₀)=0` (the direction asserted, not assumed), and the stripped floorlet prices equal floorlets **independently priced in the test from a known σI surface**. The count (N expiries → N floorlets) is the *population floor* — it proves none was dropped, not that they are right. The **discriminator is the per-node value match, and it only has power on a fixture where consecutive floorlets DIFFER** (≥3 expiries, a sloped σI and/or varying forwards): a flat/equal strip passes a mis-indexed or reversed differencing silently, because every node holds the same number. State the distinct-values premise, as T6/T8 had to | **1 — structural** | tests the structure, not a scalar off it (the T3/T8 strip pattern); a mis-indexed or reversed difference moves a specific middle node, which only distinct per-node values reveal |
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


---

## 7. Result — (a) built, and the layering call is confirmed empirically

**Split taken**, per §4's pre-authorised cut: **(a)** Black extracted to L3 + `black_implied_vol`
with oracles 2, 3, 5; **(b)** the floor→floorlet strip + the surface with oracles 1, 4, composing
(a). 204 tests, eight gates green, ruff and pyright clean, zero suppressions.

### The layering call — CONFIRMED, by running the gates, not by reading them
§1 asked Code to confirm that an L3 builder may construct and return an L1 value. It may: a throwaway
`models/_probe.py` importing `market.curve.DiscountCurve` and returning one passes both `acyclic`
and `layers`. The import is L3→L1, which is *downward* and therefore allowed; nothing about the
*return* type is constrained. Probe deleted; the gates were re-run after removal to confirm the
baseline. So §1's refinement of §2 stands as written, and (b) can be built on it.

### Black is parameterised by TOTAL VARIANCE, not `(σ, T)`
Worth recording because it is not the Black-76 textbook signature. JY's `V²` (eq 2.6.4) is a total
variance assembled from eight terms and is **not** `σ²·T`, so a `(σ, T)` signature would force
every caller to invent a `σ` the model does not have. Annualisation appears only at the inverse's
boundary, where §5.2's surface wants `σ_I` per (expiry, strike).

### The extraction is bit-identical, and that was measured
Eight cap/floor strip prices (4 strikes × cap/floor, including a negative strike) were captured as
`repr()` before the refactor and compared after: **all eight identical to the last bit**, alongside
T5's own oracles staying green. "Pure refactor" is a claim that deserves evidence, not an assertion.

## 8. Four things found while building, three of them my own errors

### 8.1 My no-arbitrage bound test asserted `<` where floating point gives `<=`
Mathematically a cap's price approaches `F` and never reaches it. At `V² = 10⁴` it *does*, exactly:
`Φ(ω·d₁)` saturates to 1.0 and `Φ(ω·d₂)` to 0.0, leaving `F·1 − K·0 = F`. The test was wrong, not
the code. Fixed to `<=`, **and the saturation is now asserted explicitly** — it is precisely why
`black_implied_vol` refuses a price at the limit rather than returning whatever its expansion
reached, so it is a load-bearing property rather than a rounding curiosity.

### 8.2 My `rtol=1e-15` was LOOSER than scipy's default
`brentq`'s default `rtol` is `4·eps ≈ 8.9e-16` — the tightest value it accepts. Passing `1e-15`
relaxed it. Dropped, so the solve now runs at scipy's floor. Found because pyright rejected the
argument's type, which turned out to be a real finding rather than a stub nuisance.

### 8.3 The `layers` gate caught a finance word in an L0 docstring — reworded, not loosened
`foundation/solvers.py`'s docstring referred to `black_implied_vol`, and `implied_vol` is finance
vocabulary in a finance-free layer. Reworded to "a monotone 1-D inverse at L3", following T4's
precedent exactly: the gate is right, the prose moves.

### 8.4 The debt gate caught a comment *mentioning* a suppression marker
A comment explaining why the scipy union is narrowed rather than silenced spelled out the marker's
literal text, and the debt gate counted it — `suppressions (1) != ledger (0)`. **The gate is right
to be literal**: one that parsed intent could be talked out of flagging a real suppression.
Reworded. Zero suppressions stands, and it is now a slightly harder thing to say by accident.

## 9. Still to build — (b)
The floor→floorlet strip (oracle 1, the non-telescoping anchor: a count plus differencing against
independently-priced floorlets) and the `ImpliedVolSurface` assembly with oracle 4's
reduction-plus-companion. Its own branch and PR, composing (a).


---

## 10. Result — (b) built, and a 50% defect the parameter NAME was hiding

224 tests, eight gates green, ruff and pyright clean, zero suppressions.

### The headline: `expiry` was the wrong name, and (b) proved it costs up to 50%
(a) shipped `black_implied_vol(price, forward, strike, **expiry**, kind)`. At its second consumer
that name turned out to be wrong: the divisor is the horizon the **variance** accumulates over, and
for a year-on-year floorlet that is the **period length**, not the time to expiry. The floorlet on
`[2y, 3y]` expires at 3y and accumulates one year of variance, because `V² = σ_I²·(Tᵢ − Tᵢ₋₁)` under
simplified JY (tranche 5's `..._variance_IS_index_vol_squared_times_the_period` is the oracle).

Measured, by passing the expiry deliberately:

| node | true `σ_I` | correct | if the expiry were passed | error |
|---|---|---|---|---|
| 1 | 0.00800 | 0.00800 | 0.00800 | **0.0%** |
| 2 | 0.01200 | 0.01200 | 0.00849 | **−29.3%** |
| 3 | 0.01800 | 0.01800 | 0.01039 | **−42.3%** |
| 4 | 0.02500 | 0.02500 | 0.01250 | **−50.0%** |

**Node 1 agrees exactly**, because its period starts at the anchor — so a single-period fixture
could never have caught this, and every oracle would have stayed green with the forward and inverse
agreeing on the wrong horizon. Renamed to `variance_years`, and the trap is now pinned by
`test_passing_the_EXPIRY_instead_of_the_PERIOD_would_understate_the_vol`, asserted on four periods
with the error required to *grow* with maturity.

This is the writing-time rule earning its place: *if a name needs a comment to explain it, the name
is wrong.* It needed one.

### Oracle 1 built as Cowork sharpened it
The count is the **population floor** (no node dropped); the **discriminator** is the per-node value
match on a fixture where consecutive floorlets differ — four periods, a sloped `σ_I` (0.008 → 0.025)
*and* sloped curves. Three premises are asserted rather than assumed: the floorlets are pairwise
distinct with gaps > 1.0; a **reversed** generating strip produces a different middle node (so a
reorder is detectable, which a count never shows); and `floor(T₀) = 0`, checked by the single-quote
case where an off-by-one in the differencing direction lands.

Two strikes also strip **independently**, with a test that bumping one column leaves the other
bit-identical — a builder differencing across *strike* instead of maturity would still produce the
right node count.

## 11. Findings for Cowork (§10 — raised, not decided)

### 11.1 Out-of-order quotes are sorted, not refused — my test had the requirement wrong
My RED suite asserted that quotes out of maturity order *raise*. The builder sorts them, and on
reflection sorting is right: market quotes arrive in whatever order a feed hands over, and refusing
them pushes a sort onto every caller for nothing. The test became "arbitrary order gives the **same**
surface", plus a new one for the case sorting genuinely cannot fix — a **duplicate** maturity at one
strike, which differences to a zero-price floorlet and would imply a bogus zero vol.

### 11.2 The surface does not interpolate, and that absence is now asserted
§3.4 deferred interpolation. `vol()` raises on an off-node request, and a test asserts it — so the
absence reads as a decision rather than an oversight, and a future consumer needing an off-node vol
has to add the scheme deliberately.

### 11.3 `market/vol_surface.py` carries `crosscheck: none` on the container
Per §8's criterion it holds no numerics — the numbers it stores are cross-checked where they are
computed (`models/black.py`, `models/implied_vol_surface.py`, both with real targets). Flagged
because it is a judgement under your criterion and therefore yours to overrule.
