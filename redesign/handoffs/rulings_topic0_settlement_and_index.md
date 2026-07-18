# Cowork → Build rulings — settlement slice + RateIndex scope (Topic 0)

Two challenges raised after the S4 (money-quantity) merge, before S5. Both accepted; Topic 0 gains
one slice and S5 widens.

---

## 1. Numeraire — real, but NOT a `Money` concern

A PV is always *in units of* something. `Money(amount, currency)` says "in EUR"; a
numeraire-relative value says "in units of the T-bond / the annuity." Swap rates are
annuity-numeraire quantities, quanto pricing is a measure change, and **collateral/CSA discounting
is literally a numeraire choice** — our `discount(ccy, collateral=X)` (D3) *is* selecting one; we
simply never named it.

**Ruling: numeraire is a MODEL/MEASURE concept (L3), not a value type.** Under A1 the model carries
the market, so the numeraire is a property of the model — `DiscountingModel(collateral=USD)` *is*
the choice. Generalising `Money` to carry it would push measure theory into L0 and fail the
membership test (*conventions count time and money; pricing computes value*).

**Do instead:**
- Name the numeraire explicitly at **L3** when the curve/model topic lands (it makes the
  measure-change story teachable — the educational constraint).
- **`PricingResult` records the basis it was computed on** — currency **and** collateral/discounting
  basis — so a PV is never ambiguous about what it is expressed in. Add to the S6 `PricingResult`.

---

## 2. NEW SLICE — `slice/settlement` (Topic 0, before S5)

`Cashflow(date, Money)` covers only cash settlement in the contract currency. Genuine gaps:

- **cash vs physical vs auction** — physical delivers a **`Quantity`** (barrels, MWh) or a security,
  not `Money`; auction settlement is the CDS credit-event path
- **settlement currency ≠ contract currency** — cash-settled quantos, NDFs
- **settlement lag** — T+2 FX, market-specific for securities
- a physical **`Delivery(date, Quantity)`** flow alongside `Cashflow(date, Money)`

**Mine:** `core/settlement.py` (398 LOC) — a **zero-fan-in orphan** in the quarry: a built,
never-wired "physical/cash/auction settlement framework." Prime material; mine the content, ignore
the structure.

**Oracles:** settlement date = trade date + lag under a calendar (published market conventions,
e.g. FX T+2, and the affected currency-pair exceptions); physical vs cash produce different flow
types for the same contract; settlement currency ≠ contract currency round-trips correctly.

---

## 3. S5 `index-identity` — WIDENED: RFR is not all the kinds

The current set (`fixing_lag`/`observation_shift`/`lookback`/`lockout`/`payment_delay`/`compounding`)
is **backward-looking overnight RFR** shaped. Required additions:

**`observation_style: BACKWARD_LOOKING | FORWARD_LOOKING`** — Term SOFR, TORF and every IBOR
(EURIBOR, TIBOR) fix **at the start** for the period ahead. Mechanically `FLAT` + `tenor` handles
them, but making the distinction explicit is what stops someone applying a lookback to a term rate.

**`spread_adjustment` — genuinely missing.** ISDA IBOR fallbacks are RFR **+ a credit adjustment
spread** (synthetic LIBOR, fallback-converted legacy trades). Post-2021 this is everywhere. It is
metadata — the category we ruled must be complete up front.

**Sibling index types, defined under the shared identity concept, populated later:**
- **Inflation** (CPI/RPI/HICP) — a published **level**, not a rate: indexation lag (typically 3M),
  daily-linear vs monthly-flat interpolation, base index. Different metadata entirely.
- **FX fixing** (WM/R 4pm) — fixing source and time; needed by NDFs and quantos.
- **Equity/commodity observation** — for Asians and averaging structures.

`RateIndex` covers **all interest-rate kinds**; the others are siblings, not `RateIndex` fields.
The rule is unchanged: **a new index is a DECLARATION, never a code change.**

**Oracles:** forward-looking term vs backward-looking compounded over the same period give
different rates; a fallback index = base RFR + spread, and the spread is not silently absorbed;
lookback vs observation-shift still differ (from the original spec).

---

## Revised Topic 0 remainder
```
S4  money-quantity     ✔ merged
S4b settlement         ← NEW: cash/physical/auction · settlement ccy · lag · Delivery(Quantity)
S5  index-identity     WIDENED: all rate kinds (observation_style, spread_adjustment)
                       + inflation / FX / equity-commodity index types under the shared concept
S6  numerics-config    + PricingResult records currency AND collateral/discounting basis
→   TOPIC 0 GATE       park the set to parked/topic-00-foundation/
```
Cadence: S4b/S5/S6 is three slices — one checkpoint at the Topic 0 gate.
