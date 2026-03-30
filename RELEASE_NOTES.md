# Release Notes

---

## v0.8.0 — 2026-03-30

PricingContext, European swaptions, and swaption vol surface.

- PricingContext: bundles valuation date, discount/projection curves, vol surfaces, credit curves, FX spots into one object
- Swaption: European payer/receiver on a vanilla IRS, Black-76 pricing on the forward swap rate
- Swaption pv_ctx: price from a PricingContext with named curves and vol surfaces
- SwaptionVolSurface: 2D expiry×tenor grid with bilinear interpolation, flat extrapolation
- Round-trip validated: payer-receiver parity (ATM + OTM), ATM symmetry, vega/delta bump-and-reprice, PricingContext consistency, vol surface integration
- 441 tests, 96% coverage

---

## v0.7.0 — 2026-03-29

European options, Black-76, and IR caps/floors — first options slice.

- Black-76 model: call/put pricing on a forward, handles zero vol and expiry edge cases
- Bachelier (normal) model: arithmetic Brownian motion, works with negative rates
- Analytical Greeks: delta, gamma, vega, theta with bump-and-reprice cross-checks
- Vol surface: flat vol and vol term structure (strike dimension anticipated, not yet built)
- IR cap/floor: strip of caplets/floorlets priced with Black-76
- Round-trip validated: put-call parity (parametrised), ATM delta ≈ ±0.5, vega maximised ATM, cap-floor parity
- 382 tests, 97% coverage

---

## v0.6.0 — 2026-03-29

FX forwards, swaps, and cross-currency basis — second currency.

- Currency and currency pair with market quoting conventions (EUR, GBP, USD)
- FX forward: covered interest rate parity pricing, forward points, PV
- FX swap: near/far legs, swap points, fair valuation
- Cross-currency basis: implied spread from market forwards, basis curve bootstrap
- Round-trip validated: CIP holds, triangular consistency (EUR/USD + GBP/USD = EUR/GBP), basis curve reprices all forwards
- 324 tests, 98% coverage

---

## v0.5.0 — 2026-03-29

CDS and credit curve — third asset class.

- Survival curve: survival probabilities, hazard rates, default probabilities
- CDS protection leg: discretised integration with mid-point approximation, analytical cross-check
- CDS premium leg: scheduled coupons contingent on survival, accrued-on-default approximation
- CDS instrument: PV, par spread, upfront, risky annuity (RPV01)
- Credit curve bootstrap: strip survival probabilities from CDS par spreads using OIS discount
- CS01: credit spread sensitivity via bump-and-reprice
- Risky bond cross-check: risk-free price minus credit adjustment
- 273 tests, 98% coverage

---

## v0.4.0 — 2026-03-29

FRA, OIS, and dual-curve framework.

- Forward rate agreement (FRA): single-period forward rate contract
- Dual-curve floating leg, swap, and FRA: separate projection and discount curves
- OIS swap: compounded overnight rate with telescoping PV
- OIS bootstrap: strip OIS par rates into a risk-free discount curve
- Dual-curve bootstrap: forward curve from IRS par rates, discounting off OIS
- Round-trip validated: OIS reprices, IRS reprices dual-curve, FRAs consistent, single-curve recovery exact
- 220 tests, 98% coverage

---

## v0.3.0 — 2026-03-28

Fixed-rate bonds and risk sensitivities.

- Fixed-rate bond: dirty/clean price, accrued interest
- Yield to maturity: Brent solver (extracted to shared solvers module)
- Macaulay duration, modified duration, convexity, yield DV01
- Curve-based risk: parallel bump DV01, key rate durations (bump and reprice)
- Round-trip validated: YTM recovery, analytical duration matches bump risk, convexity improves approximation
- 181 tests, 97% coverage

---

## v0.2.0 — 2026-03-28

Interest rate swaps and full yield curve bootstrap.

- Schedule generation: monthly, quarterly, semi-annual, annual frequencies with stub handling and end-of-month rule
- Fixed leg: cashflow generation, present value, annuity factor
- Floating leg: forward rate projection from discount curve, spread support
- Interest rate swap: payer/receiver direction, PV, par rate
- Curve bootstrap: deposits (short end) + swap par rates (long end), Brent root finder
- Round-trip validated: all input instruments reprice, forwards positive, dfs decreasing
- 139 tests, 97% coverage

---

## v0.1.0 — 2026-03-28

Foundation layer: the building blocks for curve construction.

- Day count conventions: ACT/360, ACT/365F, 30/360
- Business day calendar: USD settlement (NYSE/SIFMA), adjustment conventions (following, modified following, preceding, modified preceding)
- Money market deposit: cashflow, discount factor, present value
- Discount curve: built from discount factors, queries for df, zero rate, forward rate
- Interpolation: linear, log-linear, cubic spline, monotone cubic (Hyman filter)
- Round-trip validated: deposits bootstrap into a curve and reprice to zero
- 79 tests, 97% coverage
