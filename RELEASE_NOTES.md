# Release Notes

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
