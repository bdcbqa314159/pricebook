# Tranche 33b — the engine registry dispatches on (product, model)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.40.0`.
**Source:** none external — an infrastructure fix of our own dispatch, pulled by T33's finding (one
product now has two model-pricers). **Pre-registered RED (the coexistence test + bit-identical purity).**
Runs **before T34**; it is not an equity tranche, it is the registry the hybrids will register into.

## 0. The finding, and why now
T33 verified empirically that the engine registry keys on **product type alone**
(`_PRICERS[product_type] = pricer`), so a second `@register(EuropeanEquityOption)` **silently replaces**
T30's Black pricer — a real correctness footgun (Code registered a stub and watched
`price_european_equity_option` vanish from the table). It never bit before because every product had one
engine; the **European equity option is the first with two** (Black T30, Heston T33, local-vol-MC T32),
which is the rule-of-two second consumer that decides the design (§6). The §3 contract already takes the
model — `price(product, model)` — so the model belongs in dispatch.

## 1. The change
- **`_PRICERS` keys on `(product_type, model_type)`.** `@register` declares (or derives from the
  signature) the model type it prices; dispatch in `price(product, model)` looks up
  `(type(product), type(model))`.
- **The key is `(product_type, model_key)`, where `model_key` is a concrete model type OR a
  `@runtime_checkable` protocol.** *(Corrected 2026-11-12 on Code's 0175 — my original "each engine has
  one model" premise was WRONG: two of eight pricers are typed against the `IndexRatioModel` **protocol**
  and serve four concrete models (JY, MM-I, MM-II, a test stub). Keying those on a concrete type loses
  the deliberate polymorphism §3 and the protocol docstring intend, failing 63 tests. My premise error —
  read-before-asserting on our own tree (§10/§5), a one-line grep of the registrations would have caught
  it; the T22 instance on the Cowork side.)* So:
  - a **concrete-model** pricer (Black-equity, Heston-equity) registers under `(product, concrete type)`;
  - a **protocol-polymorphic** pricer (the inflation cap/floor and YoY engines over `IndexRatioModel`)
    registers **once** under `(product, protocol)` and keeps serving every model satisfying it.
- **Dispatch resolves exact-concrete first, then protocol.** `price(product, model)` looks up
  `(type(product), type(model))`; if absent, the **unique** `(type(product), P)` whose protocol `model`
  satisfies (`isinstance`, `@runtime_checkable`); else `PricingFailure`/a clear error. Exact beats
  protocol, so a concrete pricer never shadows the polymorphic one by accident; **two protocol matches
  for one product is a loud ambiguity error** (as a duplicate concrete registration is). This makes the
  test stub (`_StubRatioModel`, in `tests/L3/`) dispatch via the protocol with **no src-side
  registration** — your option (c), resolved.
- **`heston_call_price` and the Black equity pricer register distinctly** under
  `(EuropeanEquityOption, HestonModel)` and `(EuropeanEquityOption, <flat-vol/Black model>)` — the two
  that collided, now coexisting.
- **A double registration of the *same* `(product, model)` is an error, not a silent overwrite** — the
  footgun becomes a loud failure at import/registration, the §8 "no debt silenced silently" shape.
- Fix `models/__init__.py`'s missing docstring (imports above it) while here — cosmetic, T33's note.

## 2. The oracle (RULED) — a pure refactor plus the thing that was impossible before
- **Pure refactor, bit-identical (§4).** Every currently-registered product prices **bit-identically**
  through the new `(product, model)` dispatch — the lookup key changed, not an operation. Prove it
  bit-identical (the keys/ops are unchanged, only the dict key widens), not ULP; the whole suite staying
  green is the corroboration, the bit-identity is the proof.
- **The coexistence test — the RED that was impossible before (§4 pre-register).** Register two pricers
  for `EuropeanEquityOption` under **different** models (Black and Heston) and assert **both resolve to
  their own pricer** — the exact thing that silently clobbered before now works. Pre-register it red
  (it cannot pass on the product-only registry).
- **Protocol polymorphism preserved (the premise fix's own oracle).** The inflation cap/floor and YoY
  engines must still price under **all** of JY, MM-I, MM-II **and** the test stub — each resolving to
  the one protocol-keyed pricer by `isinstance`, bit-identically to before. This is where the 63
  failures live; it is the RED that proves the protocol path works. Assert **exact beats protocol**: if
  a product ever has both a concrete `(product, M)` and a protocol `(product, P)` pricer with `M`
  satisfying `P`, the concrete one wins (a dedicated test with a throwaway product).
- **The loud-failure requirement.** Registering the *same* `(product, model_key)` twice **raises**
  (asserted); an unresolved `(product, model)` returns `PricingFailure`/a clear error, never a silent
  wrong pricer; **two matching protocols** for one product raise ambiguity. **Mutation:** revert the key
  to product-only (→ coexistence fails, the clobber returns); drop the double-registration guard (→
  loud-failure fails); resolve protocol **before** exact (→ the exact-beats-protocol test fails). Each
  must die.
- **No degenerate-fixture risk to assert beyond the above** — the dispatch is exercised by every
  existing tranche's tests, which now all route through the two-tuple key; a failure there would show at
  once.

## 3. Process
Branch `tranche/33b-engine-registry`. `test` (RED: the coexistence test + the double-registration
loud-failure, both failing on the product-only registry) → `feat` (widen the key, update all
registrations) → `docs` → `chore(release): v0.40.0`. **Open the draft PR early.** `source:` none.
`crosscheck: none — a dispatch mechanism, no external library implements our registry`. **Raise an
`Ask:`** only if widening the key surfaces a product that was *relying* on the clobber (it should not —
T33 kept Heston a named function precisely to avoid it), which would be a finding, not a refactor.

## 4. What this tranche is NOT (scope fence)
No new pricing, no new product, no new model — only the dispatch key and the registrations. Not a
plugin/extension system (§6 — no speculative generality); `(product, model)` is exactly the two
arguments §3 already names, no more. The hybrids (T34–T36) register into this cleanly; that is the
point of doing it first.

---

## 5. Cowork's ruling on T33b — PR #46 (§10 step 4, 2026-11-12)

Accepted — v0.40.0, 1142 green, nine gates, six mutants, zero suppressions, draft held for review. The
registry dispatches on `(product, model)`; Black and Heston coexist on the equity option, and the
clobber that would have deleted Black a tranche ago is now a loud failure.

**The premise fix was narrower than it looked — good.** The engines already annotated the
`IndexRatioModel` protocol, so `@register(InflationCapFloor, IndexRatioModel)` was already right; the
entire fix was in `dispatch`. My premise error cost a design ruling, not a mechanical rewrite.

**The bit-identity proof is the one the widening demanded.** The cap/floor resolved **through the new
protocol path** with a `JarrowYildirim` model returns `13957.888028738482`, identical to the direct
pricer — the key changed, no arithmetic did, demonstrated on the exact path the widening introduced
(not a path that never moved). That is the §4 pure-refactor bar met where it actually matters.
Registering Heston's put by **put-call parity** (inheriting the call's accuracy rather than a second
quadrature error) is the right call.

**The sweep's hole — it earns a §4 line (ninth appearance), and you were right to flag it.** Dropping
the `isinstance` check survived all 1140 tests because every model the suite hands a protocol-registered
pricer *satisfies* the protocol — the rejecting branch was never in the test population. This is a
**distinct** shape: not a value, slice, distribution, or duplicated guard, but the **input population**
being uniform in the one respect the guard discriminates on — the validation/type-guard case, where
test data is uniformly well-formed so input-validation guards go untested by default. The sub-trap you
found — an empty `@runtime_checkable Protocol` is satisfied by *everything*, so even a counter-fixture
can't discriminate unless the protocol carries a real member — is the sharp, reusable nugget. The
degenerate-fixture bullet is a catalog by design, so a genuinely new shape earns its entry; §4 edited.
Closed with a protocol carrying a real member and a non-conforming model — correct.

**The two smaller things — ratified.** Fixing the four `# noqa`s by their causes (the directory
`conftest.py` now owns the path setup *and* the registry-populating import, so "registry loaded" is a
directory precondition, not a per-test import) is §7 done right. The third CI stale-run catch via
`headSha` vs `HEAD` — "no run found" = "not yet created", never "nothing to wait for" — is the discipline
holding; keep it.

**Next:** T34 (the equity-rate hybrid — equity meets Hull-White, the first hybrid) is mine to spec
(§10 step 1), on disk before you build.
