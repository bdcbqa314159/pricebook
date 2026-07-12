# Claude Code directive — slice/05 signature bundling (refactor)

Paste into the Claude Code session at the repo root. This is a **pure refactor** guarded
by already-green oracles: behaviour must not change, so no new RED is required, but the
existing oracles MUST be run and stay green (branching policy §carve-out).

---

```
Read CLAUDE.md §3b (signature discipline), §6b (simplicity), and §3 (vocabulary). We are
enforcing max 5 arguments in src/pricebook_ng/** and fixing the three current offenders by
BUNDLING cohesive parameters into frozen value objects — never by suppression.

Work on branch slice/05-signature-bundling off main.

1. feat(foundation): introduce frozen value objects at L0:
   - RollRule(calendar: Calendar | None, convention: BusinessDayConvention, eom: bool) with a
     module-level DEFAULT_ROLL. Consumers: generate_schedule + every scheduled instrument.
   - CouponPeriod(ref_start: date, ref_end: date, frequency: int). Consumers: year_fraction
     (ACT/ACT ICMA) + coupon legs.
   - ScheduleTerms(start: date, maturity: date, frequency: Frequency, roll: RollRule).
     Consumers: every scheduled instrument (bond now, swap later).
   Each has ≥2 real consumers (rule of two) — no speculative fields.

2. refactor: apply them, targeting these signatures (all land ≤ 5 args):
   - year_fraction(start, end, convention, *, period: CouponPeriod | None = None,
                   calendar: Calendar | None = None)                      # 7 → 5
   - generate_schedule(start, end, frequency, roll: RollRule = DEFAULT_ROLL)   # 6 → 4
   - fixed_rate_bond(terms: ScheduleTerms, notional: Money, coupon_rate, day_count)  # 9 → 4
     (notional+currency collapse into the ratified Money type — do not keep a separate
     currency param.)
   Update ALL call sites and tests accordingly.

3. chore(lint): enable ruff PLR0913 with max-args = 5, scoped to src/pricebook_ng/** only;
   the quarry python/pricebook/** stays excluded. Wire it into the CI ruff step.

4. Run the FULL existing oracle suite for the touched layers (verify.py tests --layer 2) plus
   verify.py acyclic / debt / provenance / version and the ruff pass. Everything green; behaviour
   byte-identical (these are the same computations, re-grouped). Do NOT add or weaken any oracle.

5. docs: provenance unchanged (same sources); CHANGELOG entry under Changed; bump __version__
   (PATCH — refactor, no capability change). chore(release) as the tip commit.

Land with rebase-and-merge only when green on both OSes. If any signature elsewhere in
src/pricebook_ng/** also exceeds 5, fix it the same way in this slice; if one is a genuinely
irreducible closed-form math signature, log it in OPEN.md with a rationale rather than
suppressing — do not use # noqa.
```

---

After this slice, PLR0913 guards every future slice: a large signature fails CI, forcing the
bundle-into-a-value-object fix at write time. The problem is corrected and prevented.
