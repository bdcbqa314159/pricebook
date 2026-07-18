# Cowork → Build rulings — Topic 0, checkpoint after Slice 3

Reviews `CP_topic0_s3_checkpoint.md` (v0.58.0, 36 L0 oracles green, all gates incl. `layers` green).
**Checkpoint accepted with one correction.** Oracles are all published-standard — the strongest tier
available for conventions. Zero suppressions. Spine conformance verified module-by-module.

---

## §3.1 — the `observe` lift: REJECTED (spot-check found a real loss)

The claim was *"every calendar observes its fixed holidays uniformly, so the lift is
behaviour-preserving."* It is not. Scanning the quarry per calendar:

```
MIXED  AUDCalendar: 2 observed, 1 NOT  →  fixed(4, 25)
MIXED  NZDCalendar: 3 observed, 1 NOT  →  fixed(4, 25)
```

**25 April is ANZAC Day**, and in AU/NZ practice it is commemorated on the actual date and **not
Mondayised**, while New Year / Christmas / Boxing Day in those *same* calendars are. So the per-rule
`observe` flag encodes real market practice, and a calendar-level regime cannot express it: it would
either Mondayise ANZAC Day (wrong) or stop Mondayising the rest of the calendar (also wrong).

**Ruling — keep both levels:**
- **`Observance` regime stays at calendar level** — it is the right abstraction (US §6103 ·
  Commonwealth · Johannesburg) and the default for every rule.
- **Add a per-rule override** (`observed=False`, defaulting to the calendar's regime) for
  documented exceptions. Currently exactly two: AUD and NZD ANZAC Day.
- **Oracle:** ANZAC Day falling on a Saturday/Sunday must **not** shift, while Christmas in the same
  calendar and year **does**. Add that as a regression test.

*(This is why the claim had to be checkable. Good that it was stated plainly enough to verify.)*

## §3.2 — Furikae as a set-level `Observance`: ACCEPTED
Japanese substitution must walk forward past consecutive holidays, so it is inherently set-level.
Modelling it as a regime rather than a bespoke Tokyo class is the correct ADAPT.

## §3.3 — ACT/365L: CORRECTED to the ISDA definition
Your implementation (366 iff a 29 Feb lies in `[start, end)`) is the **annual-frequency rule applied
universally**. ISDA 2006 §4.16(i) is **frequency-dependent**:
- **frequency = annual** → 366 iff 29 February falls within the Calculation Period, else 365;
- **frequency > annual** → 366 iff the **period END date falls in a leap year**, else 365.

Adopt the ISDA form. `CouponPeriod` already carries `frequency`, so no signature change.
**Oracle:** a semi-annual period ending in a leap year but *not containing* 29 Feb must use 366 —
that case distinguishes the two definitions.

## §3.4 — `CouponPeriod` carrying `is_final` + anchors + `frequency`: ACCEPTED
It now serves **three** conventions (ICMA · 30E/360-ISDA · ACT/365L) — rule of two comfortably met,
and it keeps `year_fraction` at ≤5 args. The *name* is now narrower than the role; renaming to
something like `ConventionContext` is optional, not required.

## §3.5 — `year_fraction` 5-arg primitive + `Accrual.year_fraction` ergonomic: CONFIRMED
Correct split. Primitive at the ceiling; the ergonomic entry point bundles start+end+day_count in
Slice 4. No `Period` type now (one consumer = premature).

## §3.6 — `BDC.EOM`: your reading is RIGHT; my hand-off wording was imprecise
EOM is a **schedule/roll** property (ISDA §4.10 anchoring), not a date-adjustment convention.
`RollRule.eom` is the correct home. **No `BusinessDayConvention.EOM` value.** `NEAREST` correctly added.

## §3.7 — long/short stubs by construction: ACCEPTED
LONG merges the stub with its neighbour, SHORT keeps it. Explicit declaration beats the shed
`first_gap < months*30*0.5` heuristic and matches desk practice.

---

## §4 — EM calendars are secular-only: DEFER the data, but MARK them

Lunar/religious holidays (Islamic Hijri, Hebrew, Chinese lunisolar, Hindu, Thai Buddhist) are a
substantial data+algorithm undertaking, not needed for EUR/USD/GBP. **Defer to the EM-rates topic.**

**But do not inherit the silence.** A calendar that quietly omits Eid returns *wrong business days*,
which is silent wrongness — the thing the project forbids. **Add an explicit completeness marker**
(e.g. `Calendar.coverage: COMPLETE | SECULAR_ONLY`) on the eight affected calendars (Riyadh, Cairo,
Istanbul, Tel Aviv, Beijing, Seoul, Mumbai, Bangkok), surfaced in the module docs.
Forward-link: *"on crossing EM rates: add lunar/religious holiday data."*

---

## Ready for Slices 4–6
Proceed to **S4 money-quantity → S5 index-identity → S6 numerics-config**, then the **Topic 0 gate**
(park the set to `parked/topic-00-foundation/`, refresh the roll-up). Topic 1 does not begin before
that gate is green.

Carry into S4: `Accrual.year_fraction` as the ergonomic entry point (§3.5).
Carry into S2 rework: the ANZAC per-rule override and the ACT/365L correction — small, do them first.
