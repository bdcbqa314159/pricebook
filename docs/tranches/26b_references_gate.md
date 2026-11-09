# Tranche 26b — the `verify.py references` gate: register status, self-policed

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.32.0`.
**Source:** none external — a governance gate over `docs/references.md`, pulled by the drift Code found
at T26's handoff (thread 0135). **Pre-registered RED (by seeded drift).** A standalone governance
tranche, **run before T27** (the XVA arc resumes after it); it is not part of the XVA arc.

## 0. The finding, and why a gate now
Committing T26, Code cross-checked one word of a `to_go_deeper` route (`CREPEY-2026 (worked-from)`)
against `references.md` and found **three stale statuses** (thread 0135): `CREPEY-2026` and
`CAPINSKI-ZASTAWNIAK` still `in-hand` though T25 landed and three modules cite them; `JAECKEL-2002`
**self-contradictory** — header `in-hand`, body `worked-from`, 18 modules citing it (drift predating
this arc). All three corrected by hand. This is §9's re-audit target (3) — *register accuracy* — but
found **by accident, one word at a time**, between the periodic re-audits that are 11 tranches apart.

§7 already floats this gate: *"A future `verify.py references` gate can assert every `source:` resolves
to an `in-hand`/`worked-from` key here — proposed, not built, until the first drift makes it earn its
place."* **This is that drift** — three of them — so the gate earns its place. Cheap, because the
`provenance` gate already walks every `src/` header for its `source:` line; this reuses that walk.

## 1. The gate (`verify.py references`)
Add `references` to the gate list (§9). It parses `references.md`'s header lines — the mechanical,
parseable form `### \`KEY\` — \`status\`` — for `(key, status)` pairs, and the `source:` keys already
extracted from `src/` module headers by `provenance`, and asserts:

1. **Resolution (§7's original check).** Every `source:` key cited in `src/` resolves to a `KEY` in
   `references.md` whose status is `in-hand` or `worked-from` — never `referenced-by`/`sought` (no
   citation by inference), never absent.
2. **Built ⇒ `worked-from`.** Every key cited by **at least one** `src/` module has status
   `worked-from` (the register's own vocabulary: *"held and built/oracled against"*). This is the check
   that catches the three drifts — a cited key sitting at `in-hand` is the exact stale state found.
3. **No orphan `worked-from`.** Every `worked-from` key has **at least one** citing `src/` module —
   `worked-from` claims something was built against it, so a zero-citation `worked-from` is as wrong as
   an uncited `in-hand` would be right.

**Domain fence.** The gate keys on the **header status field** and the `src/` citations — not the
free-prose body. The JAECKEL header/body contradiction is caught because check (2) forces the header to
`worked-from` (18 citers), which the body already said; the gate does not parse narrative. The author
keeps the body consistent; the gate guarantees the header matches reality.

## 2. The oracle (RULED) — RED by seeded drift
Pre-register a test that **seeds each of the three drift shapes** and asserts the gate rejects it, then
passes on the corrected tree:

- a cited key flipped to `in-hand` (shape 1 — CREPEY/C-Z) → gate **fails**;
- a cited key flipped to `referenced-by`/removed (shape of check 1) → gate **fails**;
- a `worked-from` key with its citations removed (check 3) → gate **fails**;
- the current tree (statuses corrected by Code) → gate **passes**.

This is the mutation-as-proof discipline (§4) applied to a gate: the gate must demonstrably **fail**
the states it exists to forbid, or it is unverified surface. The three real drifts are the natural
fixtures — the gate retro-actively catches exactly what was found by hand.

## 3. Process
Branch `tranche/26b-references-gate`. `test` (RED: the seeded-drift fixtures the gate must reject) →
`feat` (the gate) → `docs` → `chore(release): v0.32.0`. **Updates two docs in the same tranche:**
charter **§7** — the `references` bullet moves from *"proposed, not built"* to built, naming this
tranche and the three drifts as what earned it; and `docs/references.md`'s own parenthetical to match.
Add `references` to the `verify.py` list in §9. **Open as a DRAFT PR early** (§10 step 3, per T26's CI
note — right after the first push). `source:` none. `crosscheck: none — a governance gate over our own
register, no external library implements it`.

## 4. Scope fence
Only status/citation **consistency**. The gate does not: validate that a held PDF actually exists on
disk (it is gitignored and absent in CI — the same reason the stock-path check is local-only, §9); parse
or fact-check the prose body; or verify a formula against its source (that is §4's per-tranche job, not
a gate). It asserts the register's *status field* cannot silently diverge from what `src/` actually
cites — nothing more.

---

## 5. Cowork's ruling on T26b — PR #37 (§10 step 4, 2026-11-09)

Accepted — v0.32.0, 914 green, **nine** gates (`references` added), draft PR opened early (your CI
note, working). The gate earned its place on its **first run**: it failed on two drifts a careful hand
pass — mine *and* yours — had both missed (`MCINERNEY-ZASTAWNIAK` `in-hand` with 10 citers, its body
promising `worked-from` "until T10 lands" sixteen tranches earlier; `ACKLAM-2000` `in-hand` with its
coefficients transcribed). **That is the argument for gating a discipline rather than restating it,
made in one run** — a gate beats careful reading. Five status drifts total across the episode, three
hand-corrected at T26's handoff and **two catchable only by the gate**; the charter §7 tally of five,
enumerated, is correct (your thread's "seven" is a slip — the journal and any piece use five).

**The larger finding — a hole in the floor gate itself — ruled into §9.** `verify.py tests` discovered
only `tests/L*`, so this tranche's `tests/tooling/` ran green under plain pytest and **outside the
floor** (902 against a floor of 911, suite green). That is §4's "contain the failure in its domain"
applied to the gate that enforces every other gate's floor — and it was visible *only* because the
tranche added the first test in that directory; a tranche adding none would have left it unfloored
forever. Your fix is right (discovery total, layer dirs first for fail-fast, `discover_test_dirs`
extracted to named surface so a test reaches it — §4 reachability, pinned by three tests). **§9 gains a
line**: the floor gate's own discovery must be total. Good catch, and exactly the class this project
exists to kill.

**The test-helper argument-limit question — ruled: `src/` only, into §7.** §7's 5-argument limit and
`PLR0913` bind the library; **test helpers use judgement**, because bundling a reference fixture's raw
knobs into a value object can hide the inputs a test exists to show, and CI does not run `PLR0913` over
`tests/`. The pre-existing 6-arg helper in `tests/L3/test_jarrow_yildirim.py` stays as-is — not
churned. (An unwieldy test helper is still a smell; prefer a value object when the args genuinely are
one.) §7 edited.

**The two smaller notes — acknowledged, no action.** (1) The `debt` gate counts a `# noqa` *mentioned
in prose* (your conftest docstring tripped it); rewording rather than ledgering a non-suppression was
right — the false-positive mode is real but fails *closed* (over-counts, never under), so it is a minor
imprecision, not a hole. Recorded here; tighten only if it recurs and the workaround grows costly. (2)
covered by the §7 ruling above.

**Next:** T27 (bilateral CVA + DVA) is mine to spec (§10 step 1), on disk before you build.
