# Artifact #08 — Build → Design Handoff Protocol (DRAFT)

**Status:** Draft for reaction. Defines the report Claude Code (build) produces when a
layer's slices land, so this design workspace (Cowork) can update the ledger, generate
the next layer, and catch any design drift — without re-deriving state.

---

## The round-trip

```
Cowork:  rule the L<N> ledger  ─────────────►  Claude Code: build L<N> slices, bottom-up
                                                            each slice: audit→align→red→green→docs
Cowork:  update ledger, gen L<N+1>  ◄────────  Claude Code: LAYER COMPLETION REPORT
```

Each return trip is one **Layer Completion Report** — a single markdown file Claude Code
writes to `redesign/handoffs/L<N>_report.md`, plus a one-line message you paste back here.

---

## What Claude Code produces (fill-in template)

Claude Code writes this when every slice in a layer has landed green:

```markdown
# Layer Completion Report — L<N> <layer name>
Date: <date>   ·   Version: <__version__ now>   ·   Branch(es): slice/<NN>-… (landed)

## 1. Slices landed
| Slice | Title | Version | Oracle | Tolerance met |
|-------|-------|---------|--------|---------------|
| S<NN> | …     | 0.x.y   | closed-form / QuantLib / self-consistency | <e.g. 3e-13> |

## 2. Ledger deltas (what happened to each entry)
| Quarry entry | Disposition | New-tree path | Notes |
|--------------|-------------|---------------|-------|
| core/day_count.py | migrated  | foundation/time/day_count.py | Money at boundary |
| core/book.py      | re-homed  | shell/booking/book.py (L6)   | deferred to L6 slice |
| core/xxx.py       | archived  | .archive/…                   | zero consumers |
| …                 | deferred  | — (still in quarry)          | needs L<M> first |

## 3. Oracles used
One line per slice: the reference value, the method, and the achieved error.

## 4. Debt logged this layer  (mirror of new OPEN.md entries)
- <what> — <why> — <re-open trigger> — <owning slice>

## 5. Design drift  ← the important one
Anything that diverged from the spine / vocabulary / CLAUDE.md, and why. For each:
- what the design said, what the build needed, the proposed artifact/CLAUDE.md change.
(If none: "No drift — design held.")

## 6. Quarry status
- L<N> entries remaining: <count>  (0 = layer complete)
- Total quarry remaining: <count> / <original>   ·   % migrated: <x>%

## 7. Ready for next layer?
- L<N> complete: yes/no
- Next layer: L<N+1> <name>
- Prerequisites / blockers for it: <list, or "none">
- Questions for design to resolve before L<N+1>: <list, or "none">
```

## The one-line message you bring back here

Paste this into Cowork to resume:

```
L<N> complete at v0.x.y — <A> migrated, <B> re-homed, <C> archived, <D> deferred;
<E> debt items; drift: <none | see §5>; quarry <P>% empty; ready for L<N+1>.
See redesign/handoffs/L<N>_report.md.
```

That single line tells me: land the ledger deltas, whether any artifact needs amending
(drift), and to generate the L<N+1> ledger. I read the full report for detail.

---

## Why §5 (design drift) matters most

The build is where the design meets reality. An entry that "wouldn't align cleanly," an
oracle that had to be weaker than hoped, a vocabulary type that needed one more field —
these are the signals that an artifact (or `CLAUDE.md`) should change. Capturing them per
layer is how the design stays a living constitution instead of drifting silently out of
sync with the tree. No-drift layers are the norm; a drift entry is a prompt for us to
amend an artifact here before the next layer builds on it.

---

## CLAUDE.md hook
`CLAUDE.md` instructs the build agent: on completing a layer, produce this report at
`redesign/handoffs/L<N>_report.md` and stop for design review before starting the next
layer. Bottom-up is not just build order — it is a review checkpoint per layer.
