"""The risk layer (L5) — sensitivities computed on the L4 engine.

Empty from tranche 1 until tranche 22. The charter's layer box has carried `L5 RISK & CAPITAL — on
the engine + a Priceable protocol` since the spine was drawn, and Greeks by simulation are the first
thing that needed it: a sensitivity is not a price, so it does not belong at L4, and it is not data,
so it does not belong at L2.

**No `Priceable` protocol yet, and that is deliberate.** The tranche-22 spec left it a build question
("settle by what it reads, raise if unobvious"). What this module reads is Black's closed form and the
Monte Carlo machinery directly — one shape of consumer, so a protocol would be an abstraction with a
single implementation, which §6 forbids. It arrives when a second kind of priceable needs
differentiating.
"""
