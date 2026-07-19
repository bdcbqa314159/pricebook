"""Pricebook (new tree) — functional core, imperative shell.

Rebuilt clean under the **topic method** (redesign/13, /16): the earlier ng tree is
parked at ``ng_parked/`` as a *content* source (never a structural one) and this tree
is repopulated bottom-up, one topic at a time, starting with Topic 0 (the cross-asset
foundation, L0). Imports as ``pricebook_ng`` until v1.0. See CLAUDE.md for the guardrails.
"""

__version__ = "0.66.0"
