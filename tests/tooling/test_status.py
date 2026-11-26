"""The status instrument's own tests (T81). The `mutation_sweep.py` committed-instrument pattern.

**Why this file carries a known-answer test at all.** `status.py` exists to put *counts* on a page,
and re-audit #9 is the reason it is a tool rather than a hand-maintained table: five numbers filed
in that audit's own findings were wrong, every one read off a grep rather than derived (F-9.2). A
generator that miscounts is the same defect with a longer reach, because the page looks
authoritative. So the generator is tested against a **fixture tree whose answers are known by
construction**, and the page is **drift-gated** against the real tree, which is the part that makes
it unable to lie.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "tooling"))

from status import (  # noqa: E402  (the tooling dir is not a package)
    CURATED_END,
    CURATED_START,
    Inventory,
    mechanical_rows,
    render,
    survey,
)


# ============================================================================================
# 1. The known-answer test — a fixture tree whose counts are true by construction.
# ============================================================================================

def _fixture_tree(root: Path) -> None:
    """A miniature `src/pricebook` with answers known by construction.

    **Two asset classes and a greeks/no-greeks pair, deliberately** (the spec's degeneracy premise):
    a one-class tree makes the classifier trivially right, and a tree where everything has greeks
    cannot tell a working detector from one that returns `True`. The τ=1 shape, in a generator.
    """
    (root / "models").mkdir(parents=True)
    (root / "engine").mkdir()
    (root / "products").mkdir()
    (root / "risk").mkdir()
    (root / "market").mkdir()
    (root / "foundation").mkdir()

    for pkg in ("models", "engine", "products", "risk", "market", "foundation"):
        (root / pkg / "__init__.py").write_text("")

    # two models, two products, three engines -- two of them registered
    (root / "models" / "alpha_model.py").write_text("class AlphaParams: ...\n")
    (root / "models" / "beta_model.py").write_text("class BetaParams: ...\n")
    (root / "products" / "alpha_swap.py").write_text("class AlphaSwap: ...\n")
    (root / "products" / "beta_option.py").write_text("class BetaOption: ...\n")
    (root / "engine" / "alpha_swap.py").write_text(
        "@register(AlphaSwap, AlphaModel)\ndef price_alpha_swap(): ...\n"
    )
    (root / "engine" / "beta_option.py").write_text(
        "@register(BetaOption, BetaModel)\ndef price_beta_option(): ...\n"
    )
    (root / "engine" / "shared_helper.py").write_text("def helper(): ...\n")  # no @register
    # greeks for ONE class only, so the detector must discriminate
    (root / "risk" / "sensitivities.py").write_text(
        "def beta_mc_delta(): ...\ndef beta_mc_vega(): ...\n"
    )


def test_the_generator_counts_a_FIXTURE_tree_correctly(tmp_path):
    """Counts derived by the tool, against counts true by construction."""
    _fixture_tree(tmp_path / "pricebook")
    inventory = survey(tmp_path / "pricebook")

    assert inventory.models == 2, inventory
    assert inventory.products == 2, inventory
    assert inventory.engines == 3, inventory
    assert inventory.registry_pairs == 2, inventory


def test_the_generator_DISCRIMINATES_between_asset_classes_and_greeks(tmp_path):
    """The premise that makes the test above non-vacuous: the classifier does real work.

    A generator that put every engine in one bucket, or reported greeks for everything, would pass
    a single-class fixture. So the fixture has two classes and exactly one of them has greeks.
    """
    _fixture_tree(tmp_path / "pricebook")
    inventory = survey(tmp_path / "pricebook")

    assert set(inventory.engines_by_class) == {"alpha", "beta"}, inventory.engines_by_class
    assert inventory.engines_by_class["alpha"] == 1
    assert inventory.engines_by_class["beta"] == 1
    # greeks exist for beta only -- the discriminating assertion
    assert inventory.greeks_by_class == {"beta"}, inventory.greeks_by_class


def test_an_UNCLASSIFIED_engine_is_reported_not_silently_dropped(tmp_path):
    """A generator that drops what it cannot classify reports a smaller, cleaner, wrong tree.

    This is the applied-count discipline (§9) in a counting tool: an engine the classifier does not
    recognise must appear in `unclassified`, because the alternative is a page that looks complete.
    """
    _fixture_tree(tmp_path / "pricebook")
    (tmp_path / "pricebook" / "engine" / "zeta_widget.py").write_text(
        "@register(ZetaWidget, ZetaModel)\ndef price_zeta_widget(): ...\n"
    )
    inventory = survey(tmp_path / "pricebook")

    assert inventory.engines == 4
    assert "zeta_widget" in inventory.unclassified, inventory.unclassified
    # and the classified counts did NOT absorb it
    assert sum(inventory.engines_by_class.values()) == 2


# ============================================================================================
# 2. The drift gate — the anti-folklore guarantee.
# ============================================================================================

def test_the_committed_status_page_MATCHES_a_fresh_generation():
    """**The gate that makes the page unable to lie.** A stale `STATUS.md` fails here.

    The hand-maintained draft this replaces went stale **within hours** of being written (it said
    "T78 in review" after T78 merged), which is why §7's guarded-not-folklore line applies to a
    status page as much as to a convention. The curated rows between the markers are exempt — they
    are judgement, refreshed at each re-audit — and everything else is compared byte for byte.
    """
    page = (ROOT / "docs" / "STATUS.md").read_text()
    fresh = render(ROOT / "src" / "pricebook", page)
    assert page == fresh, (
        "docs/STATUS.md is stale — regenerate it with `python tests/tooling/status.py --write`"
    )


def test_the_drift_gate_FAILS_on_a_planted_divergence(tmp_path):
    """A gate asserted to work rather than assumed — F-9.1's discipline, applied to a new gate.

    §9's rule is that a gate which cannot fail is not a gate. So the divergence is planted and the
    comparison must redden: a committed page whose mechanical rows disagree with the tree is caught.
    """
    _fixture_tree(tmp_path / "pricebook")
    page = render(tmp_path / "pricebook", _seed_template())

    # the tree gains a model; the committed page does not know
    (tmp_path / "pricebook" / "models" / "gamma_model.py").write_text("class GammaParams: ...\n")
    assert render(tmp_path / "pricebook", _seed_template()) != page

    # and the real page's own comparison would redden the same way
    corrupted = page.replace("models", "moduls", 1)
    assert corrupted != render(tmp_path / "pricebook", _seed_template())


def test_the_CURATED_region_is_preserved_VERBATIM(tmp_path):
    """The judgement half survives regeneration untouched — that is the whole split.

    The mechanical rows are the tool's; the asset-class readiness, the bibliography and the 1.0
    scorecard are Cowork's, refreshed by hand at each re-audit. If regeneration rewrote them the
    tool would be destroying the only part it cannot derive.
    """
    _fixture_tree(tmp_path / "pricebook")
    curated = "## Hand-written\n\nA judgement the tool cannot derive: **●●● deep**.\n"
    template = f"{CURATED_START}\n{curated}{CURATED_END}\n"
    out = render(tmp_path / "pricebook", template)
    assert curated in out, out


# ============================================================================================
# 3. Idempotence.
# ============================================================================================

def test_generating_TWICE_is_byte_identical(tmp_path):
    """No timestamp churn: a page that changed on every run would make the drift gate meaningless."""
    _fixture_tree(tmp_path / "pricebook")
    first = render(tmp_path / "pricebook", _seed_template())
    second = render(tmp_path / "pricebook", first)
    assert first == second


def test_the_real_page_is_also_idempotent():
    page = (ROOT / "docs" / "STATUS.md").read_text()
    once = render(ROOT / "src" / "pricebook", page)
    assert once == render(ROOT / "src" / "pricebook", once)


def _seed_template() -> str:
    return f"# fixture\n\n{CURATED_START}\n## curated\n\nkept.\n{CURATED_END}\n"


# ============================================================================================
# 4. The IP constraint — structure and status only.
# ============================================================================================

def test_the_page_carries_NO_formula_or_source_content():
    """`STATUS.md` is tracked and may later render publicly, so it carries structure only.

    `_stock/` is gitignored because the sources are copyrighted; a status page that leaked a
    transcribed formula or a source line would undo that by the back door. The mechanical rows are
    counts and module names by construction — this asserts it rather than trusting it.
    """
    mechanical = "\n".join(mechanical_rows(ROOT / "src" / "pricebook"))
    for forbidden in ("math.", "def ", "return ", "\\frac", "eq (", "§"):
        assert forbidden not in mechanical, (forbidden, mechanical[:400])
