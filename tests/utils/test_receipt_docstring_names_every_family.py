"""The extraction-receipt docstring names every locator family the tuple registers.

Found stale 2026-09-23 by CORE (offered, not asked): `_extraction_receipt.__doc__` listed
'page' | 'slide' | 'sheet' | 'row' | 'none' while `_UNIT_LOCATOR_KEYS` held SIX families --
'section' (#92) and 'block' (#90) had been registered without the prose moving. A reader who
trusts the docstring over the tuple misses two families, and a consumer written from the
docstring enumerates four. The tuple decides; this pins the prose to it, so the next family
cannot leave the documentation behind silently.

Hermetic: imports the route module (conftest neutralises the vector store at import) and
reads two lines of one docstring. No I/O.
"""
import re

from app.routes.document_routes import _UNIT_LOCATOR_KEYS, _extraction_receipt


def _doc_line(prefix: str) -> str:
    lines = [
        line for line in (_extraction_receipt.__doc__ or "").splitlines()
        if line.strip().startswith(prefix)
    ]
    assert len(lines) == 1, (
        "expected exactly one `%s` line in _extraction_receipt's docstring, found %d"
        % (prefix, len(lines))
    )
    return lines[0]


def test_the_locator_kind_line_names_every_registered_family_then_none():
    """Order matters too: the docstring says 'in that order', and the tuple's order is the
    precedence the receipt applies, so the prose must not reorder it."""
    line = _doc_line("locator_kind:")
    listed = re.findall(r"'([a-z_]+)'", line)
    registered = [kind for kind, _ in _UNIT_LOCATOR_KEYS]
    assert listed == registered + ["none"], (
        "the receipt docstring's locator_kind line reads %r but _UNIT_LOCATOR_KEYS registers %r "
        "(plus 'none'). The tuple decides; update the prose in the same commit that changes the "
        "tuple." % (listed, registered)
    )


def test_the_empty_locators_clause_names_every_registered_family():
    """`empty_locators:` runs over two lines; the family words sit inside its parenthesis
    as `<kind> ints` or `<kind> names`."""
    doc = _extraction_receipt.__doc__ or ""
    m = re.search(r"empty_locators:\s*sorted locators \(([^)]*)\)", doc, re.S)
    assert m, "no `empty_locators: sorted locators (...)` clause in the docstring"
    clause = " ".join(m.group(1).split())
    for kind, _ in _UNIT_LOCATOR_KEYS:
        assert re.search(r"\b%s (ints|names)\b" % kind, clause), (
            "the empty_locators clause %r does not name the %r family" % (clause, kind)
        )
