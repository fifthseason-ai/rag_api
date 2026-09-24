"""The extraction-receipt docstring names every locator family the tuple registers.

Found stale 2026-09-23 by CORE (offered, not asked): `_extraction_receipt.__doc__` listed
'page' | 'slide' | 'sheet' | 'row' | 'none' while `_UNIT_LOCATOR_KEYS` held SIX families --
'section' (#92) and 'block' (#90) had been registered without the prose moving. A reader who
trusts the docstring over the tuple misses two families, and a consumer written from the
docstring enumerates four. The tuple decides; this pins the prose to it, so the next family
cannot leave the documentation behind silently.

Hermetic: imports the route module (conftest neutralises the vector store at import), reads
two lines of one docstring, and reads the comment above the tuple from the module's own source
file (located through the imported module, so it is present wherever the module is). No
network, no database.
"""
import inspect
import re
from pathlib import Path

from app.routes import document_routes
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


def _comment_above_the_tuple() -> str:
    """The contiguous `#` block directly above `_UNIT_LOCATOR_KEYS = (` in the module source.

    Located through the imported module (`inspect.getsourcefile`), not a repo-relative path:
    the source ships in every image that can import it, so this reads the same file in a
    checkout, on the bare runner and inside both shipped images -- unlike a test that reads a
    file `.dockerignore` excludes.
    """
    lines = Path(inspect.getsourcefile(document_routes)).read_text(encoding="utf-8").splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("_UNIT_LOCATOR_KEYS = (")]
    assert len(starts) == 1, "expected one `_UNIT_LOCATOR_KEYS = (` line, found %d" % len(starts)
    block = []
    i = starts[0] - 1
    while i >= 0 and lines[i].startswith("#"):
        block.insert(0, lines[i])
        i -= 1
    assert block, "no comment block directly above _UNIT_LOCATOR_KEYS; this pin is reading nothing"
    return "\n".join(block)


def test_the_comment_above_the_tuple_names_every_family_key_and_no_count():
    """RV-106 N1: the tuple's OWN comment still said DOCX folds to 'none' and that new
    families append after a stated number of existing ones -- stale since #92 and #90 -- and
    the two pins above could not see it, because they read only the receipt docstring. This
    extends the same pin to the comment a maintainer reads first.

    Three checks: every registered METADATA KEY appears in backticks exactly once (the comment
    is about the keys loaders emit, so that is the form it names them in); they appear in
    TUPLE ORDER, because the comment says ORDER IS PRECEDENCE and a reordered list would misstate
    precedence to its reader (RV-115 N1); and the comment states NO count of families -- a count
    written in prose is precisely the part that went stale (RV-115 N2 widened the phrasings)."""
    block = _comment_above_the_tuple()
    positions = []
    for kind, key in _UNIT_LOCATOR_KEYS:
        n = block.count("`%s`" % key)
        assert n == 1, (
            "the comment above _UNIT_LOCATOR_KEYS names the %r family's key `%s` %d time(s); it "
            "must name it exactly once (absent = stale; twice = its position is ambiguous). The "
            "tuple decides; update the comment in the same commit that changes the tuple."
            % (kind, key, n)
        )
        positions.append(block.index("`%s`" % key))
    assert positions == sorted(positions), (
        "the comment names the family keys in a different order from _UNIT_LOCATOR_KEYS (%r). "
        "The order IS the detection precedence, so the prose must list them in tuple order."
        % [key for _p, key in sorted(zip(positions, (k for _kind, k in _UNIT_LOCATOR_KEYS)))]
    )
    number = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
    count = (re.search(r"\b(?:existing|current|registered) %s\b" % number, block, re.I)
             or re.search(r"\b%s\s+(?:\w+\s+)?famil(?:y|ies)\b" % number, block, re.I))
    assert count is None, (
        "the comment above _UNIT_LOCATOR_KEYS states a family count (%r). Counts in prose go "
        "stale the moment a family is appended -- that is how this comment went wrong the "
        "first time. Name the families; do not count them." % count.group(0)
    )
