"""DOCX is not the only format that folds to a `none` unit — and CSV is the trap.

FOLLOW-ON, flagged by the worker that wrote #43 in its own checkpoint: *"this item proves the rule
for DOCX only. MD / TXT / CSV also fold to a `none` unit and have no equivalent stored-chunk
assertion."* That is right about MD and TXT. It is a trap for CSV, and the trap is the reason this
file is shaped the way it is.

**CSV's answer depends on the base.** `#36` registers `("row", "row")` in `_UNIT_LOCATOR_KEYS`, so
on a tree that contains it a CSV chunk carries `row` and MUST. On `main`, which does not, a CSV
folds to `none` like the others. A test that hardcoded either answer would be correct on one base
and wrong on the other, and the wrong one would look like a product defect rather than a stale
fixture.

So CSV is asserted **against the module**: whatever `_UNIT_LOCATOR_KEYS` registers is what a CSV
chunk must carry. That is base-independent and it catches drift in **both** directions — a `row`
that stops being emitted after being registered, and a `row` that appears without being registered.

TXT carries no locator on either base, so it is asserted directly. MD USED to be in that
set; PACKET-1 E4 registers ("section", "section_index") and gives markdown a per-section
address, so MD is now asserted the same base-independent way as CSV.
"""

import pytest

from app.routes.document_routes import _UNIT_LOCATOR_KEYS, _prepare_documents_sync
from app.utils.document_loader import get_loader

MARKER = "NONE-FAMILY-MARKER-8821"
LOCATOR_KEYS = [key for _kind, key in _UNIT_LOCATOR_KEYS]


def _prepare(docs):
    return _prepare_documents_sync(
        docs, "none-family-file", "none-family-user", False, "ORGANIC",
        "doc", None, None, "none-family-tenant",
    )


def _load(path, filename, content_type):
    loader, _known, _ext = get_loader(filename, content_type, str(path))
    return list(loader.load())


def _locator_keys_present(metadata):
    """Read from the module, never from a copied list. A literal here would keep passing while
    the real tuple gained a key, which is precisely the drift this file exists to catch."""
    meta = metadata or {}
    return sorted(key for key in LOCATOR_KEYS if key in meta)


@pytest.fixture
def markdown_chunks(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Heading\n\n%s in a markdown body.\n\n## Second\n\nMore text.\n" % MARKER,
                    encoding="utf-8")
    return _prepare(_load(path, "notes.md", "text/markdown"))


@pytest.fixture
def text_chunks(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("%s in a plain text body.\n" % MARKER, encoding="utf-8")
    return _prepare(_load(path, "notes.txt", "text/plain"))


@pytest.fixture
def csv_chunks(tmp_path):
    path = tmp_path / "ledger.csv"
    path.write_text("region,revenue\nEMEA,4200000\nAMER,3100000\n", encoding="utf-8")
    return _prepare(_load(path, "ledger.csv", "text/csv"))


# ---------------------------------------------------------------------------------------
# Preconditions. Without these the assertions below can be true of a pipeline that produced
# nothing at all, which is the failure this lane keeps meeting.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["markdown_chunks", "text_chunks", "csv_chunks"])
def test_the_fixture_really_produced_chunks(fixture, request):
    chunks = request.getfixturevalue(fixture)
    assert chunks, "no chunks were produced, so every assertion about them is vacuous"


def test_the_marker_survived_so_the_content_really_parsed(markdown_chunks, text_chunks):
    """A loader that silently produced empty documents would satisfy the locator rule
    perfectly, and would satisfy it for the wrong reason."""
    for label, chunks in (("markdown", markdown_chunks), ("text", text_chunks)):
        blob = "\n".join(c.page_content for c in chunks)
        assert MARKER in blob, "%s: the content did not survive extraction" % label


# ---------------------------------------------------------------------------------------
# The rule.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("fixture,label", [("text_chunks", "TXT")])
def test_txt_carries_no_per_unit_locator(fixture, label, request):
    """TXT exposes no page, slide, sheet, row, section or block, so a citation into it can name the
    document and quote the text and nothing more. Emitting a locator here would be inventing a
    position that does not exist. (MD is no longer in this set: PACKET-1 E4 gives markdown a
    per-section address -- see test_md_carries_section_locator_when_registered below.)"""
    chunks = request.getfixturevalue(fixture)
    offenders = [
        (i, _locator_keys_present(c.metadata))
        for i, c in enumerate(chunks)
        if _locator_keys_present(c.metadata)
    ]
    assert not offenders, (
        "%s must carry no per-unit locator key, but chunks %s did. Registered locator keys: %s"
        % (label, offenders, LOCATOR_KEYS)
    )


def test_md_carries_section_locator_when_registered(markdown_chunks):
    """PACKET-1 E4 registers ("section", "section_index"); on a tree containing it every
    markdown chunk UNDER A HEADING must carry it, so a citation into that section has a
    position to open to. Asserted against the module, never a literal, so it catches drift in
    BOTH directions -- a `section_index` that stops being emitted after being registered, and
    one that appears without being registered -- exactly like the CSV trap above.

    The fixture has no preamble, so every chunk is a heading section. A preamble chunk (no
    heading above it) legitimately carries no address; that case is pinned in
    tests/utils/test_md_heading_locator.py, not here.
    """
    registered = "section_index" in LOCATOR_KEYS
    present = ["section_index" in (c.metadata or {}) for c in markdown_chunks]
    if registered:
        assert all(present), (
            "`section_index` is registered in _UNIT_LOCATOR_KEYS but %d of %d markdown chunks "
            "do not carry it -- a citation into those sections has no position to open to"
            % (present.count(False), len(present))
        )
    else:
        assert not any(present), (
            "markdown chunks carry `section_index` but _UNIT_LOCATOR_KEYS does not register it, "
            "so the receipt will say locator_kind=none while the chunks say otherwise"
        )


def test_csv_agrees_with_whatever_the_module_registers(csv_chunks):
    """THE TRAP, and why this is not a hardcoded assertion.

    `#36` registers `("row", "row")`. On a tree containing it a CSV chunk MUST carry `row`; on
    `main`, which does not, a CSV folds to `none` like MD and TXT. Hardcoding either answer makes
    this file correct on one base and wrong on the other — and the wrong one reads as a product
    defect rather than a stale fixture.

    Asserting against the module catches drift BOTH ways: a `row` that stops being emitted after
    being registered, and a `row` that appears without being registered.
    """
    registered = "row" in LOCATOR_KEYS
    present = ["row" in (c.metadata or {}) for c in csv_chunks]

    if registered:
        assert all(present), (
            "`row` is registered in _UNIT_LOCATOR_KEYS but %d of %d CSV chunks do not carry it — "
            "a citation into those rows has no position to open to"
            % (present.count(False), len(present))
        )
    else:
        assert not any(present), (
            "CSV chunks carry `row` but _UNIT_LOCATOR_KEYS does not register it, so the receipt "
            "will report locator_kind=none while the chunks say otherwise — the two halves of "
            "the contract disagree"
        )


def test_the_locator_check_can_actually_fail():
    """Guards the guard. If `_locator_keys_present` returned nothing for everything, every
    assertion above would pass against any implementation at all."""
    assert LOCATOR_KEYS, "_UNIT_LOCATOR_KEYS is empty; this whole file is vacuous"
    for key in LOCATOR_KEYS:
        assert _locator_keys_present({key: "x"}) == [key], (
            "the locator check did not see a stamped %r" % key
        )
    assert _locator_keys_present({"not_a_locator": 1}) == []
