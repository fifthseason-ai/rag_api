"""Markdown heading-hierarchy locators (PACKET-1 E4).

Every markdown section (the span introduced by a heading) must carry a durable machine
ADDRESS -- a 0-indexed `section_index` in document order -- so a citation can be reopened
at the right place. Four properties are the substance of the design ruling and are pinned
here, not decoration:

  1. The address locates; the readable path does not. Consumers may display a heading path
     but must never depend on it to locate the source.
  2. DUPLICATE headings still resolve through the address: two sections with the same
     heading TEXT carry DIFFERENT addresses and each resolves to its own content.
  3. RENAMING a heading moves the display path WITHOUT moving the address (the address is
     positional; see the LIMIT below on what DOES move it).
  4. Content with no heading above it (preamble) carries NO address and NO path -- absence
     means UNKNOWN, never "no headings"; a locator is never fabricated for it.

Plus: the structured path must lose no text the plain (single-mode) path captures on the
SAME input; and the /embed receipt's `locator_kind` must be `section` and AGREE with the
locator the stored chunks actually carry.

LIMIT (positional addressing, stated so a reader need not discover it): the address is a
position in document order, so INSERTING or REMOVING a preceding section shifts every later
section's index. That is inherent to positional addressing and consistent with the DOCX
`block_index` family; a rename in place (this suite's property 3) does not shift it.

LIMIT (empty headings): an ATX `#` with no text (or `## `) is not emitted as a `Title` by
unstructured, so its body gets NO address and folds into the preceding unit. Content is
preserved, but a titleless heading is not citable.

INDEPENDENT ANCHOR: the expected wire keys `section_index` / `heading_path` are hardcoded
here on purpose -- the same deliberate independence the receipt-agreement anchor uses -- so
this suite pins the CONTRACT the loader must satisfy, not whatever the module happens to say.

`heading_path` is a PROPOSED, PENDING-Core display field (agreement criterion (d)); this
suite proves it is emitted and is display-only, but it is NOT an agreed contract key.

Store double: the /embed test drives the REAL route and records AsyncPgVector.aadd_documents
so the assertions see the ACTUAL stored chunk metadata, not the loader's pre-persist output.
Every fixture is SYNTHETIC and labelled so.
"""

import datetime
import io
import os
import re
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_community.document_loaders import UnstructuredMarkdownLoader

from main import app
from app.utils.document_loader import get_loader
from app.services.vector_store.async_pg_vector import AsyncPgVector

# Wire keys pinned independently of the module (see docstring).
_ADDRESS_KEY = "section_index"
_PATH_KEY = "heading_path"
_SECRET = "testsecret"

# ---------------------------------------------------------------------------
# SYNTHETIC fixture: a nested markdown doc with a preamble and a DUPLICATE heading.
# Each section body carries a unique marker so "resolves to its own content" is testable.
# ---------------------------------------------------------------------------

_PREAMBLE_MARK = "SYN-E4-PREAMBLE"
_MARK = {
    0: "SYN-E4-SEC0",  # H1 Overview
    1: "SYN-E4-SEC1",  # H2 Scope
    2: "SYN-E4-SEC2",  # H1 Results
    3: "SYN-E4-SEC3",  # H2 Overview  (DUPLICATE heading text of section 0)
    4: "SYN-E4-SEC4",  # H3 Details
}

# section_index -> expected heading path (readable, display-only)
_EXPECTED_PATH = {
    0: "Overview",
    1: "Overview > Scope",
    2: "Results",
    3: "Results > Overview",
    4: "Results > Overview > Details",
}


def _synthetic_md(details_heading="Details", results_heading="Results"):
    """SYNTHETIC nested markdown. `results_heading`/`details_heading` let a test RENAME a
    heading in place without changing the number or order of sections."""
    return (
        "%s: text before any heading.\n\n"
        "# Overview\n\n%s intro under overview.\n\n"
        "## Scope\n\n%s scope body.\n\n"
        "# %s\n\n%s results body.\n\n"
        "## Overview\n\n%s duplicate-overview body.\n\n"
        "### %s\n\n%s deep body.\n"
        % (
            _PREAMBLE_MARK,
            _MARK[0],
            _MARK[1],
            results_heading, _MARK[2],
            _MARK[3],
            details_heading, _MARK[4],
        )
    )


def _write(tmp_path, text, name="syn_e4.md"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _load_sections(path):
    """Load through the REAL dispatch (get_loader) and return the emitted Documents."""
    loader, known, ext = get_loader("syn_e4.md", "text/markdown", str(path))
    assert known is True and ext == "md"
    return list(loader.load())


def _section_of(docs, mark):
    """The single Document whose content carries `mark`."""
    hits = [d for d in docs if mark in (d.page_content or "")]
    assert len(hits) == 1, "expected exactly one section carrying %r, got %d" % (mark, len(hits))
    return hits[0]


# ---------------------------------------------------------------------------
# Precondition: the fixture really is nested and really has a duplicate heading.
# Without this the property tests could pass against a degenerate fixture.
# ---------------------------------------------------------------------------


def test_fixture_is_nested_and_has_a_duplicate_heading():
    text = _synthetic_md()
    lines = text.splitlines()
    assert "# Overview" in lines        # H1 Overview
    assert "## Overview" in lines       # H2 Overview -> SAME text, different section
    assert "### Details" in lines       # three-deep nesting exists
    # the heading TEXT "Overview" appears at two different levels -> a genuine duplicate
    assert sum(1 for ln in lines if ln.lstrip("#").strip() == "Overview"
               and ln.lstrip().startswith("#")) == 2
    # all five section markers and the preamble marker are distinct and present
    marks = [_PREAMBLE_MARK] + [_MARK[i] for i in range(5)]
    assert len(set(marks)) == len(marks)
    for m in marks:
        assert m in text


# ---------------------------------------------------------------------------
# Property 4 (preamble) + every-section-addressed.
# ---------------------------------------------------------------------------


def test_every_section_carries_its_zero_indexed_address(tmp_path):
    docs = _load_sections(_write(tmp_path, _synthetic_md()))
    for idx, mark in _MARK.items():
        sec = _section_of(docs, mark)
        assert sec.metadata.get(_ADDRESS_KEY) == idx, (
            "section carrying %r must have %s == %d, got %r"
            % (mark, _ADDRESS_KEY, idx, sec.metadata.get(_ADDRESS_KEY)))
    # addresses are 0-indexed and contiguous in document order
    addresses = [d.metadata.get(_ADDRESS_KEY) for d in docs
                 if d.metadata.get(_ADDRESS_KEY) is not None]
    assert addresses == sorted(addresses)
    assert min(addresses) == 0


def test_preamble_carries_no_address_and_no_path(tmp_path):
    docs = _load_sections(_write(tmp_path, _synthetic_md()))
    pre = _section_of(docs, _PREAMBLE_MARK)
    assert _ADDRESS_KEY not in pre.metadata, (
        "preamble before the first heading must carry NO address (UNKNOWN, not fabricated)")
    assert _PATH_KEY not in pre.metadata, (
        "preamble before the first heading must carry NO heading path")


def test_document_with_no_headings_at_all_carries_no_address(tmp_path):
    docs = _load_sections(_write(tmp_path, "Just prose. No headings here at all.\n"))
    assert docs, "a heading-less markdown file must still extract its text"
    for d in docs:
        assert _ADDRESS_KEY not in d.metadata, "no heading -> no address (never fabricated)"
        assert _PATH_KEY not in d.metadata


# ---------------------------------------------------------------------------
# Property 1: the address is a machine value, not display text.
# ---------------------------------------------------------------------------


def test_address_is_a_machine_value_never_display_text(tmp_path):
    docs = _load_sections(_write(tmp_path, _synthetic_md()))
    for d in docs:
        addr = d.metadata.get(_ADDRESS_KEY)
        if addr is None:
            continue
        assert isinstance(addr, int), "address must be an integer, not display text"
        # the readable separator must never leak into the address
        assert ">" not in str(addr)


# ---------------------------------------------------------------------------
# Property 2: duplicate headings resolve through the address, not the text.
# ---------------------------------------------------------------------------


def test_duplicate_headings_resolve_through_the_address(tmp_path):
    # The fixture has two sections with the SAME heading text "Overview" (pinned by
    # test_fixture_is_nested_and_has_a_duplicate_heading). This test stays purely about the
    # ADDRESS: no assertion here reads heading_path, so removing that PROPOSED display field
    # can never redden an address-named test. (The path side is asserted in
    # test_proposed_heading_path_matches_the_hierarchy.)
    docs = _load_sections(_write(tmp_path, _synthetic_md()))
    first = _section_of(docs, _MARK[0])   # H1 Overview
    dup = _section_of(docs, _MARK[3])     # H2 Overview -- SAME heading text

    # DIFFERENT addresses ...
    assert first.metadata.get(_ADDRESS_KEY) != dup.metadata.get(_ADDRESS_KEY)
    # ... and each address resolves to its OWN content, not the other's.
    assert _MARK[0] in first.page_content and _MARK[3] not in first.page_content
    assert _MARK[3] in dup.page_content and _MARK[0] not in dup.page_content


# ---------------------------------------------------------------------------
# Property 3: rename moves the display path, not the address.
# ---------------------------------------------------------------------------


def test_rename_moves_path_but_not_address(tmp_path):
    before = _load_sections(_write(tmp_path, _synthetic_md(results_heading="Results"),
                                   name="before.md"))
    after = _load_sections(_write(tmp_path, _synthetic_md(results_heading="Findings"),
                                  name="after.md"))

    # The renamed section is identified by its STABLE body marker, not its heading text.
    b = _section_of(before, _MARK[2])
    a = _section_of(after, _MARK[2])

    # The durable source reference is UNCHANGED by the rename ...
    assert a.metadata.get(_ADDRESS_KEY) == b.metadata.get(_ADDRESS_KEY), (
        "renaming a heading must NOT move the section's address")
    # ... while the readable display path follows the new heading text.
    assert b.metadata.get(_PATH_KEY) == "Results"
    assert a.metadata.get(_PATH_KEY) == "Findings"
    assert a.metadata.get(_PATH_KEY) != b.metadata.get(_PATH_KEY)

    # A child of the renamed heading also keeps its address while its path updates.
    b_child = _section_of(before, _MARK[3])
    a_child = _section_of(after, _MARK[3])
    assert a_child.metadata.get(_ADDRESS_KEY) == b_child.metadata.get(_ADDRESS_KEY)
    assert b_child.metadata.get(_PATH_KEY) == "Results > Overview"
    assert a_child.metadata.get(_PATH_KEY) == "Findings > Overview"


# ---------------------------------------------------------------------------
# The PROPOSED, PENDING-Core display field: present, readable, display-only.
# ---------------------------------------------------------------------------


def test_proposed_heading_path_matches_the_hierarchy(tmp_path):
    docs = _load_sections(_write(tmp_path, _synthetic_md()))
    for idx, expected in _EXPECTED_PATH.items():
        sec = _section_of(docs, _MARK[idx])
        assert sec.metadata.get(_PATH_KEY) == expected, (
            "section %d display path %r != expected %r"
            % (idx, sec.metadata.get(_PATH_KEY), expected))
    # The two DUPLICATE-heading sections both display a path ENDING in the same text
    # "Overview" yet the paths differ by their parent -- the display side of the property
    # whose ADDRESS side lives in test_duplicate_headings_resolve_through_the_address. Kept
    # here (not there) so a decline of this PROPOSED field never reddens an address test.
    first = _section_of(docs, _MARK[0])   # H1 Overview
    dup = _section_of(docs, _MARK[3])     # H2 Overview -- SAME heading text
    assert first.metadata.get(_PATH_KEY, "").endswith("Overview")
    assert dup.metadata.get(_PATH_KEY, "").endswith("Overview")
    assert first.metadata.get(_PATH_KEY) != dup.metadata.get(_PATH_KEY)


# ---------------------------------------------------------------------------
# No-loss: the structured path loses no text the plain (single-mode) path captures.
# ---------------------------------------------------------------------------


def _words(text):
    return set(w for w in re.findall(r"[A-Za-z0-9\-]+", text.lower()) if w)


def test_structured_path_loses_no_text_vs_plain_path(tmp_path):
    path = _write(tmp_path, _synthetic_md())

    plain = UnstructuredMarkdownLoader(str(path), mode="single").load()
    plain_text = "\n".join(d.page_content or "" for d in plain)

    structured = _load_sections(path)
    structured_text = "\n".join(d.page_content or "" for d in structured)

    missing = _words(plain_text) - _words(structured_text)
    assert not missing, (
        "structured markdown path dropped text the plain path captured: %s" % sorted(missing))
    # positive control: the plain path really did capture content (else the test is vacuous)
    assert _PREAMBLE_MARK.lower() in _words(plain_text)
    assert _MARK[4].lower() in _words(plain_text)


# ---------------------------------------------------------------------------
# Receipt <-> stored-chunk agreement through the REAL /embed route.
# ---------------------------------------------------------------------------


def _hdr():
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": "testuser", "tid": "tenantA", "ent": ["userA"], "act": ["write"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


@pytest.fixture(scope="module")
def embedded_md():
    os.environ["JWT_SECRET"] = _SECRET
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    added = []

    async def recording_aadd(self, docs, ids=None, executor=None):
        added.append(list(docs))
        return ids

    async def dummy_delete(self, ids=None, collection_only=False, user_id=None,
                           document_origin_type=None, subscription_id=None, executor=None, **_):
        return None

    orig_add = AsyncPgVector.aadd_documents
    orig_del = AsyncPgVector.delete
    AsyncPgVector.aadd_documents = recording_aadd
    AsyncPgVector.delete = dummy_delete
    try:
        client = TestClient(app)
        body = _synthetic_md().encode("utf-8")
        r = client.post(
            "/embed",
            data={"file_id": "f-e4", "entity_id": "userA"},
            files={"file": ("syn_e4.md", io.BytesIO(body), "text/markdown")},
            headers=_hdr(),
        )
        assert r.status_code == 200, r.text
        receipt = r.json()["extraction"]
        stored = [dict(d.metadata or {}) for batch in added for d in batch]
        stored_text = "\n".join(
            getattr(d, "page_content", "") or "" for batch in added for d in batch)
    finally:
        AsyncPgVector.aadd_documents = orig_add
        AsyncPgVector.delete = orig_del
    return {"receipt": receipt, "stored": stored, "text": stored_text}


def test_embed_stored_something_and_content_survived(embedded_md):
    assert embedded_md["stored"], "no chunks stored -> every locator assertion is vacuous"
    for mark in [_PREAMBLE_MARK] + [_MARK[i] for i in range(5)]:
        assert mark in embedded_md["text"], "content token %r did not survive extraction" % mark


def test_embed_receipt_locator_kind_is_section(embedded_md):
    assert embedded_md["receipt"]["locator_kind"] == "section", (
        "markdown receipt must report locator_kind=section, got %r"
        % embedded_md["receipt"]["locator_kind"])


def test_embed_receipt_agrees_with_stored_chunks(embedded_md):
    """The rule (F-RECEIPT-AGREE) for markdown: the receipt names `section` iff a stored
    chunk carries `section_index`, and the stored chunks really do carry it."""
    stored = embedded_md["stored"]
    carries_section = any(m.get(_ADDRESS_KEY) is not None for m in stored)
    assert carries_section, "stored markdown chunks must carry the section address"
    assert (embedded_md["receipt"]["locator_kind"] == "section") == carries_section


def test_embed_preamble_chunk_stored_without_an_address(embedded_md):
    """The negative: the preamble chunk is present in the store and did NOT acquire an
    address. Proves the family is really stamped per-unit, not blanket-applied."""
    stored = embedded_md["stored"]
    # The preamble chunk is the one carrying no address (matched on metadata, since the
    # recorded batches expose metadata; its text presence is covered by the content test).
    pre_chunks = [m for m in stored if m.get(_ADDRESS_KEY) is None]
    assert pre_chunks, "the preamble chunk (no address) must be present in the store"
    for m in pre_chunks:
        assert _ADDRESS_KEY not in m or m.get(_ADDRESS_KEY) is None
        assert _PATH_KEY not in m
