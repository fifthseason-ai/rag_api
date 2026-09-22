"""A slide's TITLE leads its extracted text (KC-FILES-EXTRACT-REPAIR, synthetic decks only).

WHY. The 2026-09-13 extraction QA recorded a deck slide that "emits body before its
title/number" -- one of the out-of-order defects behind the 179 deferred derivative case
records. SlidePowerPointLoader walked the shape tree, which is Z-ORDER, not reading order:
a title placeholder brought to the front (or re-inserted after the body) came out AFTER the
body, so the stored chunk read "body ... title" and a reader could not tell which heading
the text belonged to. Reproduced on origin/main a4b47a6 with the deck built below:

    'BODY: eight opportunity vectors\\nTITLE: Opportunity vectors'

The fix puts the title first and leaves every other shape in its existing order. These
tests pin that, and pin that decks which were already right come out byte-identical.
"""

import datetime
import io
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient
from pptx import Presentation
from pptx.util import Inches

from app.utils.document_loader import get_loader

PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
TITLE = "TITLE: Opportunity vectors"
BODY = "BODY: eight opportunity vectors"
NOTE = "BODY TWO: a second box"


def _deck(path, *, title_last, with_title=True, notes=None):
    prs = Presentation()
    layout = prs.slide_layouts[5] if with_title else prs.slide_layouts[6]  # title-only / blank
    s = prs.slides.add_slide(layout)
    s.shapes.add_textbox(Inches(1), Inches(2), Inches(6), Inches(1)).text_frame.text = BODY
    s.shapes.add_textbox(Inches(1), Inches(3), Inches(6), Inches(1)).text_frame.text = NOTE
    if with_title:
        s.shapes.title.text = TITLE
        if title_last:
            # Bring to Front: the title placeholder moves to the END of the shape tree.
            sp = s.shapes.title._element
            tree = sp.getparent()
            tree.remove(sp)
            tree.append(sp)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    # a second, ordinary slide so numbering is exercised past slide 1
    s2 = prs.slides.add_slide(prs.slide_layouts[5])
    s2.shapes.title.text = "Second slide"
    s2.shapes.add_textbox(Inches(1), Inches(2), Inches(6), Inches(1)).text_frame.text = "two"
    prs.save(path)
    return path


def _load(path):
    loader, _known, _ext = get_loader(os.path.basename(path), PPTX_MIME, str(path))
    return loader.load()


def _tree_order(path):
    """Precondition helper: the order the TITLE and BODY sit in the saved shape tree."""
    s = Presentation(str(path)).slides[0]
    texts = [sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text]
    return texts


def test_a_title_behind_the_body_in_the_shape_tree_still_leads(tmp_path):
    path = _deck(tmp_path / "front.pptx", title_last=True)
    assert _tree_order(path)[-1] == TITLE, "precondition: the fixture puts the title LAST"
    first = _load(path)[0]
    assert first.page_content == f"{TITLE}\n{BODY}\n{NOTE}", first.page_content
    assert first.metadata["slide_number"] == 1 and first.metadata["slide_title"] == TITLE


def test_the_title_is_not_emitted_twice(tmp_path):
    first = _load(_deck(tmp_path / "front.pptx", title_last=True))[0]
    assert first.page_content.count(TITLE) == 1, first.page_content


def test_an_ordinary_deck_is_byte_identical(tmp_path):
    """Title already first in the tree: output unchanged from before the fix."""
    path = _deck(tmp_path / "plain.pptx", title_last=False, notes="speaker note")
    assert _tree_order(path)[0] == TITLE, "precondition: title FIRST in the tree"
    docs = _load(path)
    assert [d.page_content for d in docs] == [
        f"{TITLE}\n{BODY}\n{NOTE}\n[Notes] speaker note",
        "Second slide\ntwo",
    ]
    assert [d.metadata["slide_number"] for d in docs] == [1, 2]


def test_a_slide_without_a_title_keeps_its_shape_order(tmp_path):
    docs = _load(_deck(tmp_path / "blank.pptx", title_last=False, with_title=False))
    assert docs[0].page_content == f"{BODY}\n{NOTE}"
    assert docs[0].metadata["slide_title"] == ""


def test_the_stored_chunk_leads_with_the_title_through_the_route(tmp_path, monkeypatch):
    """What reaches the table, not just what the loader returned."""
    from main import app
    from app.routes import document_routes
    from tests.utils.test_replace_not_accumulate import FakeStore

    secret = "testsecret"
    os.environ["JWT_SECRET"] = secret
    store = FakeStore()
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0, raising=False)
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    tok = jwt.encode({
        "id": "userA", "tid": "tenantA", "ent": ["userA"], "act": ["read", "write"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }, secret, algorithm="HS256")
    content = _deck(tmp_path / "front.pptx", title_last=True).read_bytes()
    r = TestClient(app).post(
        "/embed", data={"file_id": "kc-pptx", "entity_id": "userA"},
        headers={"Authorization": f"Bearer {tok}"},
        files={"file": ("front.pptx", io.BytesIO(content), PPTX_MIME)},
    )
    assert r.status_code == 200, r.text
    slide1 = [row.document for row in store.rows if row.metadata.get("slide_number") == 1]
    assert slide1 and slide1[0].startswith(TITLE), slide1
