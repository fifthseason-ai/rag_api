"""`/query` now publishes its response shape. This is the test that keeps it honest.

WHY THE SHAPE EXISTS. rag_api published no schema for this response, so a consumer casts instead
of parsing and no compiler on either side can see a divergence. That is how Core came to type
`locator.sheet` as `number | null` while a string arrives — fixed as an instance, with the
structural gap deliberately left open.

WHY THIS TEST EXISTS, WHICH IS A DIFFERENT THING. FastAPI **validates and re-serialises** through a
`response_model`, so a model narrower than reality silently DELETES fields. `metadata` keys differ
per format — a spreadsheet carries `page_name`/`page_number`, a PDF `page`/`page_label`/
`total_pages`, a presentation `slide_number`/`slide_title`, a CSV `row`. Twenty-seven distinct keys
were measured across five formats before the model was written.

Measured, not imagined: typing `metadata` to two named fields reduced the union from **27 keys to
2**. Every locator a citation is built from — gone — while the diff reads as a purely additive
schema addition. That is the failure this file exists to make impossible, and it is far more likely
than the divergence the model was added to prevent.
"""

import datetime
import os

import jwt
import pytest
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from langchain_core.documents import Document

# BEFORE importing main: app.config refuses to start without it, and that refusal is deliberate
# (a protected route that cannot authenticate must not come up). Set at import time, not inside a
# fixture, because the check runs while main is being imported.
os.environ.setdefault("JWT_SECRET", "testsecret")

from app.routes import document_routes  # noqa: E402
from main import app  # noqa: E402

#: Keys no model names, chosen to look like plausible future metadata rather than obvious junk —
#: the realistic case is a parser upgrade adding a field, not someone inserting nonsense.
UNMODELLED = {
    "page_name": "Cost Detail",
    "page_number": 2,
    "slide_number": 5,
    "slide_title": "Roadmap",
    "page": 0,
    "page_label": "iii",
    "total_pages": 9,
    "row": 7,
    "text_source": "ocr",
    "a_key_invented_after_this_model_was_written": "must survive",
    # PROVENANCE, added after review. The first version carried only the locator family, so a
    # narrowing that dropped just these would have passed the guard -- the guard would have been
    # correct about the keys it named and silent about the rest. They are not citation-critical,
    # which is exactly why they are the ones a careless model would omit first.
    "creator": "PyPDF",
    "producer": "pypdf",
    "category": "Table",
    "filetype": "application/pdf",
    "file_directory": "/tmp/uploads/userA",
    "creationdate": "",
    "languages": ["eng"],
    "text_as_html": "<table><tr><td>a</td></tr></table>",
}


def _auth():
    """Same shape the suite already uses; the entity must match the metadata below or the
    route filters the hit out before the response model is ever reached."""
    secret = os.environ["JWT_SECRET"]
    payload = {
        "id": "userA", "tid": "tenantA", "ent": ["userA"],
        "act": ["read", "write", "delete"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": "Bearer " + jwt.encode(payload, secret, algorithm="HS256")}


@pytest.fixture()
def client(monkeypatch):
    async def fake_retrieve(*_args, **_kwargs):
        meta = {"file_id": "f1", "user_id": "userA", "tenant_id": "tenantA"}
        meta.update(UNMODELLED)
        return [(Document(page_content="the passage", metadata=meta), 0.25)]

    monkeypatch.setattr(document_routes, "_retrieve_documents", fake_retrieve)
    monkeypatch.setattr(document_routes, "get_cached_query_embedding", lambda q: [0.1, 0.2, 0.3])
    # NOT a context manager. Entering TestClient runs the app lifespan, which opens a real
    # Postgres connection -- irrelevant to a response-shape test and unavailable here. The
    # suite's own client does the same and initialises the pool by hand.
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test")
    yield TestClient(app)


def _query(client):
    return client.post(
        "/query",
        json={"query": "q", "file_id": "f1", "k": 1, "entity_id": "userA"},
        headers=_auth(),
    )


def test_every_metadata_key_survives_the_response_model(client):
    """The one that matters. A narrowed model deletes locators and looks additive doing it."""
    r = _query(client)
    assert r.status_code == 200, r.text
    hit = r.json()[0]
    doc = hit[0] if isinstance(hit, list) else hit
    meta = doc["metadata"]

    missing = {k: v for k, v in UNMODELLED.items() if k not in meta}
    assert not missing, (
        "the response model DELETED metadata keys: %s. `metadata` must stay an open dict — "
        "typing it reduced the measured union from 27 keys to 2, removing every locator a "
        "citation is built from." % sorted(missing)
    )
    for key, value in UNMODELLED.items():
        assert meta[key] == value, (
            "%s came back as %r, not %r — the model coerced a value it should have passed "
            "through" % (key, meta[key], value)
        )


def test_the_envelope_is_still_a_pair_not_an_object(client):
    """The existing wire is `[[document, score], ...]`. A model may describe it; it may not
    reshape it, because every consumer indexes position 0 and 1 today."""
    r = _query(client)
    hit = r.json()[0]
    assert isinstance(hit, list) and len(hit) == 2, (
        "the response envelope changed shape: %r" % (hit,)
    )
    assert isinstance(hit[1], (int, float)), "the score is no longer a number: %r" % (hit[1],)


def test_strict_validation_is_a_new_failure_mode_and_is_written_down(client):
    """The semantic change #45 introduces, which the PR did not mention.

    A `response_model` does not only DESCRIBE the response -- it VALIDATES it. Before #45 a
    malformed hit was serialised as-is; now it raises `ResponseValidationError` and the caller
    gets a **500** instead of a 200 carrying an odd value.

    Independent review demonstrated this with a `None` score. It is **not reachable today**:
    `_retrieve_documents` calls `round(score, 4)`, which raises on `None` long before the
    response is built, and LangChain guarantees `page_content` is a `str`. So this test asserts
    the GUARD UPSTREAM rather than the 500 -- pinning the thing that makes the failure mode
    unreachable, instead of pinning the failure mode itself.

    If that `round()` is ever removed, this reddens and says why, rather than a 500 appearing in
    production with no explanation attached to it.
    """
    import inspect

    from app.routes import document_routes as dr

    source = inspect.getsource(dr._hybrid_or_dense_search)
    assert "round(score" in source, (
        "`round(score, 4)` has gone from _hybrid_or_dense_search. It is what makes a non-numeric "
        "score fail EARLY with a clear error; without it a None score reaches the response "
        "model and the caller gets a bare 500 ResponseValidationError instead. If this was "
        "removed deliberately, #45's strict validation now has a reachable failure mode and "
        "needs its own handling."
    )


def test_the_document_still_carries_id_and_type(client):
    """LangChain serialises both. Omitting them from the model would drop them from the wire —
    the same silent-deletion failure as the metadata case, one level up."""
    r = _query(client)
    doc = r.json()[0][0]
    for key in ("id", "type", "page_content", "metadata"):
        assert key in doc, "the model dropped %r from the document: %s" % (key, sorted(doc))
