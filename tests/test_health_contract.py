"""The /health contract, measured on the WIRE.

WHY THIS FILE EXISTS. At 36d4fb6 -- the revision deployed as task definition 24 -- an unwell
service answered its own health check with:

    HTTP 200   [{"status":"DOWN"},503]

A FastAPI route that RETURNS a (body, status) tuple does not send that status; the tuple becomes the
body. So the status code said UP, the body became a JSON ARRAY, and both consumers of a health check
-- the one that reads the code and the one that reads `status` -- were told the wrong thing at
exactly the moment it mattered. Measured, not deduced: `probe_health.py` in the evidence branch
prints the three answers off a running app.

These tests are the permanent form of that measurement. Each one fails against the pre-fix code, and
the array assertion is the original defect reproduced rather than described.
"""

import os

import pytest
from fastapi.testclient import TestClient


SECRET = "health_contract_secret"


@pytest.fixture(autouse=True)
def auth_configured(monkeypatch):
    """The fail-closed identity middleware 500s with no JWT_SECRET, which would make every case
    below pass for the wrong reason."""
    monkeypatch.setenv("JWT_SECRET", SECRET)


@pytest.fixture
def client():
    from main import app

    return TestClient(app)


@pytest.fixture
def health(monkeypatch):
    """Drive `is_health_ok` from the test. Patched where the ROUTE looks it up, not where it is
    defined -- `document_routes` imported the name, so patching `app.utils.health` would leave the
    route calling the real one and every assertion here would be about the wrong function."""
    from app.routes import document_routes

    def set(outcome):
        async def impl():
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(document_routes, "is_health_ok", impl)

    return set


def test_healthy_is_unchanged(client, health):
    """The UP path is not what was broken, and this pins that the fix did not touch it."""
    health(True)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict)
    assert body["status"] == "UP"


def test_unhealthy_sends_503_not_200(client, health):
    """THE defect. Pre-fix this returns 200 and the test reddens on the status code alone."""
    health(False)
    response = client.get("/health")
    assert response.status_code == 503


def test_unhealthy_body_is_an_object_not_a_tuple(client, health):
    """The second half, and the one a status-code-only assertion would miss: the body was
    `[{"status":"DOWN"},503]`. A consumer reading `body["status"]` got a TypeError or None, so a
    health check that parsed the body was equally misled. `isinstance(list)` is the pre-fix shape,
    asserted directly rather than via a key lookup that could fail for another reason."""
    health(False)
    body = client.get("/health").json()
    assert not isinstance(body, list), "the (body, status) tuple is being serialised as the body"
    assert isinstance(body, dict)
    assert body["status"] == "DOWN"


def test_exception_path_also_sends_503(client, health):
    """An exception inside the health check is a DOWN, not a 200 with an array."""
    health(RuntimeError("boom"))
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "DOWN"


def test_exception_detail_is_not_disclosed(client, health):
    """NON-DISCLOSURE, with something real to withhold.

    A test that raises `RuntimeError("boom")` and asserts "boom" is absent proves nothing -- the
    string is meaningless and its absence costs an attacker nothing. asyncpg's real connection
    errors carry the host, the port, the database and the user, and /health takes NO TOKEN, so that
    output is unauthenticated. The exception below is shaped like the real one; every part of it
    must be absent from the wire, and present in the log the operator reads."""
    secrets = ("vectordb.internal", "5432", "rag_prod_user", "mydatabase")
    health(
        RuntimeError(
            'connection to server at "vectordb.internal", port 5432 failed: '
            'FATAL: password authentication failed for user "rag_prod_user" '
            "(database mydatabase)"
        )
    )
    response = client.get("/health")
    assert response.status_code == 503
    text = response.text
    for secret in secrets:
        assert secret not in text, "%r reached an unauthenticated response" % secret
    body = response.json()
    assert body["status"] == "DOWN"
    # Deliberately NOT `body == {"status": "DOWN"}`. That equality would redden on any additive,
    # harmless field -- and #32 adds exactly one (`build`) to this route -- which makes it a change
    # detector for the wrong property. What must hold is that no field CARRIES THE DETAIL: assert
    # the absence of the keys an exception would leak through, not the absence of all keys.
    for leaky in ("error", "detail", "message", "traceback", "exception"):
        assert leaky not in body, "%r is how the exception text came back before" % leaky


def test_health_takes_no_token(client, health):
    """The exemption this whole file depends on. If /health ever moves behind the fail-closed
    middleware, the tests above would still pass while every probe in front of the service started
    getting 401 -- so the exemption is pinned here rather than assumed. No Authorization header."""
    health(True)
    assert client.get("/health").status_code == 200
    health(False)
    assert client.get("/health").status_code == 503
