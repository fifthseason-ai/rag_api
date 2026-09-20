"""`GET /ids` handed every file identifier in the store to every authenticated caller.

MEASURED ON THE WIRE, before this change, against a real service and a real pgvector:

    tenantB GET /ids  ->  200 ["acme-merger-2026-confidential", "tenantb-own-file"]

The route asserted the `read` ACTION and then ran a query with no entity predicate at all,
so a valid token for any tenant returned a complete list of every file the service held.

A file identifier is not page content, and it is not nothing either. It is the argument
every other route takes, it is frequently the customer's own document id or filename, and
the full list is a map of what another tenant holds. Eight sibling routes were swept in the
same pass and all eight denied correctly -- `GET /documents` and `/documents/{id}/context`
already keep only rows whose `user_id` is within the entitlement. This route was the one
that did not, which is why the predicate here is theirs and not a new policy.

The double models the TABLE -- rows that belong to owners -- rather than mocking the
method under test, so a scoping predicate that is deleted or inverted changes what these
tests see.
"""

import datetime
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient

from main import app
from app.routes import document_routes
from app.services.vector_store.async_pg_vector import AsyncPgVector

_SECRET = "testsecret"

#: The identifier itself is the disclosure: a tenant's deal name, in the id.
A_FILE = "acme-merger-2026-confidential"
B_FILE = "tenantb-own-file"


def _hdr(entity="userB", tenant="tenantB", ent=None, act=("read",)):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": entity,
        "tid": tenant,
        "ent": [entity] if ent is None else list(ent),
        "act": list(act),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


class OwnedRowStore(AsyncPgVector):
    """Rows that belong to owners, which is the only property this route turns on.

    Subclasses the real class so `isinstance(vector_store, AsyncPgVector)` picks the same
    branch production takes. `super().__init__` is deliberately not called -- there is no
    engine and no connection -- and `_bind` is set because PGVector.__del__ reads it.
    """

    def __init__(self, rows):
        self._bind = None
        self.rows = dict(rows)  # file_id -> owning entity
        self.asked_for = None

    async def get_ids_for_entities(self, entity_ids, executor=None):
        self.asked_for = list(entity_ids)
        if not entity_ids:
            return []
        return [f for f, owner in self.rows.items() if owner in entity_ids]

    async def get_all_ids(self, executor=None):
        # The unscoped primitive. If the route ever calls this again, the tests below say so.
        self.asked_for = "UNSCOPED"
        return list(self.rows)


class StoreWithoutScoping:
    """A store that cannot scope: only the unscoped primitive exists.

    Deliberately NOT a subclass of AsyncPgVector. The first version of this double
    inherited from it, which meant it inherited `get_ids_for_entities` too -- so the
    `hasattr` guard never fired and the test failed for an unrelated reason. A double that
    accidentally satisfies the capability check cannot test the capability check.
    """

    async def get_all_ids(self, executor=None):
        return list(self.rows)

    def __init__(self, rows):
        self.rows = dict(rows)


@pytest.fixture()
def rows():
    return {A_FILE: "userA", B_FILE: "userB"}


@pytest.fixture()
def client_with(monkeypatch):
    def _make(store):
        os.environ["JWT_SECRET"] = _SECRET
        if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
            app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
        monkeypatch.setattr(document_routes, "vector_store", store)
        return TestClient(app)

    return _make


# ---------------------------------------------------------------------------
# The leak this file reproduces
# ---------------------------------------------------------------------------


def test_one_tenant_never_sees_another_tenants_file_identifiers(client_with, rows):
    """THE REGRESSION. Before this change, tenant B's /ids returned tenant A's file."""
    client = client_with(OwnedRowStore(rows))
    r = client.get("/ids", headers=_hdr("userB", "tenantB"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert A_FILE not in body, "another tenant's file identifier was disclosed: %s" % body


def test_a_caller_still_sees_their_own(client_with, rows):
    """The positive half. A scoping change that returned nothing to anybody would pass
    the test above and break the route."""
    client = client_with(OwnedRowStore(rows))
    r = client.get("/ids", headers=_hdr("userB", "tenantB"))
    assert r.status_code == 200, r.text
    assert B_FILE in r.json()


def test_the_route_asks_for_the_tokens_entities_not_everything(client_with, rows):
    """Asserted at the seam, because the two failures are indistinguishable in the body
    when a caller happens to own everything: a route that scopes correctly and a route
    that asks for the whole store both return the same list to that caller."""
    store = OwnedRowStore(rows)
    client = client_with(store)
    r = client.get("/ids", headers=_hdr("userB", "tenantB"))
    assert r.status_code == 200
    assert store.asked_for == ["userB"], store.asked_for


def test_a_multi_entity_token_sees_exactly_its_entities(client_with):
    """An entitlement covering several entities is normal; it must widen to exactly
    those and no further."""
    store = OwnedRowStore({A_FILE: "userA", B_FILE: "userB", "third": "userC"})
    client = client_with(store)
    r = client.get("/ids", headers=_hdr("userB", "tenantB", ent=("userB", "userC")))
    assert r.status_code == 200
    assert sorted(r.json()) == sorted([B_FILE, "third"]), r.json()


def test_an_empty_entitlement_returns_nothing_not_everything(client_with, rows):
    """The dangerous shape. The sibling delete path in this codebase treats a falsy id
    list as "no filter"; the same reading here would turn an entitlement with no entities
    into a disclosure of the entire store."""
    store = OwnedRowStore(rows)
    client = client_with(store)
    r = client.get("/ids", headers=_hdr("userB", "tenantB", ent=()))
    # The middleware may refuse an empty entitlement outright; either way, what must never
    # happen is a 200 carrying somebody else's identifiers.
    if r.status_code == 200:
        assert r.json() == [], r.json()
    else:
        assert r.status_code in (401, 403), r.status_code


def test_a_store_that_cannot_scope_is_refused_not_fallen_back(client_with, rows):
    """FAIL CLOSED. Falling back to the unscoped list would be the exact disclosure this
    change exists to stop, and it would be invisible: the response shape is identical."""
    client = client_with(StoreWithoutScoping(rows))
    r = client.get("/ids", headers=_hdr("userB", "tenantB"))
    assert r.status_code == 501, r.text
    assert A_FILE not in r.text and B_FILE not in r.text


def test_the_unscoped_primitive_is_not_reached_by_the_route(client_with, rows):
    """A change-detector on the one call that caused this. If `/ids` ever goes back to
    `get_all_ids`, this fails by name rather than by a body that looks plausible."""
    store = OwnedRowStore(rows)
    client = client_with(store)
    client.get("/ids", headers=_hdr("userB", "tenantB"))
    assert store.asked_for != "UNSCOPED", "the route called the unscoped primitive"


def test_read_is_still_required(client_with, rows):
    """The action check must survive the scoping change -- a token without `read` is
    still refused before any identifier is considered."""
    client = client_with(OwnedRowStore(rows))
    r = client.get("/ids", headers=_hdr("userB", "tenantB", act=("write",)))
    assert r.status_code == 403, r.text
    assert B_FILE not in r.text


def test_an_unexpected_failure_does_not_describe_the_service(client_with, rows):
    """Found by the fail-closed test above, which produced

        {"detail": "'StoreWithoutScoping' object has no attribute 'EmbeddingStore'"}

    -- an internal class and an ORM attribute, handed to an unauthenticated-in-principle
    caller. On a database fault the same line would carry connection or schema details.
    Same shape as the PyJWT disclosure repaired on the protected routes: the exception
    belongs in the log under a reference, not in the response."""
    class Exploding(AsyncPgVector):
        def __init__(self):
            self._bind = None

        async def get_ids_for_entities(self, entity_ids, executor=None):
            raise RuntimeError(
                "connection to server at 'db.internal' (10.0.4.12), port 5432 failed"
            )

    client = client_with(Exploding())
    r = client.get("/ids", headers=_hdr("userB", "tenantB"))
    assert r.status_code == 500, r.text
    body = r.text
    for secret in ("db.internal", "10.0.4.12", "5432", "RuntimeError", "EmbeddingStore"):
        assert secret not in body, "the response described the service: %r" % body
    assert "reference" in body.lower(), body
