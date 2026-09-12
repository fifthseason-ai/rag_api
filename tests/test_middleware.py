import os
import jwt
import pytest
from app.middleware import security_middleware


# Dummy Request class for testing.
class DummyRequest:
    def __init__(self, path, headers):
        self.url = type("URL", (), {"path": path})
        self.headers = headers
        self.state = type("State", (), {})()


async def dummy_call_next(request):
    return type("DummyResponse", (), {"status_code": 200})()


SECRET = "testsecret"


def _token(payload, secret=SECRET):
    return jwt.encode(payload, secret, algorithm="HS256")


def _full_payload(**overrides):
    payload = {
        "id": "testuser",
        "tid": "tenantA",
        "ent": ["testuser", "kbA"],
        "act": ["read", "write", "delete"],
        "exp": 9999999999,
    }
    payload.update(overrides)
    return payload


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    # Default: a secret is configured and auth is NOT disabled.
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.delenv("RAG_AUTH_DISABLED", raising=False)
    yield


# --- Positive path ----------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_full_token_attaches_entitlement():
    headers = {"Authorization": f"Bearer {_token(_full_payload())}"}
    request = DummyRequest("/protected", headers)
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 200
    assert request.state.user["id"] == "testuser"
    ent = request.state.entitlement
    assert ent["user_id"] == "testuser"
    assert ent["tenant_id"] == "tenantA"
    assert ent["entity_ids"] == {"testuser", "kbA"}
    assert ent["actions"] == {"read", "write", "delete"}


@pytest.mark.asyncio
async def test_health_path_skips_auth():
    request = DummyRequest("/health", {})
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 200


# --- 401: signature / header problems --------------------------------------


@pytest.mark.asyncio
async def test_invalid_token_401():
    request = DummyRequest("/protected", {"Authorization": "Bearer invalidtoken"})
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_missing_authorization_header_401():
    request = DummyRequest("/protected", {})
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_expired_token_401():
    headers = {"Authorization": f"Bearer {_token(_full_payload(exp=1))}"}
    request = DummyRequest("/protected", headers)
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_wrong_secret_signature_401():
    headers = {"Authorization": f"Bearer {_token(_full_payload(), secret='other')}"}
    request = DummyRequest("/protected", headers)
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 401


# --- Fail-closed: no signing secret (D-KSPT-1) -----------------------------


@pytest.mark.asyncio
async def test_missing_secret_fails_closed_500(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    headers = {"Authorization": f"Bearer {_token(_full_payload())}"}
    request = DummyRequest("/protected", headers)
    response = await security_middleware(request, dummy_call_next)
    # Never "auth disabled": a protected route must not pass without a secret.
    assert response.status_code == 500
    assert not hasattr(request.state, "entitlement")


@pytest.mark.asyncio
async def test_missing_secret_with_explicit_optin_passes(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("RAG_AUTH_DISABLED", "true")
    request = DummyRequest("/protected", {})
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 200
    # Explicit local-dev bypass attaches no entitlement.
    assert getattr(request.state, "entitlement", None) is None


# --- Fail-closed: missing entitlement claims (D-KSPT-1) --------------------


@pytest.mark.asyncio
async def test_missing_tid_403():
    payload = _full_payload()
    del payload["tid"]
    request = DummyRequest("/protected", {"Authorization": f"Bearer {_token(payload)}"})
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 403
    assert not hasattr(request.state, "entitlement")


@pytest.mark.asyncio
async def test_empty_ent_403():
    request = DummyRequest(
        "/protected", {"Authorization": f"Bearer {_token(_full_payload(ent=[]))}"}
    )
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_missing_ent_403():
    payload = _full_payload()
    del payload["ent"]
    request = DummyRequest("/protected", {"Authorization": f"Bearer {_token(payload)}"})
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_missing_act_403():
    payload = _full_payload()
    del payload["act"]
    request = DummyRequest("/protected", {"Authorization": f"Bearer {_token(payload)}"})
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_empty_act_403():
    request = DummyRequest(
        "/protected", {"Authorization": f"Bearer {_token(_full_payload(act=[]))}"}
    )
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_legacy_token_without_claims_is_rejected():
    # A pre-KSPT token (only `id`) must no longer be honored: no fallback to id.
    request = DummyRequest(
        "/protected",
        {"Authorization": f"Bearer {_token({'id': 'testuser', 'exp': 9999999999})}"},
    )
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 403
