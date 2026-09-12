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

@pytest.fixture
def valid_jwt_header():
    jwt_secret = "testsecret"
    os.environ["JWT_SECRET"] = jwt_secret
    payload = {"id": "testuser", "exp": 9999999999}
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def invalid_jwt_header():
    return {"Authorization": "Bearer invalidtoken"}

@pytest.mark.asyncio
async def test_security_middleware_valid(valid_jwt_header):
    request = DummyRequest("/protected", valid_jwt_header)
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 200
    assert hasattr(request.state, "user")
    assert request.state.user["id"] == "testuser"

@pytest.mark.asyncio
async def test_security_middleware_invalid(invalid_jwt_header):
    request = DummyRequest("/protected", invalid_jwt_header)
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 401


# --- RATB-01: fail-closed identity ---


@pytest.mark.asyncio
async def test_security_middleware_secret_unset_fails_closed(monkeypatch):
    """No JWT_SECRET at request time => 503 and call_next is never invoked."""
    monkeypatch.delenv("JWT_SECRET", raising=False)

    called = {"next": False}

    async def tracking_call_next(request):
        called["next"] = True
        return type("DummyResponse", (), {"status_code": 200})()

    request = DummyRequest("/protected", {"Authorization": "Bearer whatever"})
    response = await security_middleware(request, tracking_call_next)

    assert response.status_code == 503
    assert called["next"] is False
    assert not hasattr(request.state, "user")


@pytest.mark.asyncio
async def test_security_middleware_token_without_id(monkeypatch):
    """A validly-signed token that carries no `id` claim => 401 Token lacks identity."""
    jwt_secret = "testsecret"
    monkeypatch.setenv("JWT_SECRET", jwt_secret)
    token = jwt.encode({"exp": 9999999999}, jwt_secret, algorithm="HS256")
    request = DummyRequest("/protected", {"Authorization": f"Bearer {token}"})

    response = await security_middleware(request, dummy_call_next)

    assert response.status_code == 401
    assert not hasattr(request.state, "user")


@pytest.mark.asyncio
async def test_security_middleware_empty_string_id(monkeypatch):
    """A validly-signed token whose `id` is empty/whitespace => 401."""
    jwt_secret = "testsecret"
    monkeypatch.setenv("JWT_SECRET", jwt_secret)
    token = jwt.encode({"id": "   ", "exp": 9999999999}, jwt_secret, algorithm="HS256")
    request = DummyRequest("/protected", {"Authorization": f"Bearer {token}"})

    response = await security_middleware(request, dummy_call_next)

    assert response.status_code == 401
    assert not hasattr(request.state, "user")


@pytest.mark.asyncio
async def test_security_middleware_health_exempt_when_secret_unset(monkeypatch):
    """Exempt paths (ECS health check) still pass through even with no secret."""
    monkeypatch.delenv("JWT_SECRET", raising=False)
    request = DummyRequest("/health", {})
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 200