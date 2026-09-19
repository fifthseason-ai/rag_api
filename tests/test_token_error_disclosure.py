"""What a REJECTED caller is told, and what they must not be able to work out from it.

The protected routes answer an unauthenticated caller, so whatever the rejection says is public.
Until this change it said what PyJWT said:

    Bearer nonsense   -> 401 {"detail":"Invalid token: Not enough segments"}
    Bearer a.b.c      -> 401 {"detail":"Invalid header padding"}
    <forged signature> -> 401 {"detail":"Signature verification failed"}

Three different answers to three different attempts is an ORACLE. The third one is the one that
matters: it tells a caller who is guessing that the token's shape was accepted and only the
signature failed -- i.e. that they are one working key away, and that everything else about their
forgery is right. The service should not help with that, and it costs nothing not to.

So the property under test is not "the message changed". It is that these cases are
INDISTINGUISHABLE on the wire while remaining fully distinguishable in the log.
"""

import os
import uuid

import jwt
import pytest
from fastapi.testclient import TestClient

SECRET = "token_disclosure_secret"
PROTECTED = "/ids"

#: Every PyJWT reason this test can provoke. Each is a real string PyJWT produces, not an invented
#: one -- asserting the absence of a phrase nobody generates would be a test that cannot fail.
PYJWT_PHRASES = (
    "Not enough segments",
    "Invalid header padding",
    "Signature verification failed",
    "Invalid crypto padding",
    "Invalid payload padding",
)


@pytest.fixture(autouse=True)
def auth_configured(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)


@pytest.fixture
def client():
    from main import app

    return TestClient(app)


def _get(client, token):
    return client.get(PROTECTED, headers={"Authorization": "Bearer %s" % token})


def _forged():
    """Structurally perfect, signed with the wrong key: PyJWT says 'Signature verification
    failed', which is the single most useful sentence to withhold."""
    return jwt.encode(
        {"id": "x", "tid": "t", "ent": ["x"], "act": ["read"]},
        "not-the-real-secret",
        algorithm="HS256",
    )


def test_a_malformed_token_is_still_rejected(client):
    """The refusal itself is not what changed, and this pins that it did not."""
    response = _get(client, "nonsense")
    assert response.status_code == 401


def test_the_pyjwt_reason_never_reaches_the_wire(client):
    """SUBSTANTIVE: the withheld strings are the ones PyJWT really produces for these inputs."""
    for token in ("nonsense", "a.b.c", _forged()):
        body = _get(client, token).text
        for phrase in PYJWT_PHRASES:
            assert phrase not in body, (
                "%r reached an unauthenticated caller for token %r" % (phrase, token[:24])
            )


def test_a_forged_signature_is_indistinguishable_from_a_malformed_token(client):
    """THE ORACLE, and the reason this is worth a PR.

    A caller who is guessing must not be able to tell 'your token is gibberish' from 'your token
    is perfect except the signature'. Both are 401; both must read the same, once the per-request
    reference is removed.
    """
    import re

    def shape(token):
        text = _get(client, token).text
        return re.sub(r"[0-9a-f]{12}", "<ref>", text)

    assert shape("nonsense") == shape(_forged()), (
        "the two rejections differ, which tells a forger their shape was accepted"
    )
    assert shape("a.b.c") == shape(_forged())


def test_the_answer_carries_a_reference_the_operator_can_grep(client, caplog):
    """Non-disclosure must not become unsupportability. The caller gets an id; the LOG gets the
    id AND the reason, so a support question is still answerable in one grep."""
    import logging
    import re

    with caplog.at_level(logging.INFO):
        response = _get(client, _forged())

    match = re.search(r"([0-9a-f]{12})", response.text)
    assert match, "the caller must be given something to quote: %r" % response.text
    reference = match.group(1)

    logged = "".join(record.getMessage() for record in caplog.records)
    assert reference in logged, "the reference is useless if it is not in the log"
    assert "Signature verification failed" in logged, (
        "the reason must be PRESERVED for the operator, not deleted -- withholding it from the "
        "wire is the point, losing it is not"
    )


def test_each_rejection_gets_its_own_reference(client):
    """Two rejections must not share an id, or the log cannot tell them apart."""
    import re

    first = re.search(r"([0-9a-f]{12})", _get(client, "nonsense").text).group(1)
    second = re.search(r"([0-9a-f]{12})", _get(client, "nonsense").text).group(1)
    assert first != second


def test_expiry_and_a_missing_header_are_unchanged(client):
    """NOT every message is a disclosure. 'Token has expired' and 'Missing or invalid
    Authorization header' tell a legitimate caller something true and actionable about their own
    request, and reveal nothing about the service. Collapsing them into the generic refusal would
    make an ordinary expiry unexplainable -- non-disclosure is not the same as silence.
    """
    expired = jwt.encode(
        {"id": "x", "tid": "t", "ent": ["x"], "act": ["read"], "exp": 1000000000},
        SECRET,
        algorithm="HS256",
    )
    response = _get(client, expired)
    assert response.status_code == 401
    assert "expired" in response.text.lower()

    missing = client.get(PROTECTED)
    assert missing.status_code == 401
    assert "Authorization header" in missing.text
