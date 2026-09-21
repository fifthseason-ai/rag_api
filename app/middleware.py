# app/middleware.py
import os
import uuid

import jwt
from jwt import ExpiredSignatureError, PyJWTError
from fastapi import Request
from datetime import datetime, timezone
from fastapi.responses import JSONResponse
from app.config import logger

# Paths that never require authentication.
PUBLIC_PATHS = {"/docs", "/openapi.json", "/health"}


async def security_middleware(request: Request, call_next):
    async def next_middleware_call():
        return await call_next(request)

    if request.url.path in PUBLIC_PATHS:
        return await next_middleware_call()

    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret:
        # Fail closed unconditionally (D-KSPT-1): without a signing secret we cannot
        # verify the caller's identity, so no protected route may be served. There
        # is NO opt-out — "missing JWT configuration must fail closed".
        logger.error(
            "JWT_SECRET not configured; refusing protected request to %s",
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Server authentication misconfigured"},
        )

    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        logger.info(
            f"Unauthorized request with missing or invalid Authorization header to: {request.url.path}"
        )
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid Authorization header"},
        )

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        # Defence in depth, NOT the live path: PyJWT already rejects an expired token inside
        # `decode` above (caught as ExpiredSignatureError). This stays as a second opinion in case
        # a future decode is configured with leeway or with `verify_exp` off, and it is documented
        # as unreachable-by-default so nobody reads its existence as proof that expiry is checked
        # here.
        exp_timestamp = payload.get("exp")
        if exp_timestamp and datetime.now(tz=timezone.utc) > datetime.fromtimestamp(
            exp_timestamp, tz=timezone.utc
        ):
            logger.info(
                f"Unauthorized request with expired token to: {request.url.path}"
            )
            return JSONResponse(
                status_code=401, content={"detail": "Token has expired"}
            )
    except ExpiredSignatureError:
        # EXPIRY IS NOT A DISCLOSURE, and collapsing it into the generic refusal below would make
        # the most ordinary 401 this service returns unexplainable to the person holding a token
        # that simply timed out.
        #
        # It is worth being precise about why it is safe, because it looks like the same oracle:
        # reaching this branch means the SIGNATURE VERIFIED. Only a caller who already holds the
        # signing key can get here, and a caller who holds the key has nothing left to learn. A
        # caller who is guessing can never provoke this message -- they get the generic one.
        #
        # This branch also exists because PyJWT checks `exp` itself during `decode`, before the
        # manual check below ever runs: without catching it here, every expired token would have
        # fallen into the generic handler and the explicit "Token has expired" answer would have
        # been unreachable. It was already unreachable before this change; that was invisible
        # because the generic handler happened to echo PyJWT's own wording.
        logger.info(f"Unauthorized request with expired token to: {request.url.path}")
        return JSONResponse(status_code=401, content={"detail": "Token has expired"})
    except PyJWTError as e:
        # The REASON stays in the log and leaves the wire. PyJWT's messages distinguish
        # "Not enough segments" from "Invalid header padding" from "Signature verification
        # failed", and this endpoint answers WITHOUT a token, so returning them hands an
        # unauthenticated caller a free oracle: try a forged signature, and the reply tells you
        # whether the shape was right and only the signature was wrong. Every rejection now reads
        # identically, so the response distinguishes nothing an unauthenticated caller should be
        # able to distinguish.
        #
        # This is the same fix made on `/text` (#26) and on `/health` (#34). It is the last
        # `str(e)` this service returns to a caller it has not authenticated.
        #
        # The reference is what keeps the operator whole: one grep in the log turns a support
        # question into the exact PyJWT reason, without the reason ever being broadcast.
        reference = uuid.uuid4().hex[:12]
        logger.info(
            f"Unauthorized request with invalid token to: {request.url.path}, "
            f"reason: {str(e)} [reference={reference}]"
        )
        return JSONResponse(
            status_code=401,
            content={
                "detail": (
                    "Invalid token. If this is unexpected, quote reference "
                    f"{reference} to an operator."
                )
            },
        )

    # --- Entitlement (D-KSPT-1) ---------------------------------------------
    # Authority comes ONLY from the signed, server-verified token, never from
    # caller-supplied path/body ids. A token that does not carry a tenant, a
    # non-empty entity set and a non-empty action set is rejected (403); we never
    # fall back to `id` or otherwise widen access.
    tenant_id = payload.get("tid")
    entity_ids = payload.get("ent")
    actions = payload.get("act")

    if not tenant_id:
        logger.info(
            "Forbidden request with missing tenant context (tid) to: %s",
            request.url.path,
        )
        return JSONResponse(
            status_code=403, content={"detail": "Missing tenant context"}
        )
    if not entity_ids or not isinstance(entity_ids, (list, tuple)):
        logger.info(
            "Forbidden request with missing/empty entitlement (ent) to: %s",
            request.url.path,
        )
        return JSONResponse(
            status_code=403, content={"detail": "Missing entitlement"}
        )
    if not actions or not isinstance(actions, (list, tuple)):
        logger.info(
            "Forbidden request with missing/empty actions (act) to: %s",
            request.url.path,
        )
        return JSONResponse(
            status_code=403, content={"detail": "Missing authorized actions"}
        )

    # Backwards-compatible: routes/helpers that still read request.state.user.
    request.state.user = payload
    request.state.entitlement = {
        "user_id": payload.get("id"),
        "tenant_id": str(tenant_id),
        "entity_ids": {str(e) for e in entity_ids},
        "actions": {str(a) for a in actions},
    }
    logger.debug(f"{request.url.path} - entitlement={request.state.entitlement}")

    return await next_middleware_call()
