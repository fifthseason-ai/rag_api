# app/middleware.py
import os
import jwt
from jwt import PyJWTError
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
    except PyJWTError as e:
        logger.info(
            f"Unauthorized request with invalid token to: {request.url.path}, reason: {str(e)}"
        )
        return JSONResponse(
            status_code=401, content={"detail": f"Invalid token: {str(e)}"}
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
