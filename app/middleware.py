# app/middleware.py
import os
import jwt
from jwt import PyJWTError
from fastapi import Request
from datetime import datetime, timezone
from fastapi.responses import JSONResponse
from app.config import logger


async def security_middleware(request: Request, call_next):
    async def next_middleware_call():
        return await call_next(request)

    if request.url.path in {"/docs", "/openapi.json", "/health"}:
        return await next_middleware_call()

    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret:
        # Fail closed (RATB-01): without a secret the service cannot verify any
        # token, so it must NOT pass requests through unauthenticated. Defense in
        # depth behind the startup check in app.config.
        logger.error(
            "JWT_SECRET not configured; refusing request to %s (fail closed)",
            request.url.path,
        )
        return JSONResponse(
            status_code=503, content={"detail": "JWT verification unavailable"}
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

        # Require a non-empty string identity claim (RATB-01). A token that
        # verifies but carries no usable `id` must not be treated as an
        # authenticated principal.
        user_id = payload.get("id")
        if not isinstance(user_id, str) or not user_id.strip():
            logger.info(
                f"Unauthorized request with token lacking identity to: {request.url.path}"
            )
            return JSONResponse(
                status_code=401, content={"detail": "Token lacks identity"}
            )

        request.state.user = payload
        logger.debug(f"{request.url.path} - {payload}")
    except PyJWTError as e:
        logger.info(
            f"Unauthorized request with invalid token to: {request.url.path}, reason: {str(e)}"
        )
        return JSONResponse(
            status_code=401, content={"detail": f"Invalid token: {str(e)}"}
        )

    return await next_middleware_call()