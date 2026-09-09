"""
Access control for the deployed API.
====================================

The deployment model is "anyone with the link *and* the code". That is not
user authentication and does not pretend to be — it exists to stop three
specific things once the app is on the public internet:

  * crawlers and scanners hitting a medical-sounding endpoint,
  * strangers spending the project's Groq quota,
  * casual discovery of a tool that must not be mistaken for medical advice.

Design notes
------------
* The code is compared with `hmac.compare_digest`, so a wrong guess takes the
  same time as a right one and the endpoint cannot be brute-forced by timing.
* `/api/health` stays open because container orchestrators and uptime checks
  need it before any secret is available. It returns nothing but liveness.
* In development (`ENV=development`) with no code set, the gate is disabled so
  the team is not forced to type a code locally. In production a missing code
  is a hard startup failure rather than a silently open API — an unprotected
  deployment is the failure mode this module exists to prevent.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("mediassist.security")

ENV = os.getenv("ENV", "development").strip().lower()
ACCESS_CODE = os.getenv("ACCESS_CODE", "").strip()
ACCESS_HEADER = "X-Access-Code"

# Paths reachable without the code. Deliberately tiny.
#   /health      — liveness, needed by orchestrators before secrets exist
#   /api/config  — tells the UI whether to show the code prompt; nothing else
PUBLIC_PATHS = {"/health", "/api/health", "/api/config"}

# Only the API is gated.
#
# The static shell — index.html, the JS bundle, the stylesheet — must always
# be served, because that shell *is* the screen where the code gets entered.
# Gating it produced a deployment where the browser received a 401 JSON body
# instead of the app, and there was no way in at all. The code protects data
# and the LLM quota; it was never meant to protect the HTML.
GATED_PREFIX = "/api/"


def access_control_enabled() -> bool:
    return bool(ACCESS_CODE)


def verify_startup_configuration() -> None:
    """
    Refuse to start an unprotected production deployment.

    Called from the app lifespan. A public URL with no gate is the one
    misconfiguration that cannot be noticed from the outside until it is
    already being abused.
    """
    if ENV == "production" and not ACCESS_CODE:
        raise RuntimeError(
            "ACCESS_CODE is not set but ENV=production.\n"
            "Set a shared access code before deploying, or run with "
            "ENV=development for local work.\n"
            "Generate one with:  python -c \"import secrets; print(secrets.token_urlsafe(12))\""
        )
    if ACCESS_CODE and len(ACCESS_CODE) < 8:
        raise RuntimeError(
            f"ACCESS_CODE is only {len(ACCESS_CODE)} characters. "
            "Use at least 8; 16+ is better."
        )
    if not ACCESS_CODE:
        logger.warning(
            "ACCESS_CODE not set — the API is OPEN. Acceptable for local "
            "development only."
        )


def _submitted_code(request: Request) -> Optional[str]:
    """Header first; query parameter as a fallback for simple link sharing."""
    header = request.headers.get(ACCESS_HEADER, "").strip()
    if header:
        return header
    return (request.query_params.get("access_code") or "").strip() or None


class AccessCodeMiddleware(BaseHTTPMiddleware):
    """Rejects any non-public request that does not carry the shared code."""

    async def dispatch(self, request: Request, call_next):
        if not ACCESS_CODE:
            return await call_next(request)

        path = request.url.path.rstrip("/") or "/"
        if path in PUBLIC_PATHS:
            return await call_next(request)

        # Everything outside the API is the static app shell — always served.
        if not request.url.path.startswith(GATED_PREFIX):
            return await call_next(request)

        # Browsers send a credential-less preflight; the real request is checked.
        if request.method == "OPTIONS":
            return await call_next(request)

        submitted = _submitted_code(request)
        if submitted and hmac.compare_digest(submitted, ACCESS_CODE):
            return await call_next(request)

        request_id = getattr(request.state, "request_id", "")
        logger.info(
            "access denied path=%s rid=%s reason=%s",
            path, request_id, "missing_code" if not submitted else "bad_code",
        )
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "code": "access_denied",
                    "message": "This assistant requires an access code.",
                    "details": {"header": ACCESS_HEADER},
                },
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id} if request_id else None,
        )
