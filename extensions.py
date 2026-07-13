from __future__ import annotations

import hmac
import os

from flask import jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Created here (not in app.py) and initialized later via limiter.init_app(app) so
# that blueprint route modules (e.g. product_engine/routes.py) can import and use
# @limiter.limit(...) without a circular import against app.py.
#
# Keyed by remote IP, which means every request proxied through one backend
# server (the intended integration pattern - your app's backend calls this
# service, end-user devices never do) shares a single bucket. Default raised
# from the framework's 60/min to something realistic for that pattern, and
# left tunable via env var since "realistic" depends entirely on your real
# traffic volume - raise RATE_LIMIT_DEFAULT in Render's environment if your
# backend legitimately needs more.
limiter = Limiter(
    get_remote_address,
    default_limits=[os.getenv("RATE_LIMIT_DEFAULT", "300 per minute")],
    storage_uri="memory://",
)


def require_api_key(req):
    """Shared X-API-Key gate for every data-returning blueprint route.

    Deliberately permissive when API_KEY isn't configured (unlike the
    fail-closed ADMIN_API_KEY check) so local dev and the test suite keep
    working without extra setup — app.py logs a loud warning at boot instead
    when this is left unset, so the gap is visible, not silent.
    """
    configured_key = os.environ.get("API_KEY", "")
    if not configured_key:
        return None

    provided_key = req.headers.get("X-API-Key", "")
    if not hmac.compare_digest(provided_key, configured_key):
        response = jsonify({"status": "error", "message": "Unauthorized"})
        response.status_code = 401
        return response
    return None


def db_unavailable_response():
    """Shared circuit-breaker response, used everywhere a Mongo-touching route
    checks is_breaker_open() — app.py's /recommend, product_bp's routes, and
    reel_bp's /zatch/* routes all call this so the response shape can never
    drift between them."""
    response = jsonify(
        {"status": "error", "message": "Database temporarily unavailable, please retry shortly"}
    )
    response.status_code = 503
    response.headers["Retry-After"] = "30"
    return response
