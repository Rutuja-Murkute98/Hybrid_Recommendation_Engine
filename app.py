from __future__ import annotations

import json
import logging
import os

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from pymongo.errors import PyMongoError

from extensions import db_unavailable_response, limiter

from reel_engine.routes import reel_bp

from product_engine.routes import product_bp
from product_engine.final_recommender import get_artifacts, get_product_health, get_product_recommendations
from product_engine.zatch_mongo_recommender import (
    ZatchConfigError,
    ensure_indexes,
    get_db,
    get_zatch_reel_health,
    get_zatch_reel_recommendations,
    is_breaker_open,
    record_failure,
)


# Structured JSON logging when available
try:
    from pythonjsonlogger import jsonlogger

    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    logger = logging.getLogger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

# Inert unless SENTRY_DSN is set — a bad/missing DSN must never block boot.
if os.getenv("SENTRY_DSN"):
    try:
        import sentry_sdk
        from sentry_sdk.integrations.flask import FlaskIntegration

        sentry_sdk.init(
            dsn=os.environ["SENTRY_DSN"],
            integrations=[FlaskIntegration()],
            traces_sample_rate=0.0,
        )
    except Exception:
        logging.getLogger(__name__).exception("Failed to initialize Sentry")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1MB — every route here is GET/no-body except
# /admin/reload-artifacts, which ignores its body entirely; this just bounds
# request buffering against oversized payloads.

CORS(app, resources={r"/*": {"origins": os.getenv("ALLOWED_ORIGINS", "*").split(",")}})
limiter.init_app(app)

if not os.environ.get("API_KEY"):
    logger.warning(
        "API_KEY is not set - all recommendation endpoints are running WITHOUT "
        "authentication. Set API_KEY (and send it as the X-API-Key header) "
        "before exposing this service publicly."
    )

# Fail-open: don't block startup if Mongo is briefly unreachable at boot.
# The manual `scripts/ensure_indexes.py` run covers first-deploy / cold-start cases.
# The reel engine's models already loaded successfully above (fail-fast) by the
# time we get here, so intent=reel traffic is unaffected by a Mongo hiccup here.
try:
    ensure_indexes(get_db())
except Exception:
    logger.warning("Could not ensure indexes at startup", exc_info=True)

# Warm the trained-artifact cache at startup instead of lazily on first request:
# get_artifacts() is a joblib.load() of the trained model (measured ~1.6-1.8s) —
# without this, whichever real user's request happens to be first to touch any
# product route pays that latency. get_artifacts() already returns None on
# failure (never raises), so this preserves the product engine's fail-open
# design — it just moves the one-time cost to boot instead of to a live request.
get_artifacts()


@app.before_request
def _log_request():
    try:
        logger.info(json.dumps({
            "event": "request_started",
            "method": request.method,
            "path": request.path,
            "remote_addr": request.remote_addr,
            "args": request.args.to_dict(),
        }))
    except Exception:
        logger.exception("Failed to log request start")


@app.after_request
def _log_response(response):
    try:
        logger.info(json.dumps({
            "event": "request_finished",
            "method": request.method,
            "path": request.path,
            "status": response.status_code,
        }))
    except Exception:
        logger.exception("Failed to log request finish")
    return response


def _error(message: str, status_code: int):
    return jsonify({"status": "error", "message": message}), status_code


app.register_blueprint(reel_bp)
app.register_blueprint(product_bp)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    # Both engines are now live-Mongo-backed, so both can genuinely degrade —
    # neither gets the "already loaded, can't fail" treatment anymore.
    try:
        reel_status = get_zatch_reel_health()
    except Exception as exc:
        logger.exception("Reel health check failed")
        reel_status = {"status": "error", "message": str(exc)}

    try:
        product_status = get_product_health()
    except ZatchConfigError as exc:
        product_status = {"status": "not_configured", "message": str(exc)}
    except Exception as exc:
        logger.exception("Product health check failed")
        product_status = {"status": "error", "message": str(exc)}

    # Deliberately stays 200 even if one engine is degraded: this is a single-
    # worker process, and a Render restart triggered by a transient Mongo
    # hiccup would cost more (dropped in-flight requests, cold caches) than
    # briefly serving a degraded status alongside a healthy one.
    overall_status = "success"
    if reel_status.get("status") not in ("ok", "not_configured"):
        overall_status = "degraded"
    if product_status.get("status") not in ("ok", "not_configured"):
        overall_status = "degraded"

    return jsonify(
        {
            "status": overall_status,
            "message": "Hybrid Recommendation Engine is running",
            "reel_engine": reel_status,
            "product_engine": product_status,
        }
    ), 200


@app.route("/recommend", methods=["GET"])
def recommend():
    """
    GET /recommend?intent=reel&user_id=<id>&video_id=<current_reel_id>&top_n=<int>
    GET /recommend?intent=product&user_id=<id>&limit=<int>&category=<name>&include_seen=<bool>

    intent is client-supplied — your app already knows whether the user is on
    the reels feed or browsing products, so it just passes the right intent.
    Omitting intent defaults to reel behavior for backward compatibility with
    clients calling this exactly as the old standalone reel service did.
    user_id/video_id are Mongo ids (or username/email for user_id) — the reel
    branch is a thin alias for /zatch/reel-recommendations/<user_id>.
    """
    intent = request.args.get("intent")

    if intent == "product":
        # Scoped breaker check: a Mongo outage must 503 this branch without
        # touching the reel branch below (see product_engine/routes.py for
        # the equivalent check on the dedicated product_bp routes).
        if is_breaker_open():
            return db_unavailable_response()

        user_id = request.args.get("user_id")
        if user_id is None:
            return _error("user_id is required. Example: /recommend?intent=product&user_id=<id>", 400)

        limit = request.args.get("limit", default=10, type=int)
        category = request.args.get("category")
        include_seen = request.args.get("include_seen", default="false").lower() == "true"

        try:
            result = get_product_recommendations(
                user_id=user_id, limit=limit, category=category, include_seen=include_seen
            )
        except ZatchConfigError as exc:
            return _error(str(exc), 503)
        except PyMongoError:
            record_failure()
            logger.exception("Product recommendation request failed (database error)")
            return _error("Database temporarily unavailable, please retry shortly", 503)
        except Exception:
            logger.exception("Product recommendation request failed")
            return _error("Internal server error while generating product recommendations", 500)

        status_code = int(result.pop("status_code", 200))
        return jsonify(result), status_code

    # intent == "reel", or no intent at all (back-compat default)
    if is_breaker_open():
        return db_unavailable_response()

    user_id = request.args.get("user_id")
    if user_id is None:
        return _error("user_id is required. Example: /recommend?user_id=<id>&video_id=<current_reel_id>", 400)

    video_id = request.args.get("video_id")
    top_n = request.args.get("top_n", default=10, type=int)

    try:
        result = get_zatch_reel_recommendations(user_id=user_id, current_reel_id=video_id, limit=top_n)
    except ZatchConfigError as exc:
        return _error(str(exc), 503)
    except PyMongoError:
        record_failure()
        logger.exception("Reel recommendation request failed (database error)")
        return _error("Database temporarily unavailable, please retry shortly", 503)
    except Exception as exc:
        logger.error("Error in /recommend: %s", exc, exc_info=True)
        return _error("Internal server error. Please try again.", 500)

    status_code = int(result.pop("status_code", 200))
    return jsonify(result), status_code


@app.errorhandler(404)
def not_found(_error_obj):
    return jsonify(
        {
            "status": "error",
            "message": "Endpoint not found",
            "available_endpoints": [
                "GET /",
                "GET /health",
                "GET /recommend?intent=reel&user_id=<id>&video_id=<current_reel_id>&top_n=<int>",
                "GET /recommend?intent=product&user_id=<id>&limit=<int>&category=<name>&include_seen=<bool>",
                "GET /reel-health",
                "GET /trending?top_n=<int>",
                "GET /user/<user_id>",
                "GET /video/<video_id>",
                "GET /zatch/reel-recommendations/<user_id>?current_reel_id=&limit=",
                "GET /zatch/health",
                "GET /product-health",
                "GET /product-recommendations/<user_id>?limit=<n>&category=<name>&include_seen=<true|false>",
                "GET /similar-products/<product_id>?limit=<n>",
                "GET /product-interactions/<user_id>",
                "POST /admin/reload-artifacts",
            ],
        }
    ), 404


@app.errorhandler(405)
def method_not_allowed(_error_obj):
    return _error("Method not allowed.", 405)


@app.errorhandler(500)
def internal_error(_error_obj):
    # Defense-in-depth: every route already has its own try/except, but this
    # catches anything that slips through (a future route added without one,
    # a bug in error-handling itself) so clients always get JSON, never
    # Flask's default HTML error page.
    return _error("Internal server error.", 500)


def main() -> None:
    port = int(os.getenv("PORT", "5000"))
    logger.info("Starting Hybrid Recommendation Engine on port %s", port)

    try:
        from waitress import serve

        serve(app, host="0.0.0.0", port=port, threads=8)
    except ImportError:
        logger.warning("waitress is not installed; falling back to Flask development server")
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
