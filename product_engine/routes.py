from __future__ import annotations

import hmac
import logging
import os

from flask import Blueprint, jsonify, request
from pymongo.errors import PyMongoError

from extensions import db_unavailable_response, limiter, require_api_key

from .final_recommender import (
    get_product_health,
    get_product_interactions,
    get_product_recommendations,
    get_similar_products,
    reload_artifacts,
)
from .zatch_mongo_recommender import ZatchConfigError, is_breaker_open, record_failure

logger = logging.getLogger(__name__)

product_bp = Blueprint("product", __name__)


def _error(message: str, status_code: int):
    return jsonify({"status": "error", "message": message}), status_code


@product_bp.before_request
def _short_circuit_on_db_outage():
    # Scoped to product_bp specifically (not a global app.before_request):
    # a Mongo outage must never block reel_bp's routes, which have nothing
    # to do with Mongo. Fails fast instead of letting every request queue up
    # behind a MONGO_TIMEOUT_MS-long block on a small thread pool.
    #
    # Admin reload is deliberately excluded: it only touches the local
    # joblib file, not Mongo, and is exactly the recovery action an operator
    # might reach for *during* a Mongo outage (forcing artifact-only mode) —
    # it must never be the thing blocked by the outage it's meant to help with.
    if request.endpoint == "product.admin_reload_artifacts":
        return None
    if is_breaker_open():
        return db_unavailable_response()


@product_bp.before_request
def _require_api_key_on_data_routes():
    # Admin reload keeps its own separate, stricter, fail-closed key
    # (_require_admin below) — it must not also require API_KEY.
    if request.endpoint == "product.admin_reload_artifacts":
        return None
    return require_api_key(request)


def _require_admin(req) -> bool:
    configured_key = os.environ.get("ADMIN_API_KEY", "")
    if not configured_key:
        return False  # fail closed: never allow open access just because ops forgot to set it
    provided_key = req.headers.get("X-Admin-Key", "")
    return hmac.compare_digest(provided_key, configured_key)


@product_bp.route("/product-health")
def product_health():
    try:
        engine_status = get_product_health()
        status_code = 200 if engine_status.get("status") == "ok" else 503
        return jsonify(
            {"status": "success" if status_code == 200 else "degraded", "product_engine": engine_status}
        ), status_code
    except ZatchConfigError as exc:
        return _error(str(exc), 503)
    except Exception as exc:
        logger.exception("Product engine health check failed")
        return _error(f"Unable to check product recommender: {exc}", 503)


@product_bp.route("/product-recommendations/<user_id>", methods=["GET"])
def product_recommendations(user_id: str):
    limit = request.args.get("limit", default=10, type=int)
    category = request.args.get("category")
    include_seen = request.args.get("include_seen", default="false").lower() == "true"

    try:
        result = get_product_recommendations(
            user_id=user_id,
            limit=limit,
            category=category,
            include_seen=include_seen,
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


@product_bp.route("/similar-products/<product_id>", methods=["GET"])
def similar_products(product_id: str):
    limit = request.args.get("limit", default=10, type=int)

    try:
        result = get_similar_products(product_id=product_id, limit=limit)
    except ZatchConfigError as exc:
        return _error(str(exc), 503)
    except PyMongoError:
        record_failure()
        logger.exception("Similar product request failed (database error)")
        return _error("Database temporarily unavailable, please retry shortly", 503)
    except Exception:
        logger.exception("Similar product request failed")
        return _error("Internal server error while generating similar products", 500)

    status_code = int(result.pop("status_code", 200))
    return jsonify(result), status_code


@product_bp.route("/product-interactions/<user_id>", methods=["GET"])
def product_interactions(user_id: str):
    try:
        result = get_product_interactions(user_id=user_id)
    except ZatchConfigError as exc:
        return _error(str(exc), 503)
    except PyMongoError:
        record_failure()
        logger.exception("Product interactions request failed (database error)")
        return _error("Database temporarily unavailable, please retry shortly", 503)
    except Exception:
        logger.exception("Product interactions request failed")
        return _error("Internal server error while loading product interactions", 500)

    status_code = int(result.pop("status_code", 200))
    return jsonify(result), status_code


@product_bp.route("/admin/reload-artifacts", methods=["POST"])
@limiter.limit("5 per minute")
def admin_reload_artifacts():
    if not _require_admin(request):
        configured = bool(os.environ.get("ADMIN_API_KEY"))
        return _error("Unauthorized", 401 if configured else 503)

    try:
        ok = reload_artifacts()
    except Exception:
        logger.exception("Failed to reload artifacts")
        return _error("Failed to reload artifacts", 500)

    if ok:
        return jsonify({"status": "success", "message": "Artifacts reloaded"}), 200
    return jsonify({"status": "ok", "message": "No artifacts found to load"}), 200
