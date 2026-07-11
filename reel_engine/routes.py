from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request
from pymongo.errors import PyMongoError

from extensions import db_unavailable_response, require_api_key

from product_engine.zatch_mongo_recommender import (
    ZatchConfigError,
    get_reel_status,
    get_reel_user_status,
    get_trending_reels,
    get_zatch_reel_health,
    get_zatch_reel_recommendations,
    is_breaker_open,
    record_failure,
)

logger = logging.getLogger(__name__)

reel_bp = Blueprint("reel", __name__)


@reel_bp.before_request
def _protect_reel_routes():
    return require_api_key(request)


def _error(message: str, status_code: int):
    return jsonify({"status": "error", "message": message}), status_code


@reel_bp.route("/reel-health")
def reel_health():
    try:
        status = get_zatch_reel_health()
    except Exception as exc:
        logger.exception("Reel health check failed")
        return _error(str(exc), 503)

    status_code = 200 if status.get("status") == "ok" else 503
    return jsonify(
        {"status": "success" if status_code == 200 else "degraded", "reel_engine": status}
    ), status_code


@reel_bp.route("/trending")
def trending():
    if is_breaker_open():
        return db_unavailable_response()

    try:
        top_n = request.args.get("top_n", default=20, type=int)
        result = get_trending_reels(limit=top_n)
    except ZatchConfigError as exc:
        return _error(str(exc), 503)
    except PyMongoError:
        record_failure()
        logger.exception("Trending request failed (database error)")
        return _error("Database temporarily unavailable, please retry shortly", 503)
    except Exception:
        logger.error("Error in /trending", exc_info=True)
        return _error("Internal server error.", 500)

    status_code = int(result.pop("status_code", 200))
    return jsonify(result), status_code


@reel_bp.route("/user/<user_id>")
def user_info(user_id: str):
    if is_breaker_open():
        return db_unavailable_response()

    try:
        status = get_reel_user_status(user_id)
    except ZatchConfigError as exc:
        return _error(str(exc), 503)
    except PyMongoError:
        record_failure()
        logger.exception("User status request failed (database error)")
        return _error("Database temporarily unavailable, please retry shortly", 503)
    except Exception:
        logger.exception("User status request failed")
        return _error("Internal server error.", 500)

    return jsonify({"status": "success", **status}), 200


@reel_bp.route("/video/<video_id>")
def video_info(video_id: str):
    if is_breaker_open():
        return db_unavailable_response()

    try:
        status = get_reel_status(video_id)
    except ZatchConfigError as exc:
        return _error(str(exc), 503)
    except PyMongoError:
        record_failure()
        logger.exception("Reel status request failed (database error)")
        return _error("Database temporarily unavailable, please retry shortly", 503)
    except Exception:
        logger.exception("Reel status request failed")
        return _error("Internal server error.", 500)

    return jsonify({"status": "success", **status}), 200


@reel_bp.route("/zatch/reel-recommendations/<user_id>")
def zatch_reel_recommendations(user_id: str):
    if is_breaker_open():
        return db_unavailable_response()

    current_reel_id = request.args.get("current_reel_id")
    limit = request.args.get("limit", default=10, type=int)
    include_types = request.args.get("include_types", default="all")

    try:
        result = get_zatch_reel_recommendations(
            user_id=user_id,
            current_reel_id=current_reel_id,
            limit=limit,
            include_types=include_types,
        )
    except ZatchConfigError as exc:
        return _error(str(exc), 503)
    except PyMongoError:
        record_failure()
        logger.exception("Zatch reel recommendation request failed (database error)")
        return _error("Database temporarily unavailable, please retry shortly", 503)
    except Exception:
        logger.exception("Zatch reel recommendation request failed")
        return _error("Internal server error while generating Zatch recommendations", 500)

    status_code = int(result.pop("status_code", 200))
    return jsonify(result), status_code


@reel_bp.route("/zatch/health")
def zatch_health():
    try:
        status = get_zatch_reel_health()
    except Exception as exc:
        logger.exception("Zatch health check failed")
        return _error(str(exc), 503)

    status_code = 200 if status.get("status") == "ok" else 503
    return jsonify(
        {"status": "success" if status_code == 200 else "degraded", "zatch_mongodb": status}
    ), status_code
