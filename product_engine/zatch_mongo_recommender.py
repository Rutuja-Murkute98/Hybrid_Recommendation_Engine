from __future__ import annotations

import datetime as dt
import logging
import os
import random
import threading
import time
from functools import lru_cache
from typing import Any

import certifi
from bson import ObjectId
from cachetools import TTLCache, cached
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError


load_dotenv()
logger = logging.getLogger(__name__)

MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "zatch")
MONGO_TIMEOUT_MS = int(os.getenv("MONGO_TIMEOUT_MS", "5000"))
MONGO_MAX_POOL_SIZE = int(os.getenv("MONGO_MAX_POOL_SIZE", "20"))
MONGO_MIN_POOL_SIZE = int(os.getenv("MONGO_MIN_POOL_SIZE", "0"))
MAX_ZATCH_LIMIT = 50

# Minimal in-process circuit breaker: after enough consecutive Mongo failures,
# short-circuit requests for a cooldown window instead of letting every one of
# them block for up to MONGO_TIMEOUT_MS while a small thread pool is exhausted.
_BREAKER_FAILURE_THRESHOLD = int(os.getenv("MONGO_BREAKER_FAILURE_THRESHOLD", "5"))
_BREAKER_COOLDOWN_SECONDS = int(os.getenv("MONGO_BREAKER_COOLDOWN_SECONDS", "30"))
_breaker_lock = threading.Lock()
_breaker_fail_count = 0
_breaker_opened_until = 0.0


def record_success() -> None:
    global _breaker_fail_count, _breaker_opened_until
    with _breaker_lock:
        _breaker_fail_count = 0
        _breaker_opened_until = 0.0


def record_failure() -> None:
    global _breaker_fail_count, _breaker_opened_until
    with _breaker_lock:
        _breaker_fail_count += 1
        if _breaker_fail_count >= _BREAKER_FAILURE_THRESHOLD:
            _breaker_opened_until = time.monotonic() + _BREAKER_COOLDOWN_SECONDS


def is_breaker_open() -> bool:
    with _breaker_lock:
        return time.monotonic() < _breaker_opened_until


# Short-TTL cache for the bits/live-session scan behind /zatch/reel-recommendations
# — same rationale as product_engine's catalog cache: a "for you" reel feed
# tolerates tens-of-seconds staleness far better than re-scanning up to 500
# documents on every single request. `db` is deliberately excluded from the
# cache key (it's a process-wide singleton via get_db()'s own lru_cache).
ZATCH_CACHE_TTL_SECONDS = int(os.getenv("ZATCH_CACHE_TTL_SECONDS", "30"))
_bits_cache: TTLCache = TTLCache(maxsize=4, ttl=ZATCH_CACHE_TTL_SECONDS)
_bits_cache_lock = threading.Lock()
_sessions_cache: TTLCache = TTLCache(maxsize=4, ttl=ZATCH_CACHE_TTL_SECONDS)
_sessions_cache_lock = threading.Lock()


@cached(cache=_bits_cache, lock=_bits_cache_lock, key=lambda db, limit: limit)
def _fetch_bits(db, limit: int) -> list[dict[str, Any]]:
    return list(db.bits.find({}).limit(limit))


@cached(cache=_sessions_cache, lock=_sessions_cache_lock, key=lambda db, limit: limit)
def _fetch_live_sessions(db, limit: int) -> list[dict[str, Any]]:
    return list(db.livesessions.find({"status": {"$ne": "draft"}}).limit(limit))


SESSION_SCORE = {
    "current_reel_similarity": 6.0,
    "watched": 5.0,
    "liked_session": 4.0,
    "product_ordered": 3.5,
    "product_carted": 3.0,
    "product_saved": 2.5,
    "host_bought_from": 2.5,
    "category_match": 2.0,
    "product_liked": 1.5,
    "recency": 1.0,
    "trending": 0.5,
}

BIT_SCORE = {
    "current_reel_similarity": 6.0,
    "liked_bit": 5.0,
    "saved_bit": 4.5,
    "product_ordered": 4.0,
    "product_carted": 3.0,
    "product_saved": 3.0,
    "creator_bought_from": 2.5,
    "product_liked": 2.0,
    "hashtag_match": 2.0,
    "category_match": 1.5,
    "trending": 1.5,
    "recency": 1.0,
    "bargain_product": 1.0,
}

GENDER_CATS = {
    "female": {
        "women",
        "beauty & personal care",
        "beauty and personal care",
        "baby & kids",
        "home",
        "bed-bath",
    },
    "male": {
        "men",
        "electronics",
        "sports & fitness",
        "sports and fitness",
    },
}


class ZatchConfigError(RuntimeError):
    pass


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _object_id(value: str) -> ObjectId | str:
    try:
        return ObjectId(value)
    except Exception:
        return value


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dt.datetime):
        return value.isoformat()
    return value


def _parse_date(value: Any) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None
    return None


def _clamp_limit(limit: int | None) -> int:
    if limit is None:
        return 10
    return max(1, min(int(limit), MAX_ZATCH_LIMIT))


def _normalize_score(score: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.0
    return round(min(1.0, score / max_score), 4)


def _shuffle_within_score_bands(items: list, score_fn, band_width: float = 1.0) -> list:
    """Rank by score (descending) but shuffle randomly within each same-score
    band instead of a strict stable sort.

    A thin, early-stage catalog produces a lot of exact or near ties
    (popularity fallback especially, where most items start at 0 views/
    likes) — a plain sort would show every user the identical ordering
    forever, biasing future popularity toward whatever happened to rank
    first. Items that are meaningfully better still always rank above
    meaningfully worse ones; only ties/near-ties within `band_width` of
    each other get reordered, and differently on every call.
    """
    bands: dict[int, list] = {}
    for item in items:
        band = int(score_fn(item) // band_width) if band_width > 0 else 0
        bands.setdefault(band, []).append(item)

    ordered = []
    for band in sorted(bands, reverse=True):
        band_items = bands[band]
        random.shuffle(band_items)
        ordered.extend(band_items)
    return ordered


def _make_client(uri: str) -> MongoClient:
    clean_uri = uri
    for param in [
        "&tls=true",
        "?tls=true",
        "&ssl=true",
        "?ssl=true",
        "&tlsInsecure=true",
        "&tlsAllowInvalidCertificates=true",
        "&authSource=admin",
    ]:
        clean_uri = clean_uri.replace(param, "")

    return MongoClient(
        clean_uri,
        serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
        socketTimeoutMS=MONGO_TIMEOUT_MS,
        connectTimeoutMS=MONGO_TIMEOUT_MS,
        tls=True,
        tlsCAFile=certifi.where(),
        authSource="admin",
        retryWrites=True,
        maxPoolSize=MONGO_MAX_POOL_SIZE,
        minPoolSize=MONGO_MIN_POOL_SIZE,
    )


@lru_cache(maxsize=1)
def get_db():
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise ZatchConfigError("MONGO_URI is not configured")

    client = _make_client(uri)
    client.admin.command("ping")
    return client[MONGO_DB_NAME]


def check_db_connection() -> dict[str, Any]:
    """Actively probe Mongo reachability, independent of get_db()'s cached singleton.

    get_db() only pings once, on its first successful call, so a later outage
    would otherwise go undetected until a real query blocks. Health checks must
    call this instead of assuming a cached client handle means "reachable."

    Skips the live ping entirely while the circuit breaker is open: every
    caller of this function (product-health, combined /health, zatch/health)
    would otherwise block for up to MONGO_TIMEOUT_MS on every single call
    during a known outage — exactly the thread-exhaustion scenario the
    breaker exists to prevent, including on Render's own healthCheckPath
    polling. Fixing this here (once) covers every caller instead of needing
    a bespoke breaker check duplicated into each health route.
    """
    if is_breaker_open():
        return {"status": "error", "message": "Circuit breaker open (recent Mongo failures)"}

    start = time.monotonic()
    try:
        db = get_db()
        db.command("ping")
    except ZatchConfigError as exc:
        return {"status": "not_configured", "message": str(exc)}
    except PyMongoError as exc:
        record_failure()
        return {
            "status": "error",
            "message": str(exc),
            "latency_ms": round((time.monotonic() - start) * 1000, 2),
        }
    record_success()
    return {"status": "ok", "latency_ms": round((time.monotonic() - start) * 1000, 2)}


def ensure_indexes(db) -> None:
    """Create the indexes the hot-path queries rely on. Idempotent — safe to call on every start."""
    db.products.create_index(
        [("isSold", 1), ("totalStock", 1), ("status", 1)], name="idx_active_products"
    )
    db.products.create_index(
        [("category", 1)],
        name="idx_category_ci",
        collation={"locale": "en", "strength": 2},
    )
    db.products.create_index([("sellerId", 1)], name="idx_seller")
    db.orders.create_index([("buyerId", 1)], name="idx_orders_buyer")
    db.carts.create_index([("user", 1)], name="idx_carts_user")
    db.reviews.create_index([("reviewerId", 1)], name="idx_reviews_reviewer")
    db.bargains.create_index([("buyerId", 1)], name="idx_bargains_buyer")
    db.bits.create_index([("likes", 1)], name="idx_bits_likes")
    db.livesessions.create_index([("viewers.userId", 1)], name="idx_livesessions_viewer")
    db.livesessions.create_index([("status", 1)], name="idx_livesessions_status")
    db.users.create_index([("username", 1)], name="idx_users_username")
    db.users.create_index([("email", 1)], name="idx_users_email")


def _find_user(db, user_id: str) -> dict[str, Any] | None:
    query_values = [user_id]
    oid = _object_id(user_id)
    if isinstance(oid, ObjectId):
        query_values.append(oid)

    return db.users.find_one(
        {
            "$or": [
                {"_id": {"$in": query_values}},
                {"username": user_id},
                {"email": user_id},
            ],
            "isAdmin": {"$ne": True},
        },
        {
            "_id": 1,
            "username": 1,
            "email": 1,
            "gender": 1,
            "shoppingPreferences": 1,
            "savedProducts": 1,
            "likedProducts": 1,
            "savedBits": 1,
            "searchHistory": 1,
        },
    )


def _find_reel(db, reel_id: str | None) -> tuple[str | None, dict[str, Any] | None]:
    if not reel_id:
        return None, None

    query_values = [reel_id]
    oid = _object_id(reel_id)
    if isinstance(oid, ObjectId):
        query_values.append(oid)

    bit = db.bits.find_one({"_id": {"$in": query_values}})
    if bit:
        return "bit", bit

    session = db.livesessions.find_one({"_id": {"$in": query_values}})
    if session:
        return "live_session", session

    return None, None


def _user_categories(user: dict[str, Any]) -> set[str]:
    preferences = user.get("shoppingPreferences") or {}
    categories = {
        str(category).lower()
        for category in _safe_list(preferences.get("categories"))
        if category
    }
    gender = str(user.get("gender") or "").lower().strip()
    return categories | GENDER_CATS.get(gender, set())


def _product_categories(db, product_ids: set[str]) -> set[str]:
    if not product_ids:
        return set()

    object_ids = []
    string_ids = []
    for product_id in product_ids:
        oid = _object_id(product_id)
        if isinstance(oid, ObjectId):
            object_ids.append(oid)
        string_ids.append(product_id)

    categories = set()
    for product in db.products.find(
        {"$or": [{"_id": {"$in": object_ids}}, {"_id": {"$in": string_ids}}]},
        {"category": 1, "subCategory": 1},
    ):
        for key in ["category", "subCategory"]:
            value = product.get(key)
            if value:
                categories.add(str(value).lower())
    return categories


def _user_product_signals(db, user: dict[str, Any]) -> dict[str, set[str]]:
    user_id = _string(user["_id"])
    oid = _object_id(user_id)
    signals = {
        "ordered": set(),
        "carted": set(),
        "saved": {_string(pid) for pid in _safe_list(user.get("savedProducts")) if pid},
        "liked": {_string(pid) for pid in _safe_list(user.get("likedProducts")) if pid},
        "saved_bits": {_string(bit_id) for bit_id in _safe_list(user.get("savedBits")) if bit_id},
        "bargained": set(),
        "hosts": set(),
    }

    for order in db.orders.find({"buyerId": oid}, {"items.productId": 1, "sellerId": 1, "hostId": 1}):
        for item in _safe_list(order.get("items")):
            product_id = _string(item.get("productId"))
            if product_id:
                signals["ordered"].add(product_id)
        seller_id = _string(order.get("sellerId") or order.get("hostId"))
        if seller_id:
            signals["hosts"].add(seller_id)

    cart = db.carts.find_one({"user": oid}, {"items.productId": 1})
    if cart:
        for item in _safe_list(cart.get("items")):
            product_id = _string(item.get("productId"))
            if product_id:
                signals["carted"].add(product_id)

    for bargain in db.bargains.find({"buyerId": oid}, {"productId": 1}):
        product_id = _string(bargain.get("productId"))
        if product_id:
            signals["bargained"].add(product_id)

    return signals


def _current_reel_context(db, current_reel: dict[str, Any] | None) -> dict[str, set[str]]:
    if not current_reel:
        return {"products": set(), "hashtags": set(), "creators": set(), "categories": set()}

    products = {_string(product_id) for product_id in _safe_list(current_reel.get("products")) if product_id}
    hashtags = {
        str(tag).lower()
        for tag in _safe_list(current_reel.get("hashtags"))
        if tag
    }
    creators = {
        _string(
            current_reel.get("creatorId")
            or current_reel.get("hostId")
            or current_reel.get("userId")
        )
    }
    creators.discard("")

    return {
        "products": products,
        "hashtags": hashtags,
        "creators": creators,
        "categories": _product_categories(db, products),
    }


def _score_current_similarity(
    item_products: set[str],
    item_hashtags: set[str],
    item_creator: str,
    context: dict[str, set[str]],
) -> bool:
    if item_products & context["products"]:
        return True
    if item_hashtags & context["hashtags"]:
        return True
    if item_hashtags & context["categories"]:
        return True
    if item_creator and item_creator in context["creators"]:
        return True
    return False


def _reason(labels: list[str]) -> str:
    readable = {
        "current_reel_similarity": "similar to the current reel",
        "watched": "watched earlier",
        "liked_session": "liked live session",
        "liked_bit": "liked this bit",
        "saved_bit": "saved this bit",
        "product_ordered": "features an ordered product",
        "product_carted": "features a cart product",
        "product_saved": "features a saved product",
        "product_liked": "features a liked product",
        "host_bought_from": "host you bought from",
        "creator_bought_from": "creator you bought from",
        "category_match": "matches your shopping style",
        "hashtag_match": "matches your interests",
        "trending": "trending",
        "recency": "recent",
        "bargain_product": "features bargained product",
    }
    if not labels:
        return "popular fallback"
    return " - ".join(readable.get(label, label) for label in labels[:3])


def _score_session(
    session: dict[str, Any],
    user: dict[str, Any],
    user_signals: dict[str, set[str]],
    user_categories: set[str],
    current_context: dict[str, set[str]],
    now: dt.datetime,
) -> tuple[float, list[str]]:
    user_id = _string(user["_id"])
    session_products = {_string(pid) for pid in _safe_list(session.get("products")) if pid}
    session_hashtags = {str(tag).lower() for tag in _safe_list(session.get("hashtags")) if tag}
    host_id = _string(session.get("hostId"))
    viewers = {_string(viewer.get("userId")) for viewer in _safe_list(session.get("viewers")) if isinstance(viewer, dict)}
    likes = {_string(like) for like in _safe_list(session.get("likes")) if like}

    score = 0.0
    labels = []

    if _score_current_similarity(session_products, session_hashtags, host_id, current_context):
        score += SESSION_SCORE["current_reel_similarity"]
        labels.append("current_reel_similarity")
    if user_id in viewers:
        score += SESSION_SCORE["watched"]
        labels.append("watched")
    if user_id in likes:
        score += SESSION_SCORE["liked_session"]
        labels.append("liked_session")
    if session_products & user_signals["ordered"]:
        score += SESSION_SCORE["product_ordered"]
        labels.append("product_ordered")
    if session_products & user_signals["carted"]:
        score += SESSION_SCORE["product_carted"]
        labels.append("product_carted")
    if session_products & user_signals["saved"]:
        score += SESSION_SCORE["product_saved"]
        labels.append("product_saved")
    if session_products & user_signals["liked"]:
        score += SESSION_SCORE["product_liked"]
        labels.append("product_liked")
    if host_id and host_id in user_signals["hosts"]:
        score += SESSION_SCORE["host_bought_from"]
        labels.append("host_bought_from")
    if session_hashtags & user_categories:
        score += SESSION_SCORE["category_match"]
        labels.append("category_match")

    created_at = _parse_date(session.get("scheduledStartTime") or session.get("createdAt"))
    if created_at and (now - created_at).days <= 7:
        score += SESSION_SCORE["recency"]
        labels.append("recency")

    views = int(session.get("views") or 0)
    peak = int(session.get("peakViewers") or 0)
    if views > 50 or peak > 10:
        score += SESSION_SCORE["trending"]
        labels.append("trending")

    return score, labels


def _score_bit(
    bit: dict[str, Any],
    user: dict[str, Any],
    user_signals: dict[str, set[str]],
    user_categories: set[str],
    current_context: dict[str, set[str]],
    now: dt.datetime,
) -> tuple[float, list[str]]:
    user_id = _string(user["_id"])
    bit_products = {_string(pid) for pid in _safe_list(bit.get("products")) if pid}
    bit_hashtags = {str(tag).lower() for tag in _safe_list(bit.get("hashtags")) if tag}
    creator_id = _string(bit.get("creatorId") or bit.get("hostId") or bit.get("userId"))
    likes = {_string(like) for like in _safe_list(bit.get("likes")) if like}
    bit_id = _string(bit.get("_id"))

    score = 0.0
    labels = []

    if _score_current_similarity(bit_products, bit_hashtags, creator_id, current_context):
        score += BIT_SCORE["current_reel_similarity"]
        labels.append("current_reel_similarity")
    if user_id in likes:
        score += BIT_SCORE["liked_bit"]
        labels.append("liked_bit")
    if bit_id in user_signals["saved_bits"]:
        score += BIT_SCORE["saved_bit"]
        labels.append("saved_bit")
    if bit_products & user_signals["ordered"]:
        score += BIT_SCORE["product_ordered"]
        labels.append("product_ordered")
    if bit_products & user_signals["carted"]:
        score += BIT_SCORE["product_carted"]
        labels.append("product_carted")
    if bit_products & user_signals["saved"]:
        score += BIT_SCORE["product_saved"]
        labels.append("product_saved")
    if bit_products & user_signals["liked"]:
        score += BIT_SCORE["product_liked"]
        labels.append("product_liked")
    if bit_products & user_signals["bargained"]:
        score += BIT_SCORE["bargain_product"]
        labels.append("bargain_product")
    if creator_id and creator_id in user_signals["hosts"]:
        score += BIT_SCORE["creator_bought_from"]
        labels.append("creator_bought_from")
    if bit_hashtags & user_categories:
        score += BIT_SCORE["hashtag_match"]
        labels.append("hashtag_match")

    created_at = _parse_date(bit.get("createdAt"))
    if created_at and (now - created_at).days <= 14:
        score += BIT_SCORE["recency"]
        labels.append("recency")

    views = int(bit.get("viewCount") or bit.get("views") or 0)
    likes_count = len(likes)
    if likes_count > 20 or (views > 0 and likes_count / max(views, 1) > 0.1):
        score += BIT_SCORE["trending"]
        labels.append("trending")

    return score, labels


def _format_session(session: dict[str, Any], rank: int, score: float, labels: list[str]) -> dict[str, Any]:
    return {
        "rank": rank,
        "id": _string(session.get("_id")),
        "type": "live_session",
        "title": session.get("title") or session.get("name") or "",
        "status": session.get("status") or "",
        "hostId": _string(session.get("hostId")),
        "products": [_string(pid) for pid in _safe_list(session.get("products")) if pid],
        "hashtags": _safe_list(session.get("hashtags")),
        "views": session.get("views") or 0,
        "peakViewers": session.get("peakViewers") or 0,
        "thumbnail": session.get("thumbnail") or {},
        "score": round(score, 4),
        "normalized_score": _normalize_score(score, sum(SESSION_SCORE.values())),
        "signals": labels,
        "reason": _reason(labels),
    }


def _format_bit(bit: dict[str, Any], rank: int, score: float, labels: list[str]) -> dict[str, Any]:
    return {
        "rank": rank,
        "id": _string(bit.get("_id")),
        "type": "bit",
        "title": bit.get("title") or bit.get("caption") or "",
        "creatorId": _string(bit.get("creatorId") or bit.get("hostId") or bit.get("userId")),
        "products": [_string(pid) for pid in _safe_list(bit.get("products")) if pid],
        "hashtags": _safe_list(bit.get("hashtags")),
        "likes": len(_safe_list(bit.get("likes"))),
        "views": bit.get("viewCount") or bit.get("views") or 0,
        "thumbnail": bit.get("thumbnail") or bit.get("coverImage") or "",
        "videoUrl": bit.get("videoUrl") or bit.get("url") or "",
        "createdAt": _clean(bit.get("createdAt")),
        "score": round(score, 4),
        "normalized_score": _normalize_score(score, sum(BIT_SCORE.values())),
        "signals": labels,
        "reason": _reason(labels),
    }


def _popular_fallback(
    db, user: dict[str, Any] | None, limit: int, current_reel_id: str | None
) -> list[dict[str, Any]]:
    user_categories = _user_categories(user) if user else set()
    fallback = []
    now = dt.datetime.utcnow()

    for bit in _fetch_bits(db, 300):
        bit_id = _string(bit.get("_id"))
        if current_reel_id and bit_id == current_reel_id:
            continue
        hashtags = {str(tag).lower() for tag in _safe_list(bit.get("hashtags")) if tag}
        views = int(bit.get("viewCount") or bit.get("views") or 0)
        likes = len(_safe_list(bit.get("likes")))
        score = min(5.0, (views / 100.0) + likes)
        labels = []
        if hashtags & user_categories:
            score += 2.0
            labels.append("category_match")
        created_at = _parse_date(bit.get("createdAt"))
        if created_at and (now - created_at).days <= 14:
            score += 1.0
            labels.append("recency")
        if views or likes:
            labels.append("trending")
        fallback.append(("bit", bit, score, labels))

    for session in _fetch_live_sessions(db, 300):
        session_id = _string(session.get("_id"))
        if current_reel_id and session_id == current_reel_id:
            continue
        hashtags = {str(tag).lower() for tag in _safe_list(session.get("hashtags")) if tag}
        views = int(session.get("views") or 0)
        peak = int(session.get("peakViewers") or 0)
        score = min(5.0, (views / 100.0) + peak)
        labels = []
        if hashtags & user_categories:
            score += 2.0
            labels.append("category_match")
        created_at = _parse_date(session.get("scheduledStartTime") or session.get("createdAt"))
        if created_at and (now - created_at).days <= 7:
            score += 1.0
            labels.append("recency")
        if views or peak:
            labels.append("trending")
        fallback.append(("live_session", session, score, labels))

    fallback = _shuffle_within_score_bands(fallback, score_fn=lambda item: item[2], band_width=1.0)
    formatted = []
    for rank, (kind, document, score, labels) in enumerate(fallback[:limit], start=1):
        if kind == "bit":
            formatted.append(_format_bit(document, rank, score, labels))
        else:
            formatted.append(_format_session(document, rank, score, labels))
    return formatted


def get_zatch_reel_recommendations(
    user_id: str,
    current_reel_id: str | None = None,
    limit: int = 10,
    include_types: str = "all",
) -> dict[str, Any]:
    limit = _clamp_limit(limit)
    db = get_db()
    user = _find_user(db, user_id)
    if not user:
        return {
            "success": False,
            "message": f"User not found: {user_id}",
            "status_code": 404,
        }

    current_type, current_reel = _find_reel(db, current_reel_id)
    user_signals = _user_product_signals(db, user)
    user_categories = _user_categories(user)
    current_context = _current_reel_context(db, current_reel)
    now = dt.datetime.utcnow()

    scored = []
    current_id = _string(current_reel.get("_id")) if current_reel else current_reel_id

    if include_types in {"all", "bits", "bit"}:
        for bit in _fetch_bits(db, 500):
            bit_id = _string(bit.get("_id"))
            if current_id and bit_id == current_id:
                continue
            score, labels = _score_bit(bit, user, user_signals, user_categories, current_context, now)
            if score > 0:
                scored.append(("bit", bit, score, labels))

    if include_types in {"all", "sessions", "live_sessions", "live_session"}:
        for session in _fetch_live_sessions(db, 500):
            session_id = _string(session.get("_id"))
            if current_id and session_id == current_id:
                continue
            score, labels = _score_session(session, user, user_signals, user_categories, current_context, now)
            if score > 0:
                scored.append(("live_session", session, score, labels))

    scored = _shuffle_within_score_bands(scored, score_fn=lambda item: item[2], band_width=1.0)

    recommendations = []
    for rank, (kind, document, score, labels) in enumerate(scored[:limit], start=1):
        if kind == "bit":
            recommendations.append(_format_bit(document, rank, score, labels))
        else:
            recommendations.append(_format_session(document, rank, score, labels))

    strategy = "zatch-signal-hybrid"
    if len(recommendations) < limit:
        existing_ids = {item["id"] for item in recommendations}
        fallback = [
            item
            for item in _popular_fallback(db, user, limit * 2, current_id)
            if item["id"] not in existing_ids
        ]
        for item in fallback:
            item["rank"] = len(recommendations) + 1
            recommendations.append(item)
            if len(recommendations) == limit:
                break
        if not scored:
            strategy = "zatch-cold-start"

    return _clean({
        "success": True,
        "userId": _string(user["_id"]),
        "username": user.get("username") or "",
        "current_reel_id": current_reel_id,
        "current_reel_type": current_type,
        "strategy": strategy,
        "count": len(recommendations),
        "recommendations": recommendations[:limit],
    })


def get_trending_reels(limit: int = 20) -> dict[str, Any]:
    """Global, no-login trending bits/live sessions — replaces the old
    CSV/pickle reel engine's popularity fallback with a live Mongo equivalent."""
    limit = _clamp_limit(limit)
    db = get_db()
    recommendations = _popular_fallback(db, None, limit, None)
    return _clean({
        "success": True,
        "strategy": "zatch-trending",
        "count": len(recommendations),
        "recommendations": recommendations,
    })


def get_reel_status(reel_id: str) -> dict[str, Any]:
    """Existence check for a bit/live-session id, replacing the old reel
    engine's video-index lookup against the now-retired CSV dataset."""
    db = get_db()
    reel_type, reel = _find_reel(db, reel_id)
    return {
        "reel_id": reel_id,
        "exists": reel is not None,
        "type": reel_type,
    }


def get_reel_user_status(user_id: str) -> dict[str, Any]:
    """Existence + engagement check for a user, replacing the old reel
    engine's cold-start check against the now-retired CSV dataset."""
    db = get_db()
    user = _find_user(db, user_id)
    if not user:
        return {"user_id": user_id, "exists": False, "recommendation_mode": "cold_start"}

    signals = _user_product_signals(db, user)
    engagement_signal_count = (
        len(signals["ordered"])
        + len(signals["carted"])
        + len(signals["saved"])
        + len(signals["liked"])
        + len(_safe_list(user.get("savedBits")))
    )
    return {
        "user_id": _string(user["_id"]),
        "username": user.get("username") or "",
        "exists": True,
        "engagement_signal_count": engagement_signal_count,
        "recommendation_mode": "personalized" if engagement_signal_count > 0 else "cold_start",
    }


_ZATCH_HEALTH_COLLECTIONS = ["users", "bits", "livesessions", "products", "orders", "carts", "bargains"]


def get_zatch_reel_health() -> dict[str, Any]:
    db_status = check_db_connection()

    if db_status.get("status") == "ok":
        # get_db() is safe to call again here: check_db_connection() just
        # confirmed it's reachable, and get_db()'s own @lru_cache means this
        # returns the same already-connected client instantly — no second
        # connection attempt, no second ping.
        db = get_db()
        collections = {}
        for name in _ZATCH_HEALTH_COLLECTIONS:
            try:
                collections[name] = db[name].estimated_document_count()
            except Exception:
                collections[name] = None
    else:
        # Mongo is known-unreachable (or the breaker is open) — don't attempt
        # a second connection just to watch it fail again.
        collections = {name: None for name in _ZATCH_HEALTH_COLLECTIONS}

    # Unlike the product engine, there's no artifact-mode fallback here — the
    # zatch reel endpoints are 100% Mongo-dependent, so "not_configured" must
    # NOT be folded into "ok" the way product_engine's health check does.
    status = db_status.get("status", "error")
    if status not in ("ok", "not_configured"):
        status = "degraded"

    return {
        "status": status,
        "database": MONGO_DB_NAME,
        "database_connection": db_status,
        "collections": collections,
        "max_limit": MAX_ZATCH_LIMIT,
    }
