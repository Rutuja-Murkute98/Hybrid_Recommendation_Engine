from __future__ import annotations

import datetime as dt
import logging
import math
import os
import threading
from collections import defaultdict
from typing import Any

from bson import ObjectId
from cachetools import TTLCache, cached

from .zatch_mongo_recommender import get_db


logger = logging.getLogger(__name__)

MAX_PRODUCT_LIMIT = 50
PRODUCT_CATALOG_SCAN_LIMIT = int(os.getenv("PRODUCT_CATALOG_SCAN_LIMIT", "5000"))
PRODUCT_CATALOG_CACHE_TTL_SECONDS = int(os.getenv("PRODUCT_CATALOG_CACHE_TTL_SECONDS", "45"))
_CATEGORY_COLLATION = {"locale": "en", "strength": 2}

SIGNAL_SCORE = {
    "order": 5.0,
    "cart": 4.0,
    "review_5star": 4.5,
    "review_4star": 3.5,
    "review_3star": 2.0,
    "review_2star": 1.0,
    "review_1star": 0.5,
    "bargain_accepted": 4.0,
    "bargain_pending": 3.0,
    "bargain_rejected": 1.0,
    "bargain_expired": 1.0,
    "saved": 2.5,
    "liked": 2.0,
    "saved_bit_product": 2.0,
    "liked_bit_product": 1.5,
    "watched_live_product": 1.5,
}

WEIGHTS = {
    "direct_signal": 0.34,
    "category_affinity": 0.22,
    "subcategory_affinity": 0.14,
    "seller_affinity": 0.08,
    "price_affinity": 0.08,
    "popularity": 0.06,
    "discount": 0.04,
    "stock": 0.04,
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


def _clamp_limit(limit: int | None) -> int:
    if limit is None:
        return 10
    return max(1, min(int(limit), MAX_PRODUCT_LIMIT))


def _query_values(raw_id: str) -> list[Any]:
    values: list[Any] = [raw_id]
    oid = _object_id(raw_id)
    if isinstance(oid, ObjectId):
        values.append(oid)
    return values


def _extract_product_id(item: dict[str, Any]) -> str:
    return _string(
        item.get("product")
        or item.get("productId")
        or item.get("product_id")
        or item.get("_id")
    )


def _find_user(db, user_id: str) -> dict[str, Any] | None:
    return db.users.find_one(
        {
            "$or": [
                {"_id": {"$in": _query_values(user_id)}},
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
            "following": 1,
        },
    )


def _active_product_query(category: str | None = None) -> dict[str, Any]:
    query: dict[str, Any] = {
        "isSold": {"$ne": True},
        "totalStock": {"$gt": 0},
        "status": {"$in": ["active", "inactive", "", None]},
    }
    if category:
        query["category"] = category
    return query


_PRODUCT_CATALOG_CACHE: TTLCache = TTLCache(maxsize=64, ttl=PRODUCT_CATALOG_CACHE_TTL_SECONDS)
_PRODUCT_CATALOG_CACHE_LOCK = threading.Lock()


@cached(cache=_PRODUCT_CATALOG_CACHE, lock=_PRODUCT_CATALOG_CACHE_LOCK, key=lambda db, category=None: category or "__all__")
def _load_products(db, category: str | None = None) -> list[dict[str, Any]]:
    # Cached with a short TTL: a recommendation feed tolerates tens-of-seconds
    # staleness on the catalog far better than it tolerates a full collection
    # scan on every single request. Per-user signals are never cached (see
    # _build_user_interactions) since those must stay live.
    projection = {
        "_id": 1,
        "sellerId": 1,
        "name": 1,
        "description": 1,
        "category": 1,
        "subCategory": 1,
        "price": 1,
        "discountedPrice": 1,
        "images": 1,
        "viewCount": 1,
        "saveCount": 1,
        "likeCount": 1,
        "shareCount": 1,
        "totalStock": 1,
        "status": 1,
        "isSold": 1,
        "isTopPick": 1,
        "tags": 1,
        "searchKeywords": 1,
        "analytics": 1,
        "createdAt": 1,
        "updatedAt": 1,
    }
    cursor = db.products.find(_active_product_query(category), projection).limit(
        PRODUCT_CATALOG_SCAN_LIMIT
    )
    if category:
        try:
            cursor = cursor.collation(_CATEGORY_COLLATION)
        except Exception:
            # Some backends (e.g. mongomock in tests) don't support collation;
            # degrade to exact-case matching rather than failing the request.
            logger.debug("Collation not supported by this Mongo backend; using exact-case category match")
    products = list(cursor)
    if len(products) >= PRODUCT_CATALOG_SCAN_LIMIT:
        logger.warning(
            "Active product catalog hit PRODUCT_CATALOG_SCAN_LIMIT=%s (category=%s)",
            PRODUCT_CATALOG_SCAN_LIMIT,
            category,
        )
    return products


def _find_product(db, product_id: str) -> dict[str, Any] | None:
    return db.products.find_one({"_id": {"$in": _query_values(product_id)}})


def _user_preference_categories(user: dict[str, Any]) -> set[str]:
    preferences = user.get("shoppingPreferences") or {}
    categories = {
        str(category).lower()
        for category in _safe_list(preferences.get("categories"))
        if category
    }
    gender = str(user.get("gender") or "").lower().strip()
    return categories | GENDER_CATS.get(gender, set())


def _add_signal(
    interactions: dict[str, dict[str, Any]],
    product_id: str,
    score: float,
    signal: str,
) -> None:
    if not product_id:
        return
    record = interactions.setdefault(product_id, {"score": 0.0, "signals": set()})
    record["score"] = max(float(record["score"]), score)
    record["signals"].add(signal)


def _build_user_interactions(db, user: dict[str, Any]) -> dict[str, dict[str, Any]]:
    user_id = _string(user["_id"])
    oid = _object_id(user_id)
    interactions: dict[str, dict[str, Any]] = {}

    for order in db.orders.find({"buyerId": oid}, {"items": 1}):
        for item in _safe_list(order.get("items")):
            _add_signal(interactions, _extract_product_id(item), SIGNAL_SCORE["order"], "order")

    cart = db.carts.find_one({"user": oid}, {"items": 1})
    if cart:
        for item in _safe_list(cart.get("items")):
            _add_signal(interactions, _extract_product_id(item), SIGNAL_SCORE["cart"], "cart")

    for review in db.reviews.find({"reviewerId": oid}, {"productId": 1, "rating": 1}):
        rating = max(1, min(5, int(review.get("rating") or 3)))
        signal = f"review_{rating}star"
        _add_signal(
            interactions,
            _string(review.get("productId")),
            SIGNAL_SCORE[signal],
            signal,
        )

    for bargain in db.bargains.find({"buyerId": oid}, {"productId": 1, "status": 1}):
        status = str(bargain.get("status") or "pending").lower()
        signal = f"bargain_{status}"
        if signal not in SIGNAL_SCORE:
            signal = "bargain_pending"
        _add_signal(
            interactions,
            _string(bargain.get("productId")),
            SIGNAL_SCORE[signal],
            signal,
        )

    for product_id in _safe_list(user.get("savedProducts")):
        _add_signal(interactions, _string(product_id), SIGNAL_SCORE["saved"], "saved")

    for product_id in _safe_list(user.get("likedProducts")):
        _add_signal(interactions, _string(product_id), SIGNAL_SCORE["liked"], "liked")

    saved_bit_ids = [_object_id(_string(bit_id)) for bit_id in _safe_list(user.get("savedBits"))]
    if saved_bit_ids:
        for bit in db.bits.find({"_id": {"$in": saved_bit_ids}}, {"products": 1}):
            for product_id in _safe_list(bit.get("products")):
                _add_signal(
                    interactions,
                    _string(product_id),
                    SIGNAL_SCORE["saved_bit_product"],
                    "saved_bit_product",
                )

    for bit in db.bits.find({"likes": {"$in": [oid, user_id]}}, {"products": 1}):
        for product_id in _safe_list(bit.get("products")):
            _add_signal(
                interactions,
                _string(product_id),
                SIGNAL_SCORE["liked_bit_product"],
                "liked_bit_product",
            )

    for session in db.livesessions.find({"viewers.userId": oid}, {"products": 1}):
        for product_id in _safe_list(session.get("products")):
            _add_signal(
                interactions,
                _string(product_id),
                SIGNAL_SCORE["watched_live_product"],
                "watched_live_product",
            )

    return interactions


def _product_price(product: dict[str, Any]) -> float:
    price = product.get("discountedPrice") or product.get("price") or 0
    try:
        return float(price)
    except (TypeError, ValueError):
        return 0.0


def _discount_score(product: dict[str, Any]) -> float:
    price = float(product.get("price") or 0)
    discounted = float(product.get("discountedPrice") or price or 0)
    if price <= 0 or discounted >= price:
        return 0.0
    return max(0.0, min(1.0, (price - discounted) / price))


def _stock_score(product: dict[str, Any]) -> float:
    stock = int(product.get("totalStock") or 0)
    if stock <= 0:
        return 0.0
    if stock <= 3:
        return 0.45
    if stock <= 10:
        return 0.75
    return 1.0


def _popularity_score(product: dict[str, Any], max_popularity: float) -> float:
    raw = (
        float(product.get("viewCount") or 0)
        + 3.0 * float(product.get("likeCount") or 0)
        + 2.0 * float(product.get("saveCount") or 0)
        + 1.5 * float(product.get("shareCount") or 0)
    )
    if max_popularity <= 0:
        return 0.0
    return min(1.0, raw / max_popularity)


def _category_affinities(
    interactions: dict[str, dict[str, Any]],
    products_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], list[float]]:
    category_weights: dict[str, float] = defaultdict(float)
    subcategory_weights: dict[str, float] = defaultdict(float)
    seller_weights: dict[str, float] = defaultdict(float)
    prices: list[float] = []

    for product_id, record in interactions.items():
        product = products_by_id.get(product_id)
        if not product:
            continue
        weight = float(record["score"])
        category = str(product.get("category") or "").lower()
        subcategory = str(product.get("subCategory") or "").lower()
        seller = _string(product.get("sellerId"))

        if category:
            category_weights[category] += weight
        if subcategory:
            subcategory_weights[subcategory] += weight
        if seller:
            seller_weights[seller] += weight
        price = _product_price(product)
        if price > 0:
            prices.append(price)

    return dict(category_weights), dict(subcategory_weights), dict(seller_weights), prices


def _normalized_lookup(weights: dict[str, float], key: str) -> float:
    if not weights or not key:
        return 0.0
    total = max(weights.values()) or 1.0
    return min(1.0, float(weights.get(key.lower(), 0.0)) / total)


def _price_affinity(product_price: float, interacted_prices: list[float], user: dict[str, Any]) -> float:
    preferences = user.get("shoppingPreferences") or {}
    price_range = preferences.get("priceRange") or {}

    if interacted_prices:
        mean = sum(interacted_prices) / len(interacted_prices)
        variance = sum((price - mean) ** 2 for price in interacted_prices) / len(interacted_prices)
        std = math.sqrt(variance) or max(mean * 0.25, 1.0)
        return float(math.exp(-0.5 * ((product_price - mean) / std) ** 2))

    min_price = float(price_range.get("min") or 0)
    max_price = float(price_range.get("max") or 0)
    if max_price > 0:
        if min_price <= product_price <= max_price:
            return 1.0
        return max(0.0, 1.0 - abs(product_price - max_price) / (max_price + 1.0))

    return 0.5


def _reason(signals: list[str], category_score: float, price_score: float, popularity: float) -> str:
    labels = {
        "order": "similar to products you ordered",
        "cart": "based on your cart",
        "review_5star": "based on your high ratings",
        "review_4star": "based on your ratings",
        "review_3star": "based on your reviews",
        "review_2star": "based on your reviews",
        "review_1star": "based on your reviews",
        "bargain_accepted": "based on your accepted bargains",
        "bargain_pending": "based on your bargains",
        "bargain_rejected": "based on your bargains",
        "bargain_expired": "based on your bargains",
        "saved": "similar to saved products",
        "liked": "similar to liked products",
        "saved_bit_product": "featured in saved reels",
        "liked_bit_product": "featured in liked reels",
        "watched_live_product": "featured in watched live sessions",
    }
    reasons = [labels.get(signal, signal) for signal in signals[:2]]
    if category_score > 0.45:
        reasons.append("matches your preferred category")
    if price_score > 0.75:
        reasons.append("fits your price range")
    if popularity > 0.55:
        reasons.append("popular on Zatch")
    return " - ".join(reasons[:3]) if reasons else "recommended for you"


def _format_product(
    product: dict[str, Any],
    rank: int,
    score: float,
    breakdown: dict[str, float],
    signals: list[str],
) -> dict[str, Any]:
    return _clean({
        "rank": rank,
        "id": _string(product.get("_id")),
        "name": product.get("name") or "",
        "category": product.get("category") or "",
        "subCategory": product.get("subCategory") or "",
        "sellerId": _string(product.get("sellerId")),
        "price": product.get("price") or 0,
        "discountedPrice": product.get("discountedPrice") or product.get("price") or 0,
        "totalStock": product.get("totalStock") or 0,
        "status": product.get("status") or "",
        "images": product.get("images") or [],
        "viewCount": product.get("viewCount") or 0,
        "likeCount": product.get("likeCount") or 0,
        "saveCount": product.get("saveCount") or 0,
        "score": round(score, 4),
        "signals": signals,
        "reason": _reason(
            signals,
            breakdown.get("category_affinity", 0.0),
            breakdown.get("price_affinity", 0.0),
            breakdown.get("popularity", 0.0),
        ),
        "score_breakdown": {key: round(value, 4) for key, value in breakdown.items()},
    })


def get_product_recommendations(
    user_id: str,
    limit: int = 10,
    category: str | None = None,
    include_seen: bool = False,
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

    products = _load_products(db, category=category)
    products_by_id = {_string(product["_id"]): product for product in products}
    interactions = _build_user_interactions(db, user)
    category_weights, subcategory_weights, seller_weights, interacted_prices = _category_affinities(
        interactions,
        products_by_id,
    )

    preference_categories = _user_preference_categories(user)
    max_popularity = max(
        (
            float(product.get("viewCount") or 0)
            + 3.0 * float(product.get("likeCount") or 0)
            + 2.0 * float(product.get("saveCount") or 0)
            + 1.5 * float(product.get("shareCount") or 0)
            for product in products
        ),
        default=1.0,
    )

    max_direct = max((float(record["score"]) for record in interactions.values()), default=1.0)
    scored = []
    for product in products:
        product_id = _string(product["_id"])
        record = interactions.get(product_id, {"score": 0.0, "signals": set()})
        signals = sorted(record["signals"])
        already_seen = bool(signals)

        if already_seen and not include_seen:
            continue

        category_key = str(product.get("category") or "").lower()
        subcategory_key = str(product.get("subCategory") or "").lower()
        seller_key = _string(product.get("sellerId"))
        product_price = _product_price(product)

        direct_signal = min(1.0, float(record["score"]) / max_direct) if already_seen else 0.0
        category_affinity = max(
            _normalized_lookup(category_weights, category_key),
            0.75 if category_key in preference_categories else 0.0,
        )
        subcategory_affinity = _normalized_lookup(subcategory_weights, subcategory_key)
        seller_affinity = _normalized_lookup(seller_weights, seller_key)
        price_affinity = _price_affinity(product_price, interacted_prices, user)
        popularity = _popularity_score(product, max_popularity)
        discount = _discount_score(product)
        stock = _stock_score(product)

        score = (
            WEIGHTS["direct_signal"] * direct_signal
            + WEIGHTS["category_affinity"] * category_affinity
            + WEIGHTS["subcategory_affinity"] * subcategory_affinity
            + WEIGHTS["seller_affinity"] * seller_affinity
            + WEIGHTS["price_affinity"] * price_affinity
            + WEIGHTS["popularity"] * popularity
            + WEIGHTS["discount"] * discount
            + WEIGHTS["stock"] * stock
        )

        if product.get("isTopPick"):
            score += 0.03
        if already_seen:
            score *= 0.72

        breakdown = {
            "direct_signal": direct_signal,
            "category_affinity": category_affinity,
            "subcategory_affinity": subcategory_affinity,
            "seller_affinity": seller_affinity,
            "price_affinity": price_affinity,
            "popularity": popularity,
            "discount": discount,
            "stock": stock,
        }
        scored.append((score, product, breakdown, signals))

    scored.sort(key=lambda item: item[0], reverse=True)
    recommendations = [
        _format_product(product, rank, score, breakdown, signals)
        for rank, (score, product, breakdown, signals) in enumerate(scored[:limit], start=1)
    ]

    strategy = "zatch-product-signal-hybrid" if interactions else "zatch-product-cold-start"
    return {
        "success": True,
        "userId": _string(user["_id"]),
        "username": user.get("username") or "",
        "strategy": strategy,
        "interaction_count": len(interactions),
        "count": len(recommendations),
        "recommendations": recommendations,
    }


def get_similar_products(product_id: str, limit: int = 10) -> dict[str, Any]:
    limit = _clamp_limit(limit)
    db = get_db()
    source = _find_product(db, product_id)
    if not source:
        return {
            "success": False,
            "message": f"Product not found: {product_id}",
            "status_code": 404,
        }

    products = _load_products(db)
    source_id = _string(source["_id"])
    source_category = str(source.get("category") or "").lower()
    source_subcategory = str(source.get("subCategory") or "").lower()
    source_seller = _string(source.get("sellerId"))
    source_price = _product_price(source)

    max_popularity = max((float(product.get("viewCount") or 0) for product in products), default=1.0)
    scored = []
    for product in products:
        product_id_value = _string(product["_id"])
        if product_id_value == source_id:
            continue

        category = str(product.get("category") or "").lower()
        subcategory = str(product.get("subCategory") or "").lower()
        seller = _string(product.get("sellerId"))
        price = _product_price(product)

        category_score = 1.0 if category == source_category else 0.0
        subcategory_score = 1.0 if subcategory and subcategory == source_subcategory else 0.0
        seller_score = 0.5 if seller and seller == source_seller else 0.0
        price_score = 1.0 / (1.0 + abs(price - source_price) / max(source_price, 1.0))
        popularity = min(1.0, float(product.get("viewCount") or 0) / max_popularity)
        stock = _stock_score(product)

        score = (
            0.36 * category_score
            + 0.26 * subcategory_score
            + 0.12 * seller_score
            + 0.12 * price_score
            + 0.08 * popularity
            + 0.06 * stock
        )
        breakdown = {
            "category_affinity": category_score,
            "subcategory_affinity": subcategory_score,
            "seller_affinity": seller_score,
            "price_affinity": price_score,
            "popularity": popularity,
            "stock": stock,
        }
        scored.append((score, product, breakdown, []))

    scored.sort(key=lambda item: item[0], reverse=True)
    similar = [
        _format_product(product, rank, score, breakdown, signals)
        for rank, (score, product, breakdown, signals) in enumerate(scored[:limit], start=1)
    ]
    return {
        "success": True,
        "productId": source_id,
        "name": source.get("name") or "",
        "count": len(similar),
        "similar_products": similar,
    }


def get_product_interactions(user_id: str) -> dict[str, Any]:
    db = get_db()
    user = _find_user(db, user_id)
    if not user:
        return {
            "success": False,
            "message": f"User not found: {user_id}",
            "status_code": 404,
        }

    interactions = _build_user_interactions(db, user)
    product_ids = [_object_id(pid) for pid in interactions]
    products_by_id = {
        _string(product["_id"]): product
        for product in (db.products.find({"_id": {"$in": product_ids}}) if product_ids else [])
    }

    records = []
    for product_id, record in sorted(interactions.items(), key=lambda item: item[1]["score"], reverse=True):
        product = products_by_id.get(product_id, {})
        records.append({
            "productId": product_id,
            "name": product.get("name") or "",
            "category": product.get("category") or "",
            "subCategory": product.get("subCategory") or "",
            "score": record["score"],
            "signals": sorted(record["signals"]),
        })

    return {
        "success": True,
        "userId": _string(user["_id"]),
        "username": user.get("username") or "",
        "count": len(records),
        "interactions": records,
    }


def get_product_health() -> dict[str, Any]:
    db = get_db()
    try:
        recommendable_products = db.products.count_documents(_active_product_query())
    except Exception:
        recommendable_products = None

    collections = {}
    for name in ["users", "products", "orders", "carts", "bargains", "reviews", "bits", "livesessions"]:
        try:
            collections[name] = db[name].estimated_document_count()
        except Exception:
            collections[name] = None

    return {
        "status": "ok",
        "engine": "zatch-product-signal-hybrid",
        "collections": collections,
        "recommendable_products": recommendable_products,
        "max_limit": MAX_PRODUCT_LIMIT,
        "weights": WEIGHTS,
        "signals": SIGNAL_SCORE,
    }
