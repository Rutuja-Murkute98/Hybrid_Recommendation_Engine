from __future__ import annotations

import datetime as dt
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from bson import ObjectId

from .zatch_mongo_recommender import check_db_connection, get_db

from . import recommender as live_recommender


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
ARTIFACT_PATH = MODEL_DIR / "product_recommendation_artifacts.joblib"


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, ObjectId):
        return str(value)
    return value


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    minimum = float(values.min())
    maximum = float(values.max())
    if math.isclose(minimum, maximum):
        return np.ones_like(values, dtype=float)
    return (values - minimum) / (maximum - minimum)


def _load_artifacts() -> dict[str, Any] | None:
    if not ARTIFACT_PATH.exists():
        return None
    try:
        artifacts = joblib.load(ARTIFACT_PATH)
        logger.info("Loaded trained product artifacts from %s", ARTIFACT_PATH)
        return artifacts
    except Exception:
        logger.exception("Failed to load product artifacts")
        return None


ARTIFACTS: dict[str, Any] | None = None


def get_artifacts() -> dict[str, Any] | None:
    """Lazily load artifacts on first use and return them.

    Returns None when no artifact file is present or on load failure.
    """
    global ARTIFACTS
    if ARTIFACTS is None:
        ARTIFACTS = _load_artifacts()
    return ARTIFACTS


def reload_artifacts() -> bool:
    """Force reloading artifacts from disk. Returns True when loaded."""
    global ARTIFACTS
    ARTIFACTS = _load_artifacts()
    return ARTIFACTS is not None


def _artifact_catalog() -> dict[str, dict[str, Any]]:
    artifacts = get_artifacts()
    return artifacts.get("product_metadata", {}) if artifacts else {}


def _artifact_history(user_id: str) -> dict[str, float]:
    artifacts = get_artifacts()
    if not artifacts:
        return {}
    history = artifacts.get("user_histories", {}).get(_string(user_id), [])
    return {str(product_id): float(score) for product_id, score in history}


def _live_history(db, user: dict[str, Any]) -> dict[str, float]:
    interactions = live_recommender._build_user_interactions(db, user)
    return {
        product_id: float(record["score"])
        for product_id, record in interactions.items()
    }


def _build_user_profile(product_index: dict[str, int], product_matrix, history: dict[str, float]):
    if not history:
        return None

    indices = []
    weights = []
    for product_id, score in history.items():
        index = product_index.get(product_id)
        if index is None:
            continue
        indices.append(index)
        weights.append(float(score))

    if not indices:
        return None

    weighted_matrix = product_matrix[indices].multiply(np.asarray(weights)[:, None])
    profile = weighted_matrix.sum(axis=0)
    return np.asarray(profile).ravel()


def _content_scores(profile_vector, product_matrix) -> np.ndarray:
    if profile_vector is None:
        return np.zeros(product_matrix.shape[0], dtype=float)
    raw = np.asarray(profile_vector @ product_matrix.T).ravel()
    return _normalize(raw.astype(float))


def _collaborative_scores(product_index: dict[str, int], item_neighbors: dict[str, list[tuple[str, float]]], history: dict[str, float], size: int) -> np.ndarray:
    scores = np.zeros(size, dtype=float)
    for product_id, history_weight in history.items():
        for neighbor_id, similarity in item_neighbors.get(product_id, []):
            neighbor_index = product_index.get(neighbor_id)
            if neighbor_index is None:
                continue
            scores[neighbor_index] += float(history_weight) * float(similarity)
    return _normalize(scores)


def _business_scores(product_ids: list[str], product_catalog: dict[str, dict[str, Any]], popularity: dict[str, float]) -> np.ndarray:
    scores = []
    max_popularity = max(popularity.values()) if popularity else 0.0
    for product_id in product_ids:
        metadata = product_catalog.get(product_id, {})
        popularity_score = float(popularity.get(product_id, 0.0))
        if max_popularity > 0:
            popularity_score = popularity_score / max_popularity

        stock = float(metadata.get("totalStock") or 0)
        if stock <= 0:
            stock_score = 0.0
        elif stock <= 3:
            stock_score = 0.45
        elif stock <= 10:
            stock_score = 0.75
        else:
            stock_score = 1.0

        price = float(metadata.get("price") or 0)
        discounted = float(metadata.get("discountedPrice") or price or 0)
        if price <= 0 or discounted >= price:
            discount_score = 0.0
        else:
            discount_score = min(1.0, (price - discounted) / price)

        top_pick_score = 0.05 if metadata.get("isTopPick") else 0.0
        scores.append(0.55 * popularity_score + 0.25 * stock_score + 0.15 * discount_score + top_pick_score)

    return _normalize(np.asarray(scores, dtype=float))


def _format_product(product_id: str, rank: int, score: float, content: float, collab: float, business: float, metadata: dict[str, Any], signals: list[str]) -> dict[str, Any]:
    return _clean(
        {
            "rank": rank,
            "id": product_id,
            "name": metadata.get("name") or "",
            "category": metadata.get("category") or "",
            "subCategory": metadata.get("subCategory") or "",
            "sellerId": _string(metadata.get("sellerId")),
            "price": metadata.get("price") or 0,
            "discountedPrice": metadata.get("discountedPrice") or metadata.get("price") or 0,
            "totalStock": metadata.get("totalStock") or 0,
            "status": metadata.get("status") or "",
            "images": metadata.get("images") or [],
            "viewCount": metadata.get("viewCount") or 0,
            "likeCount": metadata.get("likeCount") or 0,
            "saveCount": metadata.get("saveCount") or 0,
            "score": round(float(score), 4),
            "score_breakdown": {
                "content": round(float(content), 4),
                "collaborative": round(float(collab), 4),
                "business": round(float(business), 4),
            },
            "signals": signals,
            "reason": "trained hybrid product recommendation",
        }
    )


def _fetch_live_products(db, product_ids: list[str]) -> dict[str, Any]:
    """Batch-fetch current, active product docs for the given ids, keyed by
    string id — used so artifact-mode results always show live name/price/
    stock/images instead of the training-time snapshot."""
    if not product_ids:
        return {}

    query_values: list[Any] = []
    for product_id in product_ids:
        query_values.extend(live_recommender._query_values(product_id))

    query = {"_id": {"$in": query_values}, **live_recommender._active_product_query()}
    return {live_recommender._string(doc["_id"]): doc for doc in db.products.find(query)}


def _fallback_live_recommendations(user_id: str, limit: int, category: str | None, include_seen: bool):
    return live_recommender.get_product_recommendations(
        user_id=user_id,
        limit=limit,
        category=category,
        include_seen=include_seen,
    )


def get_product_recommendations(
    user_id: str,
    limit: int = 10,
    category: str | None = None,
    include_seen: bool = False,
) -> dict[str, Any]:
    if not get_artifacts():
        return _fallback_live_recommendations(user_id, limit, category, include_seen)

    limit = max(1, min(int(limit), 50))
    db = get_db()
    user = live_recommender._find_user(db, user_id)
    if not user:
        return {
            "success": False,
            "message": f"User not found: {user_id}",
            "status_code": 404,
        }

    artifacts = get_artifacts() or {}
    product_ids = artifacts.get("product_ids", [])
    product_index = artifacts.get("product_index", {})
    product_matrix = artifacts.get("product_tfidf_matrix")
    product_catalog = _artifact_catalog()
    item_neighbors = artifacts.get("item_neighbors", {})
    popularity = artifacts.get("popularity", {})

    if not product_ids or product_matrix is None:
        return _fallback_live_recommendations(user_id, limit, category, include_seen)

    history = _artifact_history(user_id)
    if not history:
        history = _live_history(db, user)

    seen = set(history)

    profile_vector = _build_user_profile(product_index, product_matrix, history)
    content_scores = _content_scores(profile_vector, product_matrix)
    collaborative_scores = _collaborative_scores(product_index, item_neighbors, history, len(product_ids))
    business_scores = _business_scores(product_ids, product_catalog, popularity)

    preferred_categories = live_recommender._user_preference_categories(user)
    candidates = []
    for index, product_id in enumerate(product_ids):
        if not include_seen and product_id in seen:
            continue

        metadata_snapshot = product_catalog.get(product_id, {})
        product_category = str(metadata_snapshot.get("category") or "").lower()
        if category and product_category != category.lower():
            continue

        content = float(content_scores[index])
        collaborative = float(collaborative_scores[index])
        business = float(business_scores[index])
        category_boost = 0.05 if product_category and product_category in preferred_categories else 0.0
        score = 0.48 * content + 0.36 * collaborative + 0.16 * business + category_boost
        candidates.append((score, product_id, content, collaborative, business))

    candidates.sort(key=lambda item: item[0], reverse=True)

    # Scoring stays trained (expensive to recompute, fine to be periodically
    # stale); displayed metadata is always resolved live so a sold-out,
    # re-priced, or renamed product never shows stale data. Buffer beyond
    # `limit` since some trained-time candidates may no longer be live-active.
    buffer_ids = [product_id for _, product_id, *_ in candidates[: limit * 3]]
    live_products = _fetch_live_products(db, buffer_ids)

    recommendations = []
    for score, product_id, content, collaborative, business in candidates:
        metadata = live_products.get(product_id)
        if metadata is None:
            continue  # sold out / deactivated / deleted since training
        recommendations.append(
            _format_product(product_id, len(recommendations) + 1, score, content, collaborative, business, metadata, [])
        )
        if len(recommendations) == limit:
            break

    strategy = "trained-hybrid-product-recommender"
    if not recommendations:
        return _fallback_live_recommendations(user_id, limit, category, include_seen)

    return {
        "success": True,
        "userId": _string(user["_id"]),
        "username": user.get("username") or "",
        "strategy": strategy,
        "artifact_mode": True,
        "interaction_count": len(history),
        "count": len(recommendations),
        "recommendations": recommendations,
    }


def get_similar_products(product_id: str, limit: int = 10) -> dict[str, Any]:
    if not get_artifacts():
        return live_recommender.get_similar_products(product_id=product_id, limit=limit)

    limit = max(1, min(int(limit), 50))
    artifacts = get_artifacts() or {}
    item_neighbors = artifacts.get("item_neighbors", {})
    neighbors = item_neighbors.get(_string(product_id), [])
    if not neighbors:
        return live_recommender.get_similar_products(product_id=product_id, limit=limit)

    db = get_db()
    buffer_neighbors = neighbors[: limit * 3]
    live_products = _fetch_live_products(db, [neighbor_id for neighbor_id, _ in buffer_neighbors])

    similar = []
    for neighbor_id, similarity in buffer_neighbors:
        metadata = live_products.get(_string(neighbor_id))
        if metadata is None:
            continue  # sold out / deactivated / deleted since training
        similar.append(
            _format_product(
                neighbor_id,
                len(similar) + 1,
                float(similarity),
                float(similarity),
                float(similarity),
                float(similarity),
                metadata,
                ["trained_similarity"],
            )
        )
        if len(similar) == limit:
            break

    if not similar:
        return live_recommender.get_similar_products(product_id=product_id, limit=limit)

    live_source = live_recommender._find_product(db, product_id) or _artifact_catalog().get(_string(product_id), {})
    return {
        "success": True,
        "productId": _string(product_id),
        "name": live_source.get("name") or "",
        "count": len(similar),
        "similar_products": similar,
        "artifact_mode": True,
    }


def get_product_interactions(user_id: str) -> dict[str, Any]:
    artifacts = get_artifacts() or {}
    if artifacts and _string(user_id) in artifacts.get("user_histories", {}):
        db = get_db()
        user = live_recommender._find_user(db, user_id)
        if user:
            history = artifacts["user_histories"][_string(user_id)]
            catalog = _artifact_catalog()
            live_products = _fetch_live_products(db, [product_id for product_id, _ in history])
            records = []
            for product_id, score in history:
                # Fall back to the trained snapshot for products that are no
                # longer live-active, so historical entries for sold-out/
                # deleted products still render something instead of blanks.
                metadata = live_products.get(product_id) or catalog.get(product_id, {})
                records.append(
                    {
                        "productId": product_id,
                        "name": metadata.get("name") or "",
                        "category": metadata.get("category") or "",
                        "subCategory": metadata.get("subCategory") or "",
                        "score": float(score),
                        "signals": ["trained_history"],
                    }
                )
            return {
                "success": True,
                "userId": _string(user["_id"]),
                "username": user.get("username") or "",
                "count": len(records),
                "interactions": records,
                "artifact_mode": True,
            }

    return live_recommender.get_product_interactions(user_id=user_id)


def get_product_health() -> dict[str, Any]:
    db_status = check_db_connection()

    artifacts = get_artifacts()
    if not artifacts:
        result = live_recommender.get_product_health()
        result["database"] = db_status
        if db_status.get("status") not in ("ok", "not_configured"):
            result["status"] = "degraded"
        return result

    trained_at = artifacts.get("trained_at")
    artifact_age_hours = None
    if trained_at:
        try:
            # train_model.py stores a naive UTC timestamp (datetime.utcnow().isoformat())
            trained_dt = dt.datetime.fromisoformat(str(trained_at))
            artifact_age_hours = round(
                (dt.datetime.utcnow() - trained_dt).total_seconds() / 3600.0, 2
            )
        except ValueError:
            artifact_age_hours = None

    status = "ok" if db_status.get("status") in ("ok", "not_configured") else "degraded"

    return {
        "status": status,
        "engine": "trained-hybrid-product-recommender",
        "artifact_mode": True,
        "artifact_path": str(ARTIFACT_PATH),
        "model_loaded": True,
        "products": len(artifacts.get("product_ids", [])),
        "users": len(artifacts.get("user_histories", {})),
        "max_limit": 50,
        "trained_at": trained_at,
        "artifact_age_hours": artifact_age_hours,
        "database": db_status,
    }