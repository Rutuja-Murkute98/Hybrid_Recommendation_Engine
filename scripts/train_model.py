from __future__ import annotations

import datetime as dt
import logging
import os
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_engine.zatch_mongo_recommender import ZatchConfigError, get_db
from product_engine import recommender as live_recommender


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = ROOT / "product_engine" / "models"
MODEL_PATH = MODEL_DIR / "product_recommendation_artifacts.joblib"


def _product_text(product: dict) -> str:
    parts = [
        product.get("name"),
        product.get("description"),
        product.get("category"),
        product.get("subCategory"),
        " ".join(str(tag) for tag in product.get("tags") or []),
        " ".join(str(keyword) for keyword in product.get("searchKeywords") or []),
    ]
    return " ".join(str(part) for part in parts if part).strip()


def _build_item_neighbors(product_matrix, product_ids: list[str], top_k: int = 50) -> dict[str, list[tuple[str, float]]]:
    neighbor_count = min(top_k + 1, len(product_ids))
    if neighbor_count <= 1:
        return {product_id: [] for product_id in product_ids}

    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=neighbor_count)
    model.fit(product_matrix)
    distances, indices = model.kneighbors(product_matrix)

    neighbors: dict[str, list[tuple[str, float]]] = {}
    for row_index, product_id in enumerate(product_ids):
        row_neighbors: list[tuple[str, float]] = []
        for neighbor_distance, neighbor_index in zip(distances[row_index][1:], indices[row_index][1:]):
            row_neighbors.append((product_ids[int(neighbor_index)], float(max(0.0, 1.0 - neighbor_distance))))
        neighbors[product_id] = row_neighbors
    return neighbors


def _build_user_histories(db) -> dict[str, list[tuple[str, float]]]:
    histories: dict[str, list[tuple[str, float]]] = {}
    cursor = db.users.find({"isAdmin": {"$ne": True}}, {"_id": 1, "username": 1, "email": 1, "gender": 1, "shoppingPreferences": 1, "savedProducts": 1, "likedProducts": 1, "savedBits": 1, "following": 1})

    for user in cursor:
        interactions = live_recommender._build_user_interactions(db, user)
        ordered = sorted(interactions.items(), key=lambda item: item[1]["score"], reverse=True)
        histories[str(user["_id"])] = [(product_id, float(record["score"])) for product_id, record in ordered[:100]]

    return histories


def _backup_existing_artifact(keep: int = 3) -> None:
    if not MODEL_PATH.exists():
        return
    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_path = MODEL_DIR / f"product_recommendation_artifacts.{timestamp}.joblib.bak"
    backup_path.write_bytes(MODEL_PATH.read_bytes())

    backups = sorted(MODEL_DIR.glob("product_recommendation_artifacts.*.joblib.bak"))
    for stale_backup in backups[:-keep]:
        stale_backup.unlink(missing_ok=True)


def _build_popularity(histories: dict[str, list[tuple[str, float]]]) -> dict[str, float]:
    popularity: dict[str, float] = {}
    for history in histories.values():
        for product_id, score in history:
            popularity[product_id] = popularity.get(product_id, 0.0) + float(score)
    return popularity


def main() -> int:
    try:
        db = get_db()
    except ZatchConfigError as exc:
        print(f"Cannot train without MongoDB: {exc}")
        return 1

    products = live_recommender._load_products(db)
    if not products:
        print("No active products found. Aborting.")
        return 1

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    product_ids = [str(product["_id"]) for product in products]
    product_index = {product_id: index for index, product_id in enumerate(product_ids)}
    product_metadata = {
        product_id: {
            "name": product.get("name") or "",
            "description": product.get("description") or "",
            "category": product.get("category") or "",
            "subCategory": product.get("subCategory") or "",
            "sellerId": str(product.get("sellerId") or ""),
            "price": product.get("price") or 0,
            "discountedPrice": product.get("discountedPrice") or product.get("price") or 0,
            "totalStock": product.get("totalStock") or 0,
            "status": product.get("status") or "",
            "images": product.get("images") or [],
            "viewCount": product.get("viewCount") or 0,
            "likeCount": product.get("likeCount") or 0,
            "saveCount": product.get("saveCount") or 0,
            "isTopPick": bool(product.get("isTopPick")),
            "tags": product.get("tags") or [],
            "searchKeywords": product.get("searchKeywords") or [],
        }
        for product_id, product in zip(product_ids, products)
    }

    corpus = [_product_text(product) for product in products]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2))
    product_tfidf_matrix = vectorizer.fit_transform(corpus)

    item_neighbors = _build_item_neighbors(product_tfidf_matrix, product_ids, top_k=50)
    user_histories = _build_user_histories(db)
    popularity = _build_popularity(user_histories)

    artifacts = {
        "trained_at": dt.datetime.utcnow().isoformat(),
        "product_ids": product_ids,
        "product_index": product_index,
        "product_metadata": product_metadata,
        "product_tfidf_matrix": product_tfidf_matrix,
        "tfidf_vectorizer": vectorizer,
        "item_neighbors": item_neighbors,
        "user_histories": user_histories,
        "popularity": popularity,
    }

    _backup_existing_artifact()

    tmp_path = MODEL_PATH.with_suffix(".joblib.tmp")
    joblib.dump(artifacts, tmp_path)
    os.replace(tmp_path, MODEL_PATH)  # atomic swap: never leave a half-written artifact

    print(f"Saved trained product artifacts to {MODEL_PATH}")
    print(f"Products: {len(product_ids)} | Users: {len(user_histories)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
