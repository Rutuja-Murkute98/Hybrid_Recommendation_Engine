from __future__ import annotations

import os
import sys
from pathlib import Path

# Must run before `product_engine.zatch_mongo_recommender` (and anything
# importing it) is loaded: python-dotenv's load_dotenv() won't override an
# already-set env var, so this keeps every test in this suite off the real
# Atlas cluster and real API keys even when a live .env is present in the repo
# root. Setting these to "" (not popping) is deliberate: popping would leave
# them unset, and load_dotenv() would then fill them back in from .env the
# moment zatch_mongo_recommender is first imported — an empty string still
# counts as "already set" to load_dotenv, so it's the only way to keep .env
# from leaking into the test session at all.
os.environ["MONGO_URI"] = "mongodb://localhost:1/?serverSelectionTimeoutMS=200"
os.environ["MONGO_TIMEOUT_MS"] = "200"
os.environ["ADMIN_API_KEY"] = ""
os.environ["API_KEY"] = ""

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mongomock
import pytest
from bson import ObjectId


@pytest.fixture
def mongo_seed():
    """A mongomock database seeded with one user and eight products (4 Women / 4 Men)."""
    client = mongomock.MongoClient()
    db = client["zatch_test"]

    user_id = ObjectId()
    product_ids = [ObjectId() for _ in range(8)]
    categories = ["Women", "Women", "Women", "Women", "Men", "Men", "Men", "Men"]

    db.users.insert_one(
        {
            "_id": user_id,
            "username": "test_user",
            "email": "test@example.com",
            "gender": "female",
            "shoppingPreferences": {"categories": ["women"], "priceRange": {"min": 10, "max": 100}},
            "savedProducts": [str(product_ids[2])],
            "likedProducts": [str(product_ids[3])],
            "savedBits": [],
            "isAdmin": False,
        }
    )

    for idx, pid in enumerate(product_ids):
        db.products.insert_one(
            {
                "_id": pid,
                "sellerId": ObjectId(),
                "name": f"Product {idx}",
                "description": "a nice product",
                "category": categories[idx],
                "subCategory": "tops" if categories[idx] == "Women" else "shirts",
                "price": 50 + idx,
                "discountedPrice": 40 + idx,
                "images": [],
                "viewCount": 10 * idx,
                "saveCount": idx,
                "likeCount": idx,
                "shareCount": 0,
                "totalStock": 5,
                "status": "active",
                "isSold": False,
                "isTopPick": idx == 0,
                "tags": [],
                "searchKeywords": [],
            }
        )

    # product_ids[0] ordered, product_ids[1] carted -> both count as "seen"
    db.orders.insert_one({"buyerId": user_id, "items": [{"productId": str(product_ids[0])}]})
    db.carts.insert_one({"user": user_id, "items": [{"productId": str(product_ids[1])}]})

    bit_id = ObjectId()
    db.bits.insert_one({"_id": bit_id, "products": [str(product_ids[4])], "likes": []})
    db.livesessions.insert_one({"_id": ObjectId(), "status": "live", "viewers": [], "products": []})

    return {"db": db, "user_id": user_id, "product_ids": product_ids}


@pytest.fixture
def patched_get_db(monkeypatch, mongo_seed):
    """Points every module's get_db() at the seeded mongomock db and clears the catalog cache."""
    from product_engine import zatch_mongo_recommender
    from product_engine import final_recommender
    from product_engine import recommender as live_recommender

    db = mongo_seed["db"]
    monkeypatch.setattr(zatch_mongo_recommender, "get_db", lambda: db)
    monkeypatch.setattr(live_recommender, "get_db", lambda: db)
    monkeypatch.setattr(final_recommender, "get_db", lambda: db)

    # All three caches key on something other than the db instance itself
    # (a "db is a process-wide singleton" assumption that's only true in
    # production, not across per-test mongomock fixtures) — clear them all
    # so one test's seeded data can't leak into another's assertions.
    live_recommender._PRODUCT_CATALOG_CACHE.clear()
    zatch_mongo_recommender._bits_cache.clear()
    zatch_mongo_recommender._sessions_cache.clear()

    return mongo_seed
