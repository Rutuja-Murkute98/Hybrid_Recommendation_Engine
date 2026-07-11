from __future__ import annotations

from product_engine import recommender as rec


def test_unknown_user_returns_not_found(patched_get_db):
    result = rec.get_product_recommendations(user_id="does-not-exist", limit=5)
    assert result["success"] is False
    assert result["status_code"] == 404


def test_known_user_excludes_seen_products_by_default(patched_get_db):
    seed = patched_get_db
    user_id = str(seed["user_id"])
    ordered_product_id = str(seed["product_ids"][0])

    result = rec.get_product_recommendations(user_id=user_id, limit=10)
    assert result["success"] is True
    returned_ids = {item["id"] for item in result["recommendations"]}
    assert ordered_product_id not in returned_ids


def test_include_seen_true_can_return_seen_products(patched_get_db):
    seed = patched_get_db
    user_id = str(seed["user_id"])
    ordered_product_id = str(seed["product_ids"][0])

    result = rec.get_product_recommendations(user_id=user_id, limit=10, include_seen=True)
    returned_ids = {item["id"] for item in result["recommendations"]}
    assert ordered_product_id in returned_ids


def test_category_filter_narrows_results(patched_get_db):
    seed = patched_get_db
    user_id = str(seed["user_id"])

    result = rec.get_product_recommendations(user_id=user_id, limit=10, category="Men")
    categories = {item["category"] for item in result["recommendations"]}
    assert categories <= {"Men"}
    assert len(result["recommendations"]) > 0


def test_cold_start_user_has_no_interactions(patched_get_db, monkeypatch):
    seed = patched_get_db
    db = seed["db"]

    from bson import ObjectId

    cold_user_id = ObjectId()
    db.users.insert_one(
        {
            "_id": cold_user_id,
            "username": "cold_user",
            "email": "cold@example.com",
            "gender": "",
            "shoppingPreferences": {},
            "savedProducts": [],
            "likedProducts": [],
            "savedBits": [],
            "isAdmin": False,
        }
    )

    result = rec.get_product_recommendations(user_id=str(cold_user_id), limit=5)
    assert result["success"] is True
    assert result["strategy"] == "zatch-product-cold-start"


def test_similar_products_excludes_source(patched_get_db):
    seed = patched_get_db
    source_id = str(seed["product_ids"][0])

    result = rec.get_similar_products(product_id=source_id, limit=10)
    assert result["success"] is True
    returned_ids = {item["id"] for item in result["similar_products"]}
    assert source_id not in returned_ids


def test_product_interactions_batches_product_lookup(patched_get_db):
    seed = patched_get_db
    user_id = str(seed["user_id"])

    result = rec.get_product_interactions(user_id=user_id)
    assert result["success"] is True
    # ordered + carted + saved + liked = 4 distinct interacted products
    assert result["count"] == 4
    names = {record["name"] for record in result["interactions"]}
    assert "" not in names  # batched lookup found metadata for every interacted product
