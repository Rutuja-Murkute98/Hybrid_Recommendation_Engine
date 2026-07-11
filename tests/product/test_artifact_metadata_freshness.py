from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from product_engine import final_recommender


def _fake_artifacts(seed, stale_product_id: str) -> dict:
    product_ids = [str(pid) for pid in seed["product_ids"]]
    product_index = {pid: index for index, pid in enumerate(product_ids)}
    matrix = csr_matrix(np.eye(len(product_ids)))

    return {
        "product_ids": product_ids,
        "product_index": product_index,
        "product_tfidf_matrix": matrix,
        "item_neighbors": {},
        "popularity": {},
        # Deliberately stale/wrong: a real deploy would only see this if the
        # product hasn't been retrained since it changed — name/price/stock
        # here must never reach the response once the live-refresh fix works.
        "product_metadata": {
            stale_product_id: {
                "name": "STALE NAME",
                "category": "Men",
                "price": 99999,
                "totalStock": 0,
            }
        },
        "user_histories": {},
    }


def test_recommendations_use_live_metadata_not_stale_artifact_snapshot(patched_get_db, monkeypatch):
    seed = patched_get_db
    user_id = str(seed["user_id"])
    stale_product_id = str(seed["product_ids"][4])

    monkeypatch.setattr(
        final_recommender, "get_artifacts", lambda: _fake_artifacts(seed, stale_product_id)
    )
    # Force the trained artifact to use this single product as the user's
    # entire history, so it scores highest and is guaranteed to appear.
    monkeypatch.setattr(
        final_recommender, "_artifact_history", lambda uid: {stale_product_id: 1.0}
    )

    result = final_recommender.get_product_recommendations(user_id=user_id, limit=5, include_seen=True)

    assert result["success"] is True
    assert result.get("artifact_mode") is True

    top = result["recommendations"][0]
    assert top["id"] == stale_product_id
    assert top["name"] != "STALE NAME"
    assert top["price"] != 99999
    assert top["totalStock"] > 0


def test_sold_out_product_is_skipped_even_if_top_scored(patched_get_db, monkeypatch):
    seed = patched_get_db
    user_id = str(seed["user_id"])
    stale_product_id = str(seed["product_ids"][4])

    monkeypatch.setattr(
        final_recommender, "get_artifacts", lambda: _fake_artifacts(seed, stale_product_id)
    )
    monkeypatch.setattr(
        final_recommender, "_artifact_history", lambda uid: {stale_product_id: 1.0}
    )

    db = seed["db"]
    db.products.update_one({"_id": seed["product_ids"][4]}, {"$set": {"totalStock": 0}})

    result = final_recommender.get_product_recommendations(user_id=user_id, limit=5, include_seen=True)

    ids = [item["id"] for item in result["recommendations"]]
    assert stale_product_id not in ids
