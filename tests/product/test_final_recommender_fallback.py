from __future__ import annotations

from product_engine import final_recommender


def test_no_artifacts_falls_back_to_live_recommendations(patched_get_db, monkeypatch):
    seed = patched_get_db
    monkeypatch.setattr(final_recommender, "get_artifacts", lambda: None)

    result = final_recommender.get_product_recommendations(user_id=str(seed["user_id"]), limit=5)

    assert result["success"] is True
    assert result.get("artifact_mode") is not True


def test_empty_product_ids_falls_back_to_live_recommendations(patched_get_db, monkeypatch):
    seed = patched_get_db
    monkeypatch.setattr(
        final_recommender,
        "get_artifacts",
        lambda: {"product_ids": [], "product_tfidf_matrix": None},
    )

    result = final_recommender.get_product_recommendations(user_id=str(seed["user_id"]), limit=5)

    assert result["success"] is True
    assert result.get("artifact_mode") is not True


def test_no_neighbors_falls_back_for_similar_products(patched_get_db, monkeypatch):
    seed = patched_get_db
    monkeypatch.setattr(
        final_recommender,
        "get_artifacts",
        lambda: {"item_neighbors": {}, "product_metadata": {}},
    )

    source_id = str(seed["product_ids"][0])
    result = final_recommender.get_similar_products(product_id=source_id, limit=5)

    assert result["success"] is True
    assert result.get("artifact_mode") is not True


def test_no_artifacts_falls_back_for_health(patched_get_db, monkeypatch):
    monkeypatch.setattr(final_recommender, "get_artifacts", lambda: None)

    result = final_recommender.get_product_health()

    assert result["engine"] == "zatch-product-signal-hybrid"
