from __future__ import annotations

from product_engine import zatch_mongo_recommender as zatch


def test_get_zatch_reel_recommendations_known_user(patched_get_db):
    seed = patched_get_db

    result = zatch.get_zatch_reel_recommendations(user_id=str(seed["user_id"]), limit=5)

    assert result["success"] is True
    assert result["count"] >= 0
    assert result["count"] <= 5


def test_get_zatch_reel_recommendations_unknown_user_returns_404(patched_get_db):
    result = zatch.get_zatch_reel_recommendations(user_id="does-not-exist", limit=5)

    assert result["success"] is False
    assert result["status_code"] == 404


def test_get_trending_reels_shape(patched_get_db):
    result = zatch.get_trending_reels(limit=10)

    assert result["success"] is True
    assert result["strategy"] == "zatch-trending"
    assert isinstance(result["recommendations"], list)
    assert result["count"] == len(result["recommendations"])


def test_get_reel_status_for_existing_bit(patched_get_db):
    seed = patched_get_db
    bit = seed["db"].bits.find_one({})

    status = zatch.get_reel_status(str(bit["_id"]))

    assert status["exists"] is True
    assert status["type"] == "bit"


def test_get_reel_status_for_unknown_id(patched_get_db):
    status = zatch.get_reel_status("does-not-exist")

    assert status["exists"] is False
    assert status["type"] is None


def test_get_reel_user_status_for_existing_user(patched_get_db):
    seed = patched_get_db

    status = zatch.get_reel_user_status(str(seed["user_id"]))

    assert status["exists"] is True
    assert status["recommendation_mode"] in {"personalized", "cold_start"}


def test_get_reel_user_status_for_unknown_user(patched_get_db):
    status = zatch.get_reel_user_status("does-not-exist")

    assert status["exists"] is False
    assert status["recommendation_mode"] == "cold_start"
