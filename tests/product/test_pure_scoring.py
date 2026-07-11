from __future__ import annotations

from bson import ObjectId

from product_engine import recommender as rec


def test_discount_score_no_discount():
    assert rec._discount_score({"price": 100, "discountedPrice": 100}) == 0.0


def test_discount_score_half_off():
    assert rec._discount_score({"price": 100, "discountedPrice": 50}) == 0.5


def test_discount_score_missing_price():
    assert rec._discount_score({}) == 0.0


def test_stock_score_tiers():
    assert rec._stock_score({"totalStock": 0}) == 0.0
    assert rec._stock_score({"totalStock": 2}) == 0.45
    assert rec._stock_score({"totalStock": 8}) == 0.75
    assert rec._stock_score({"totalStock": 50}) == 1.0


def test_popularity_score_zero_max_is_safe():
    assert rec._popularity_score({"viewCount": 5}, max_popularity=0) == 0.0


def test_popularity_score_capped_at_one():
    product = {"viewCount": 1000, "likeCount": 1000, "saveCount": 1000, "shareCount": 1000}
    assert rec._popularity_score(product, max_popularity=1.0) == 1.0


def test_price_affinity_falls_back_to_user_preference_range():
    user = {"shoppingPreferences": {"priceRange": {"min": 10, "max": 50}}}
    assert rec._price_affinity(30, [], user) == 1.0


def test_price_affinity_with_no_signal_at_all():
    assert rec._price_affinity(30, [], {}) == 0.5


def test_clamp_limit_defaults_to_ten():
    assert rec._clamp_limit(None) == 10


def test_clamp_limit_rejects_negative():
    assert rec._clamp_limit(-5) == 1


def test_clamp_limit_caps_at_max():
    assert rec._clamp_limit(9999) == rec.MAX_PRODUCT_LIMIT


def test_clean_converts_objectid_and_datetime():
    import datetime as dt

    oid = ObjectId()
    now = dt.datetime(2026, 1, 1)
    cleaned = rec._clean({"id": oid, "when": now, "nested": [oid]})
    assert cleaned == {"id": str(oid), "when": now.isoformat(), "nested": [str(oid)]}
