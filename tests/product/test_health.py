from __future__ import annotations

import pytest
from flask import Flask


@pytest.fixture
def client():
    # Deliberately builds a minimal app around product_bp only, NOT the real
    # app.py — importing app.py would trigger reel_engine's full ~930MB model
    # load, which this fast test suite must not depend on.
    from extensions import limiter
    from product_engine.routes import product_bp
    import product_engine.routes as routes_module

    app = Flask(__name__)
    app.config["TESTING"] = True
    limiter.init_app(app)
    app.register_blueprint(product_bp)
    return routes_module, app.test_client()


def test_product_health_returns_200_when_engine_reports_ok(client, monkeypatch):
    routes_module, test_client = client
    monkeypatch.setattr(routes_module, "get_product_health", lambda: {"status": "ok"})

    response = test_client.get("/product-health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "success"


def test_product_health_returns_503_when_engine_reports_degraded(client, monkeypatch):
    routes_module, test_client = client
    monkeypatch.setattr(
        routes_module,
        "get_product_health",
        lambda: {"status": "degraded", "database": {"status": "error", "message": "boom"}},
    )

    response = test_client.get("/product-health")

    assert response.status_code == 503
    assert response.get_json()["status"] == "degraded"
