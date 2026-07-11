from __future__ import annotations

import pytest
from flask import Flask


@pytest.fixture
def client():
    from extensions import limiter
    from product_engine.routes import product_bp

    app = Flask(__name__)
    app.config["TESTING"] = True
    limiter.init_app(app)
    app.register_blueprint(product_bp)
    return app.test_client()


def test_unauthenticated_when_api_key_not_configured(client, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)

    response = client.get("/product-health")

    # No API_KEY configured server-side -> permissive by design, request proceeds
    # to the route itself (whatever status that route would otherwise return).
    assert response.status_code != 401


def test_missing_header_is_unauthorized_when_key_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "correct-horse-battery-staple")

    response = client.get("/product-health")

    assert response.status_code == 401


def test_wrong_header_is_unauthorized_when_key_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "correct-horse-battery-staple")

    response = client.get("/product-health", headers={"X-API-Key": "wrong"})

    assert response.status_code == 401


def test_correct_header_is_authorized(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "correct-horse-battery-staple")

    response = client.get("/product-health", headers={"X-API-Key": "correct-horse-battery-staple"})

    assert response.status_code != 401


def test_admin_route_is_not_gated_by_api_key(client, monkeypatch):
    # /admin/reload-artifacts keeps its own separate ADMIN_API_KEY check —
    # API_KEY must never be required (or accepted) there too.
    monkeypatch.setenv("API_KEY", "correct-horse-battery-staple")
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)

    response = client.post("/admin/reload-artifacts", headers={"X-API-Key": "correct-horse-battery-staple"})

    # Fails closed on the admin check (no ADMIN_API_KEY configured) -> 503,
    # not 401 from the API_KEY gate.
    assert response.status_code == 503
