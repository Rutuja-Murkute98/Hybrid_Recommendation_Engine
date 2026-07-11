from __future__ import annotations

import pytest
from flask import Flask


@pytest.fixture
def client(monkeypatch):
    from extensions import limiter
    from product_engine.routes import product_bp
    import product_engine.routes as routes_module

    app = Flask(__name__)
    app.config["TESTING"] = True
    limiter.init_app(app)
    app.register_blueprint(product_bp)
    monkeypatch.setattr(routes_module, "reload_artifacts", lambda: True)
    return app.test_client()


def test_no_header_is_unauthorized_when_key_configured(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "correct-horse-battery-staple")

    response = client.post("/admin/reload-artifacts")

    assert response.status_code == 401


def test_wrong_key_is_unauthorized(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "correct-horse-battery-staple")

    response = client.post("/admin/reload-artifacts", headers={"X-Admin-Key": "wrong"})

    assert response.status_code == 401


def test_correct_key_is_authorized(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "correct-horse-battery-staple")

    response = client.post(
        "/admin/reload-artifacts", headers={"X-Admin-Key": "correct-horse-battery-staple"}
    )

    assert response.status_code == 200


def test_fails_closed_when_no_key_configured_server_side(client, monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)

    response = client.post("/admin/reload-artifacts", headers={"X-Admin-Key": "anything"})

    assert response.status_code == 503
