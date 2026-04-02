"""Unit tests for the empire_route decorator in web/routes/api.py."""

from __future__ import annotations

import json

import pytest
from flask import Flask

from web.routes.api import api_bp, empire_route


@pytest.fixture()
def app():
    """Create a minimal Flask app with the API blueprint."""
    app = Flask(__name__)
    app.config["EMPIRE_ID"] = "test-empire-123"
    app.config["TESTING"] = True
    app.register_blueprint(api_bp, url_prefix="/api")
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def test_empire_route_injects_empire_id(app):
    """Decorated functions receive empire_id as first arg."""
    received = {}

    @app.route("/test-inject")
    @empire_route
    def handler(empire_id):
        received["empire_id"] = empire_id
        return {"ok": True}

    with app.test_client() as c:
        resp = c.get("/test-inject")
        assert resp.status_code == 200
        assert received["empire_id"] == "test-empire-123"


def test_empire_route_jsonifies_dict(app):
    """Returned dicts are automatically jsonified."""
    @app.route("/test-dict")
    @empire_route
    def handler(empire_id):
        return {"name": "test", "count": 42}

    with app.test_client() as c:
        resp = c.get("/test-dict")
        assert resp.status_code == 200
        assert resp.content_type == "application/json"
        data = json.loads(resp.data)
        assert data["name"] == "test"
        assert data["count"] == 42


def test_empire_route_jsonifies_list(app):
    """Returned lists are automatically jsonified."""
    @app.route("/test-list")
    @empire_route
    def handler(empire_id):
        return [1, 2, 3]

    with app.test_client() as c:
        resp = c.get("/test-list")
        assert resp.status_code == 200
        assert json.loads(resp.data) == [1, 2, 3]


def test_empire_route_handles_tuple_status(app):
    """Returned (data, status_code) tuples set the HTTP status."""
    @app.route("/test-created", methods=["POST"])
    @empire_route
    def handler(empire_id):
        return {"id": "new-123"}, 201

    with app.test_client() as c:
        resp = c.post("/test-created")
        assert resp.status_code == 201
        assert json.loads(resp.data)["id"] == "new-123"


def test_empire_route_catches_exceptions(app):
    """Exceptions are caught and returned as 500 JSON errors."""
    @app.route("/test-error")
    @empire_route
    def handler(empire_id):
        raise ValueError("something broke")

    with app.test_client() as c:
        resp = c.get("/test-error")
        assert resp.status_code == 500
        data = json.loads(resp.data)
        assert "error" in data


def test_empire_route_passes_through_flask_response(app):
    """Flask Response objects pass through without double-wrapping."""
    from flask import Response

    @app.route("/test-response")
    @empire_route
    def handler(empire_id):
        return Response("<h1>hi</h1>", mimetype="text/html")

    with app.test_client() as c:
        resp = c.get("/test-response")
        assert resp.status_code == 200
        assert resp.content_type == "text/html; charset=utf-8"
        assert b"<h1>hi</h1>" in resp.data


def test_empire_route_with_path_params(app):
    """Path parameters are passed correctly alongside empire_id."""
    @app.route("/test-param/<item_id>")
    @empire_route
    def handler(empire_id, item_id):
        return {"empire": empire_id, "item": item_id}

    with app.test_client() as c:
        resp = c.get("/test-param/abc-456")
        data = json.loads(resp.data)
        assert data["empire"] == "test-empire-123"
        assert data["item"] == "abc-456"
