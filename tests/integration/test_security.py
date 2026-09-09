"""
Access control.

The gate has to protect the API without locking anyone out of the page that
asks for the code. An earlier version gated every path, so a deployed
container answered `GET /` with a 401 JSON body — the app could not be loaded
at all, and there was no way to enter a code. `test_app_shell_is_never_gated`
is that bug.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

CODE = "test-access-code-1234"


@pytest.fixture()
def gated_client(monkeypatch):
    """An app instance built with an access code configured."""
    monkeypatch.setenv("ACCESS_CODE", CODE)
    monkeypatch.setenv("ENV", "development")

    import app.security as security
    import app.main as main

    importlib.reload(security)
    importlib.reload(main)

    with TestClient(main.app) as client:
        yield client

    # Leave the modules as the rest of the suite expects them.
    monkeypatch.delenv("ACCESS_CODE", raising=False)
    importlib.reload(security)
    importlib.reload(main)


# ------------------------------------------------------------------ gating
def test_api_requires_the_code(gated_client):
    assert gated_client.get("/api/meta").status_code == 401


def test_api_rejects_a_wrong_code(gated_client):
    response = gated_client.get("/api/meta", headers={"X-Access-Code": "nope"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "access_denied"


def test_api_accepts_the_right_code(gated_client):
    assert gated_client.get("/api/meta", headers={"X-Access-Code": CODE}).status_code == 200


def test_code_also_accepted_as_a_query_parameter(gated_client):
    """Supports sharing a single pre-authorised link."""
    assert gated_client.get(f"/api/meta?access_code={CODE}").status_code == 200


def test_chat_is_gated(gated_client):
    response = gated_client.post(
        "/api/chat", json={"disease": "hypertension", "age": 50, "message": "metformin?"}
    )
    assert response.status_code == 401


# ------------------------------------------------------------------ public
def test_health_is_public(gated_client):
    """Orchestrators probe liveness before any secret is available."""
    assert gated_client.get("/api/health").status_code == 200


def test_config_is_public_and_leaks_nothing(gated_client):
    response = gated_client.get("/api/config")
    assert response.status_code == 200

    body = response.json()
    assert body["access_required"] is True
    # Only the two fields the gate screen needs — no versions, model names or
    # configuration that would help someone probing the deployment.
    assert set(body) == {"access_required", "access_header"}


def test_app_shell_is_never_gated(gated_client):
    """
    The static shell must load without a code, because it IS the screen where
    the code is entered. Gating it makes the deployment unusable.
    """
    for path in ("/", "/interactions", "/some/deep/link"):
        response = gated_client.get(path)
        assert response.status_code != 401, f"{path} was gated — nobody can reach the code prompt"


def test_unknown_api_path_is_json_not_the_html_shell(gated_client):
    """A mistyped endpoint must not return HTML and confuse the client parser."""
    response = gated_client.get("/api/does-not-exist", headers={"X-Access-Code": CODE})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# ------------------------------------------------------------- startup guard
def test_production_without_a_code_refuses_to_start(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.delenv("ACCESS_CODE", raising=False)

    import app.security as security

    importlib.reload(security)
    with pytest.raises(RuntimeError, match="ACCESS_CODE"):
        security.verify_startup_configuration()

    monkeypatch.setenv("ENV", "development")
    importlib.reload(security)


def test_a_short_code_is_refused(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("ACCESS_CODE", "abc")

    import app.security as security

    importlib.reload(security)
    with pytest.raises(RuntimeError, match="characters"):
        security.verify_startup_configuration()

    monkeypatch.setenv("ENV", "development")
    monkeypatch.delenv("ACCESS_CODE", raising=False)
    importlib.reload(security)
