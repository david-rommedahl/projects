"""Smoke tests for the FastAPI app skeleton."""

from fastapi.testclient import TestClient


def test_ping(client: TestClient) -> None:
    """The health check returns pong."""
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "pong"}
