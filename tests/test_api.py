from fastapi.testclient import TestClient

from api.app import create_app


def test_health_and_generation() -> None:
    client = TestClient(create_app(lambda prompt, limit: f"{prompt}:{limit}"))
    assert client.get("/health").json() == {"ok": True, "model_ready": True}
    response = client.post(
        "/generate", json={"request_id": "r1", "prompt": "hello", "max_new_tokens": 3}
    )
    assert response.status_code == 200
    assert response.json()["text"] == "hello:3"


def test_unavailable_generator_returns_503() -> None:
    client = TestClient(create_app())
    assert client.post("/generate", json={"request_id": "r", "prompt": "x"}).status_code == 503
