from fastapi.testclient import TestClient

from legal_rag.api import app, service


def test_local_ui_serves_without_model():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "German Legal RAG" in response.text
    assert "Ask documents" in response.text


def test_missing_index_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_STATE_DIR", str(tmp_path / "empty"))
    service.cache_clear()
    try:
        client = TestClient(app)
        assert client.get("/health").json()["status"] == "not_ready"
        assert client.post("/ask", json={"query": "Was steht im Dokument?"}).status_code == 400
    finally:
        service.cache_clear()


def test_invalid_request_blocked_before_running_models():
    response = TestClient(app).post("/ask", json={"query": ""})
    assert response.status_code == 422


def test_api_question_defaults_to_pilot_best_dense():
    from legal_rag.api import Question

    assert Question(query="Was regelt § 1 BGB?").mode == "dense"
