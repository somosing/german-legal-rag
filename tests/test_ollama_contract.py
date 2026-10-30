from legal_rag.generation import StructuredAnswer
from legal_rag.ollama import Ollama


def test_structured_chat_passes_schema_to_ollama(monkeypatch):
    backend = Ollama("http://127.0.0.1:11434", "embeddinggemma", "qwen3:4b-instruct")
    observed = {}

    def fake_post(route, payload, timeout=120.0):
        observed.update(payload)
        assert route == "/api/chat"
        return {
            "message": {
                "content": '{"status":"insufficient_evidence","answer":"","citations":[]}'
            }
        }

    monkeypatch.setattr(backend, "_post", fake_post)
    reply = backend.chat("System", "Query", StructuredAnswer.model_json_schema())
    assert "insufficient_evidence" in reply
    assert observed["format"]["type"] == "object"
    assert observed["options"]["temperature"] == 0
    assert observed["stream"] is False
