from __future__ import annotations

import asyncio
from typing import Any, cast

from fastapi.testclient import TestClient

from api.hf_server import DONE, GenerationJob, HfStaticBatchEngine, create_hf_app


class FakeEngine:
    ready = True

    def metrics(self) -> dict[str, int | bool]:
        return {"ready": True, "requests_total": 1}

    def submit(self, prompt: str, max_tokens: int) -> GenerationJob:
        job = GenerationJob(prompt, max_tokens, "cmpl-test", asyncio.get_running_loop())
        job.output.put_nowait(
            {
                "id": "cmpl-test",
                "object": "text_completion",
                "choices": [{"index": 0, "text": "42", "finish_reason": None}],
            }
        )
        job.output.put_nowait(DONE)
        return job


def test_hf_openai_stream_contract() -> None:
    engine = cast(HfStaticBatchEngine, cast(Any, FakeEngine()))
    client = TestClient(create_hf_app(engine))
    assert client.get("/health").json() == {"ok": True, "model_ready": True}
    response = client.post(
        "/v1/completions",
        json={"model": "test", "prompt": "answer", "temperature": 0, "stream": True},
    )
    assert response.status_code == 200
    assert '"text": "42"' in response.text
    assert "data: [DONE]" in response.text


def test_hf_server_rejects_stochastic_request() -> None:
    engine = cast(HfStaticBatchEngine, cast(Any, FakeEngine()))
    client = TestClient(create_hf_app(engine))
    response = client.post(
        "/v1/completions",
        json={"model": "test", "prompt": "answer", "temperature": 0.7, "stream": True},
    )
    assert response.status_code == 400
