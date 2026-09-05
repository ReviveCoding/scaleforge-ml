from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    latency_ms: float


class UnavailableGenerator:
    ready = False

    def __call__(self, prompt: str, max_new_tokens: int) -> str:
        raise RuntimeError("generator is not configured")


Generator = Callable[[str, int], str]


def create_app(generator: Generator | None = None) -> FastAPI:
    runtime = generator or UnavailableGenerator()
    app = FastAPI(title="ScaleForge-ML", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True, "model_ready": bool(getattr(runtime, "ready", generator is not None))}

    @app.post("/generate")
    def generate(request: GenerateRequest) -> GenerateResponse:
        started = perf_counter()
        try:
            text = runtime(request.prompt, request.max_new_tokens)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return GenerateResponse(
            request_id=request.request_id,
            text=text,
            latency_ms=(perf_counter() - started) * 1000,
        )

    @app.get("/metrics")
    def metrics() -> dict[str, str]:
        return {"status": "instrumentation_pending"}

    return app


app = create_app()
