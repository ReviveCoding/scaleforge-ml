from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter

import structlog
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


@dataclass
class ServiceMetrics:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    lock: Lock = field(default_factory=Lock)

    def record(self, *, latency_ms: float, success: bool) -> None:
        with self.lock:
            self.requests += 1
            self.successes += int(success)
            self.failures += int(not success)
            self.total_latency_ms += latency_ms

    def snapshot(self) -> dict[str, int | float]:
        with self.lock:
            return {
                "requests_total": self.requests,
                "successes_total": self.successes,
                "failures_total": self.failures,
                "total_latency_ms": self.total_latency_ms,
            }


def create_app(generator: Generator | None = None) -> FastAPI:
    runtime = generator or UnavailableGenerator()
    counters = ServiceMetrics()
    logger = structlog.get_logger("scaleforge.serving")
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
            latency_ms = (perf_counter() - started) * 1000
            counters.record(latency_ms=latency_ms, success=False)
            logger.error(
                "generation_failed",
                request_id=request.request_id,
                latency_ms=latency_ms,
                failure_type=type(error).__name__,
            )
            raise HTTPException(status_code=503, detail=str(error)) from error
        latency_ms = (perf_counter() - started) * 1000
        counters.record(latency_ms=latency_ms, success=True)
        logger.info(
            "generation_completed",
            request_id=request.request_id,
            latency_ms=latency_ms,
            max_new_tokens=request.max_new_tokens,
        )
        return GenerateResponse(request_id=request.request_id, text=text, latency_ms=latency_ms)

    @app.get("/metrics")
    def metrics() -> dict[str, int | float]:
        return counters.snapshot()

    return app


app = create_app()
