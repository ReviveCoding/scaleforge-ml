from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import structlog
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DONE = object()


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = 0.0
    stream: bool = True


@dataclass
class GenerationJob:
    prompt: str
    max_tokens: int
    request_id: str
    loop: asyncio.AbstractEventLoop
    output: asyncio.Queue[dict[str, Any] | object] = field(default_factory=asyncio.Queue)


class HfStaticBatchEngine:
    """Competent BF16 HF baseline with short-window static request batching."""

    def __init__(self, *, max_batch_size: int = 16, batch_wait_ms: float = 5.0) -> None:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 CUDA is required by the serving baseline")
        self.max_batch_size = max_batch_size
        self.batch_wait_ms = batch_wait_ms
        cache = Path("artifacts/cache/huggingface")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=cache
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        loaded_model: Any = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=cache, dtype=torch.bfloat16
        )
        self.model = loaded_model.to("cuda")
        self.model.eval()
        self.jobs: queue.Queue[GenerationJob] = queue.Queue()
        self.requests = 0
        self.failures = 0
        self.generated_tokens = 0
        self._metrics_lock = threading.Lock()
        self._worker = threading.Thread(target=self._work, daemon=True, name="hf-static-batcher")
        self._worker.start()
        self.ready = True

    def submit(self, prompt: str, max_tokens: int) -> GenerationJob:
        job = GenerationJob(
            prompt, max_tokens, f"cmpl-{uuid.uuid4().hex}", asyncio.get_running_loop()
        )
        with self._metrics_lock:
            self.requests += 1
        self.jobs.put(job)
        return job

    def metrics(self) -> dict[str, int | bool]:
        with self._metrics_lock:
            return {
                "ready": self.ready,
                "requests_total": self.requests,
                "failures_total": self.failures,
                "generated_tokens_total": self.generated_tokens,
                "queue_depth": self.jobs.qsize(),
            }

    @staticmethod
    def _send(job: GenerationJob, item: dict[str, Any] | object) -> None:
        job.loop.call_soon_threadsafe(job.output.put_nowait, item)

    def _work(self) -> None:
        logger = structlog.get_logger("scaleforge.hf_batcher")
        while True:
            first = self.jobs.get()
            batch = [first]
            deadline = time.perf_counter() + self.batch_wait_ms / 1000
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.jobs.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                self._generate(batch)
            except Exception as error:
                logger.exception(
                    "hf_batch_failed", batch_size=len(batch), error_type=type(error).__name__
                )
                with self._metrics_lock:
                    self.failures += len(batch)
                for job in batch:
                    self._send(job, {"error": str(error)})
                    self._send(job, DONE)

    def _generate(self, jobs: list[GenerationJob]) -> None:
        encoded = self.tokenizer(
            [job.prompt for job in jobs], return_tensors="pt", padding=True
        ).to("cuda")
        attention_mask = encoded["attention_mask"]
        current_ids = encoded["input_ids"]
        past = None
        finished = [False] * len(jobs)
        token_ids: list[list[int]] = [[] for _ in jobs]
        emitted = [""] * len(jobs)
        max_steps = max(job.max_tokens for job in jobs)
        eos_id = int(self.tokenizer.eos_token_id)
        with torch.inference_mode():
            for step in range(max_steps):
                outputs = self.model(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
                past = outputs.past_key_values
                next_ids = outputs.logits[:, -1].argmax(dim=-1)
                for index, job in enumerate(jobs):
                    if finished[index]:
                        continue
                    token_id = int(next_ids[index].item())
                    if token_id == eos_id or step + 1 >= job.max_tokens:
                        finished[index] = True
                    if token_id != eos_id:
                        token_ids[index].append(token_id)
                        text = cast(
                            str,
                            self.tokenizer.decode(token_ids[index], skip_special_tokens=True),
                        )
                        delta = text[len(emitted[index]) :]
                        emitted[index] = text
                        if delta:
                            self._send(
                                job,
                                {
                                    "id": job.request_id,
                                    "object": "text_completion",
                                    "choices": [{"index": 0, "text": delta, "finish_reason": None}],
                                },
                            )
                if all(finished):
                    break
                current_ids = next_ids.unsqueeze(1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((len(jobs), 1), dtype=attention_mask.dtype, device="cuda"),
                    ],
                    dim=1,
                )
        with self._metrics_lock:
            self.generated_tokens += sum(len(tokens) for tokens in token_ids)
        for job in jobs:
            self._send(
                job,
                {
                    "id": job.request_id,
                    "object": "text_completion",
                    "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                },
            )
            self._send(job, DONE)


def create_hf_app(engine: HfStaticBatchEngine) -> FastAPI:
    app = FastAPI(title="ScaleForge HF Serving Baseline", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True, "model_ready": engine.ready}

    @app.get("/metrics")
    def metrics() -> dict[str, int | bool]:
        return engine.metrics()

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest) -> StreamingResponse:
        if request.temperature != 0.0 or not request.stream:
            raise HTTPException(status_code=400, detail="only deterministic streaming is supported")
        job = engine.submit(request.prompt, request.max_tokens)

        async def events() -> Any:
            while True:
                item = await job.output.get()
                if item is DONE:
                    yield "data: [DONE]\n\n"
                    break
                if isinstance(item, dict) and "error" in item:
                    yield f"data: {json.dumps(item)}\n\n"
                    continue
                yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
