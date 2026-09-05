from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from transformers import AutoTokenizer

from scaleforge.protocol import sha256_file


def start_telemetry(path: Path) -> subprocess.Popen[str] | None:
    fields = "timestamp,index,name,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used"
    try:
        return subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
                "--loop-ms=500",
                f"--filename={path}",
            ],
            text=True,
        )
    except OSError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--protocol-identity", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--state", choices=("DEVELOPMENT", "QUALIFICATION"), required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument(
        "--corpus", type=Path, default=Path("artifacts/data/serving_request_corpus.parquet")
    )
    return parser.parse_args()


async def one_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    row: Any,
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    measured: bool,
    attempt_id: str,
) -> dict[str, Any]:
    scheduled = time.perf_counter()
    chunks: list[str] = []
    chunk_times: list[float] = []
    status = 0
    failure_type = None
    failure_message = None
    async with semaphore:
        acquired = time.perf_counter()
        try:
            payload = {
                "model": args.model,
                "prompt": str(row.rendered_prompt),
                "max_tokens": args.max_new_tokens,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            async with client.stream("POST", "/v1/completions", json=payload) as response:
                status = response.status_code
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    if "error" in event:
                        failure_type = "service_failure"
                        failure_message = str(event["error"])
                        break
                    choices = event.get("choices") or []
                    if choices and choices[0].get("text"):
                        chunks.append(str(choices[0]["text"]))
                        chunk_times.append(time.perf_counter())
        except httpx.TimeoutException as error:
            failure_type = "timeout"
            failure_message = str(error)
        except httpx.HTTPStatusError as error:
            failure_type = "service_failure"
            failure_message = str(error)
        except (httpx.HTTPError, json.JSONDecodeError) as error:
            failure_type = "invalid_service_response"
            failure_message = str(error)
    completed = time.perf_counter()
    prediction = "".join(chunks)
    output_tokens = len(tokenizer.encode(prediction, add_special_tokens=False)) if prediction else 0
    success = failure_type is None and status == 200 and output_tokens > 0
    if failure_type is None and not success:
        failure_type = "invalid_generation"
        failure_message = "successful HTTP response contained no output tokens"
    ttft_ms = (chunk_times[0] - acquired) * 1000 if chunk_times else None
    itl_ms = [(right - left) * 1000 for left, right in pairwise(chunk_times)]
    tpot_ms = (
        ((completed - chunk_times[0]) * 1000 / max(output_tokens - 1, 1))
        if chunk_times and output_tokens > 1
        else None
    )
    return {
        "run_id": args.run_id,
        "protocol_identity": args.protocol_identity,
        "state": args.state,
        "request_id": attempt_id,
        "corpus_request_id": str(row.request_id),
        "example_id": str(row.example_id),
        "reference_final_answer": str(row.normalized_final_answer),
        "config_id": args.config_id,
        "replicate": args.replicate,
        "concurrency": args.concurrency,
        "measured": measured,
        "prompt_tokens": int(row.prompt_tokens),
        "output_tokens": output_tokens,
        "queue_ms": (acquired - scheduled) * 1000,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "itl_ms": itl_ms,
        "e2e_ms": (completed - acquired) * 1000,
        "http_status": status,
        "success": success,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "prediction": prediction,
        "gpu_telemetry_run_id": args.run_id,
    }


def load_and_validate_corpus(args: argparse.Namespace) -> pd.DataFrame:
    corpus = pd.read_parquet(args.corpus)
    required = {
        "request_id",
        "example_id",
        "rendered_prompt",
        "prompt_tokens",
        "normalized_final_answer",
        "split_role",
    }
    missing = required - set(corpus.columns)
    if missing or corpus.empty:
        raise ValueError(f"invalid serving corpus; missing columns: {sorted(missing)}")
    manifest = json.loads(
        Path("artifacts/manifests/serving_corpus.json").read_text(encoding="utf-8")
    )
    if (
        manifest["path"] != str(args.corpus)
        or manifest["corpus_sha256"] != sha256_file(args.corpus)
        or set(corpus["split_role"]) != {"POLICY"}
        or len(corpus) != int(manifest["rows"])
    ):
        raise ValueError("serving corpus differs from the POLICY-only manifest")
    if int(corpus["prompt_tokens"].max()) + args.max_new_tokens > args.max_model_len:
        raise ValueError("serving request would exceed the configured model context")
    return corpus


async def execute(
    args: argparse.Namespace, *, corpus: pd.DataFrame | None = None, tokenizer: Any = None
) -> tuple[list[dict[str, Any]], float]:
    if args.concurrency < 1 or args.requests < 1 or args.warmup < 0:
        raise ValueError("invalid load-test dimensions")
    corpus = load_and_validate_corpus(args) if corpus is None else corpus
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            cache_dir=Path("artifacts/cache/huggingface"),
        )
    timeout = httpx.Timeout(args.timeout_s)
    limits = httpx.Limits(
        max_connections=args.concurrency, max_keepalive_connections=args.concurrency
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout, limits=limits) as client:
        warmup_records = []
        for index in range(args.warmup):
            warmup_records.append(
                await one_request(
                    client,
                    semaphore,
                    corpus.iloc[index % len(corpus)],
                    args=args,
                    tokenizer=tokenizer,
                    measured=False,
                    attempt_id=f"{args.run_id}-warmup-{index:04d}",
                )
            )
        started = time.perf_counter()
        tasks = [
            one_request(
                client,
                semaphore,
                corpus.iloc[index % len(corpus)],
                args=args,
                tokenizer=tokenizer,
                measured=True,
                attempt_id=f"{args.run_id}-measured-{index:04d}",
            )
            for index in range(args.requests)
        ]
        measured_records = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - started
    return [*warmup_records, *measured_records], elapsed


def run_and_persist(
    args: argparse.Namespace, *, corpus: pd.DataFrame | None = None, tokenizer: Any = None
) -> dict[str, Any]:
    raw_dir = Path("artifacts/raw/serving")
    analysis_dir = Path("artifacts/analysis/serving")
    raw_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output = raw_dir / f"{args.run_id}-requests.jsonl"
    telemetry_path = raw_dir / f"{args.run_id}-gpu.csv"
    summary_path = analysis_dir / f"{args.run_id}.json"
    if output.exists() or summary_path.exists():
        raise FileExistsError(f"serving run ID exists: {args.run_id}")
    telemetry = start_telemetry(telemetry_path)
    try:
        try:
            records, elapsed = asyncio.run(execute(args, corpus=corpus, tokenizer=tokenizer))
        except Exception as error:
            failure_path = Path("artifacts/failures") / f"{args.run_id}.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "created_at": datetime.now(UTC).isoformat(),
                        "run_id": args.run_id,
                        "protocol_identity": args.protocol_identity,
                        "state": args.state,
                        "config_id": args.config_id,
                        "failure_type": "service_failure",
                        "exception_type": type(error).__name__,
                        "failure_message": str(error),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            raise
    finally:
        if telemetry is not None:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:
                telemetry.kill()
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    measured = [record for record in records if record["measured"]]
    successes = [record for record in measured if record["success"]]
    summary = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": args.run_id,
        "protocol_identity": args.protocol_identity,
        "state": args.state,
        "status": "COMPLETED",
        "config_id": args.config_id,
        "replicate": args.replicate,
        "concurrency": args.concurrency,
        "corpus_sha256": sha256_file(args.corpus),
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "warmup_requests": len(records) - len(measured),
        "requests": len(measured),
        "successes": len(successes),
        "failures": len(measured) - len(successes),
        "error_rate": (len(measured) - len(successes)) / len(measured),
        "elapsed_s": elapsed,
        "successful_requests_per_s": len(successes) / elapsed,
        "successful_output_tokens_per_s": (
            sum(int(record["output_tokens"]) for record in successes) / elapsed
        ),
        "ttft_p50_ms": statistics.median(float(record["ttft_ms"]) for record in successes)
        if successes
        else None,
        "e2e_p95_ms": float(pd.Series([record["e2e_ms"] for record in successes]).quantile(0.95))
        if successes
        else None,
        "failures_included_in_reliability_and_throughput": True,
        "telemetry_path": str(telemetry_path) if telemetry is not None else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run_and_persist(parse_args())


if __name__ == "__main__":
    main()
