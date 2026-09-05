from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    return parser.parse_args()


def completion(client: httpx.Client, *, model: str, prompt: str, max_tokens: int) -> str:
    chunks: list[str] = []
    with client.stream(
        "POST",
        "/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        },
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event: dict[str, Any] = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            choices = event.get("choices") or []
            if choices and choices[0].get("text"):
                chunks.append(str(choices[0]["text"]))
    return "".join(chunks)


def endpoint_payload(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {
        "ok": True,
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type"),
        "body_characters": len(response.text),
    }


def main() -> None:
    args = parse_args()
    corpus = pd.read_parquet("artifacts/data/serving_request_corpus.parquet")
    prompt = str(corpus.iloc[0]["rendered_prompt"])
    with httpx.Client(base_url=args.base_url, timeout=180.0) as client:
        health = endpoint_payload(client.get("/health"))
        models = endpoint_payload(client.get("/v1/models"))
        before = endpoint_payload(client.get("/metrics"))
        first = completion(client, model=args.model, prompt=prompt, max_tokens=args.max_tokens)
        second = completion(client, model=args.model, prompt=prompt, max_tokens=args.max_tokens)
        after = endpoint_payload(client.get("/metrics"))
    if not health.get("ok") or health.get("model_ready") is False:
        raise RuntimeError("health endpoint did not report a ready model")
    if not models.get("data") or not first or first != second:
        raise RuntimeError("streamed deterministic completion smoke failed")
    if "requests_total" in after:
        if int(after["requests_total"]) - int(before["requests_total"]) != 2:
            raise RuntimeError("HF request counter did not advance by two")
        if int(after["failures_total"]) != int(before["failures_total"]):
            raise RuntimeError("HF failure counter advanced during smoke")
    artifact = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": args.run_id,
        "protocol_identity": "SF-SERVE-v1",
        "state": "DEVELOPMENT",
        "runtime": args.runtime,
        "health": health,
        "models": models,
        "metrics_before": before,
        "metrics_after": after,
        "responses_equal": first == second,
        "response_characters": len(first),
        "protected_data_used": False,
        "status": "PASS",
    }
    output = Path("artifacts/analysis/serving") / f"{args.run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"smoke run already exists: {output}")
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
