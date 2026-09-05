from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from scripts.load_test import load_and_validate_corpus, run_and_persist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--protocol-identity", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--state", choices=("DEVELOPMENT", "QUALIFICATION"), required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", required=True)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--corpus", type=Path, default=Path("artifacts/data/serving_request_corpus.parquet")
    )
    return parser.parse_args()


def main() -> None:
    grid = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        cache_dir=Path("artifacts/cache/huggingface"),
    )
    for concurrency in grid.concurrency:
        args = argparse.Namespace(
            base_url=grid.base_url,
            model=grid.model,
            config_id=grid.config_id,
            protocol_identity=grid.protocol_identity,
            run_id=f"{grid.run_prefix}-c{concurrency}-r{grid.replicate}",
            state=grid.state,
            replicate=grid.replicate,
            concurrency=concurrency,
            requests=grid.requests,
            warmup=grid.warmup,
            max_new_tokens=grid.max_new_tokens,
            max_model_len=grid.max_model_len,
            timeout_s=grid.timeout_s,
            corpus=grid.corpus,
        )
        corpus = load_and_validate_corpus(args)
        run_and_persist(args, corpus=corpus, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
