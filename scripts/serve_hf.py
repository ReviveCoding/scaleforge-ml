from __future__ import annotations

import argparse

import uvicorn

from api.hf_server import HfStaticBatchEngine, create_hf_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--batch-wait-ms", type=float, default=5.0)
    args = parser.parse_args()
    engine = HfStaticBatchEngine(
        max_batch_size=args.max_batch_size, batch_wait_ms=args.batch_wait_ms
    )
    uvicorn.run(create_hf_app(engine), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
