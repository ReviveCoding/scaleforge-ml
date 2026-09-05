from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/raw/distributed"))
    parser.add_argument("--backend", choices=("gloo", "nccl"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = args.backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    device = torch.device(f"cuda:{local_rank}" if backend == "nccl" else "cpu")
    value = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2
    if float(value.cpu()) != expected:
        raise RuntimeError("distributed all-reduce produced an invalid result")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.run_id}-rank{rank}.json"
    if output.exists():
        raise FileExistsError(f"distributed run artifact exists: {output}")
    output.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "created_at": datetime.now(UTC).isoformat(),
                "run_id": args.run_id,
                "state": "PILOT",
                "backend": backend,
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "all_reduce_sum": float(value.cpu()),
                "status": "COMPLETED",
                "numerical_scaling_claim_allowed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
