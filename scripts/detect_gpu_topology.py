from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/manifests/gpu_topology.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    devices = []
    for line in query.stdout.splitlines():
        index, uuid, name, memory_mib, driver = [field.strip() for field in line.split(",", 4)]
        devices.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(memory_mib),
                "driver_version": driver,
            }
        )
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    physical_count = len(devices)
    payload = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "physical_cuda_gpu_count": physical_count,
        "pytorch_cuda_device_count": cuda_count,
        "devices": devices,
        "distributed_numerical_qualification": (
            "ELIGIBLE" if physical_count >= 2 and cuda_count >= 2 else "BLOCKED_EXTERNAL"
        ),
        "reason": (
            None
            if physical_count >= 2 and cuda_count >= 2
            else "Fewer than two physical CUDA GPUs are visible locally."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
