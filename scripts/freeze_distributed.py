from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml

from scaleforge.protocol import canonical_sha256, sha256_file

CONFIG = Path("configs/distributed/qualification.yaml")
OUTPUT = Path("artifacts/manifests/distributed_freeze.json")
SOURCES = (
    CONFIG,
    Path("scripts/benchmark_ddp_training.py"),
    Path("src/scaleforge/distributed/workload.py"),
    Path("src/scaleforge/training/cache.py"),
    Path("src/scaleforge/training/examples.py"),
    Path("configs/model/baselines.yaml"),
)


def main() -> None:
    if torch.cuda.device_count() < 2:
        raise RuntimeError("distributed freeze requires at least two visible physical CUDA GPUs")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["state"] != "FROZEN":
        raise ValueError("set distributed config state to FROZEN in a reviewed commit first")
    if OUTPUT.exists():
        raise FileExistsError(f"distributed freeze already exists: {OUTPUT}")
    payload = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_identity": config["protocol_identity"],
        "state": "FROZEN",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "configuration_path": str(CONFIG),
        "configuration_sha256": canonical_sha256(config),
        "file_sha256": {str(path): sha256_file(path) for path in SOURCES},
        "data_sha256": sha256_file(Path(config["data"]["path"])),
        "physical_cuda_gpu_count": torch.cuda.device_count(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
