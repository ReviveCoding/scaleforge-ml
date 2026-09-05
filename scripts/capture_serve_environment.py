from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import vllm

OUTPUT = Path("artifacts/manifests/serve_environment.json")
INVENTORY = Path("artifacts/manifests/serve_environment.txt")
LOCK = Path("requirements-serve.lock")
INPUT = Path("requirements-serve.in")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    device_count = torch.cuda.device_count()
    if not torch.cuda.is_available() or device_count != 1:
        raise RuntimeError("serving qualification expects the detected single CUDA GPU")
    packages = [line for line in INVENTORY.read_text(encoding="utf-8").splitlines() if line]
    gpu = command(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    )
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "environment": ".venv-serve",
        "purpose": "isolated WSL vLLM and serving benchmark runtime",
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "vllm": vllm.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": device_count,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "nvidia_smi_name_driver_memory_mib": gpu,
        "requirements_input_sha256": sha256_file(INPUT),
        "lock_sha256": sha256_file(LOCK),
        "inventory_sha256": sha256_file(INVENTORY),
        "inventory_packages": len(packages),
        "git_sha": command("git", "rev-parse", "HEAD"),
        "separation_note": "vLLM owns this environment's compatible PyTorch stack.",
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
