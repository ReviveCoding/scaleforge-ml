from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from scaleforge.protocol import canonical_sha256, sha256_file

CONFIG = Path("configs/training/qualification.yaml")
OUTPUT = Path("artifacts/manifests/training_freeze.json")
FILES = [
    CONFIG,
    Path("scripts/benchmark_training.py"),
    Path("src/scaleforge/benchmarks/training.py"),
    Path("src/scaleforge/training/cache.py"),
    Path("src/scaleforge/training/examples.py"),
    Path("configs/model/baselines.yaml"),
]


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["state"] != "FROZEN" or config["protocol_identity"] != "SF-TRAIN-v1":
        raise ValueError("invalid training qualification configuration")
    if len(config["qualification"]["balanced_order"]) != 6:
        raise ValueError("training qualification requires six balanced runs")
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_identity": "SF-TRAIN-v1",
        "state": "FROZEN",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "configuration_path": str(CONFIG),
        "configuration_sha256": canonical_sha256(config),
        "file_sha256": {str(path): sha256_file(path) for path in FILES},
        "data_sha256": sha256_file(Path(config["data"]["path"])),
        "pilot_runtime_estimate_minutes": 40,
        "selection_basis": {
            "t0_unprofiled_tokens_per_s": 514.4152034404076,
            "t1_unprofiled_tokens_per_s": 1389.9511134541733,
            "t4_steady_tokens_per_s": 4401.73814997755,
            "t4_cold_setup_plus_first_step_s": 405.4109575030161,
            "estimated_compile_break_even_steps_vs_t1": 227,
            "target_training_steps": 378,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
