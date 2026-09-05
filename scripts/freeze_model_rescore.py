from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from scaleforge.protocol import canonical_sha256, sha256_file

CONFIG = Path("configs/model/qualification_v3.yaml")
OUTPUT = Path("FREEZE_MANIFEST.json")
FILES = [
    Path("configs/model/qualification_v3.yaml"),
    Path("src/scaleforge/evaluation/answers.py"),
    Path("src/scaleforge/analysis/quality.py"),
    Path("src/scaleforge/protocol.py"),
    Path("scripts/rescore_model_v3.py"),
    Path("scripts/analyze_model_qualification.py"),
]


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["state"] != "FROZEN" or config["protocol_identity"] != "SF-MODEL-v3":
        raise ValueError("invalid v3 freeze configuration")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    sources = {
        f"{dataset_id}:{config_id}": sha256_file(Path(path))
        for dataset_id, dataset in config["source_predictions"].items()
        for config_id, path in dataset.items()
    }
    source_paths = [
        Path(path) for dataset in config["source_predictions"].values() for path in dataset.values()
    ]
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_identity": "SF-MODEL-v3",
        "state": "FROZEN",
        "git_sha": git_sha,
        "configuration_path": str(CONFIG),
        "configuration_sha256": canonical_sha256(config),
        "file_sha256": {str(path): sha256_file(path) for path in FILES},
        "source_prediction_sha256": {str(path): sha256_file(path) for path in source_paths},
        "protected_dataset_sha256": {
            dataset_id: sha256_file(Path(dataset["protected_path"]))
            for dataset_id, dataset in config["datasets"].items()
        },
        "source_config_hashes_by_id": sources,
        "freeze_scope": (
            "Corrected answer parser, immutable complete v2 prediction texts, raw protected "
            "references, paired statistics, and unchanged release criteria."
        ),
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
