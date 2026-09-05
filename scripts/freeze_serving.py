from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from scaleforge.protocol import canonical_sha256, sha256_file

CONFIG = Path("configs/serving/qualification.yaml")
SLO = Path("configs/serving/slo.yaml")
OUTPUT = Path("artifacts/manifests/serving_freeze.json")
FILES = (
    CONFIG,
    SLO,
    Path("scripts/load_test.py"),
    Path("scripts/run_serving_grid.py"),
    Path("scripts/analyze_serving_development.py"),
    Path("api/hf_server.py"),
    Path("src/scaleforge/evaluation/answers.py"),
    Path("src/scaleforge/analysis/serving.py"),
    Path("requirements-serve.lock"),
    Path("artifacts/manifests/serve_environment.json"),
    Path("artifacts/manifests/serving_corpus.json"),
)


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    slo = yaml.safe_load(SLO.read_text(encoding="utf-8"))
    order = config["qualification"]["balanced_order"]
    if config["state"] != "FROZEN" or config["protocol_identity"] != "SF-SERVE-v2":
        raise ValueError("invalid serving qualification configuration")
    if slo["state"] != "FROZEN" or slo["protocol_identity"] != config["protocol_identity"]:
        raise ValueError("qualification and frozen SLO identities differ")
    if len(order) != 6 or sorted((row["config_id"], row["replicate"]) for row in order) != [
        (config_id, replicate)
        for config_id in ("s0_hf_bf16", "s2_vllm_v1_runner")
        for replicate in (1, 2, 3)
    ]:
        raise ValueError("serving qualification requires six balanced runtime replicates")
    if any(sorted(row["concurrency"]) != [1, 2, 4, 8, 16, 32] for row in order):
        raise ValueError("each qualification replicate must contain the complete concurrency grid")
    corpus_path = Path(config["corpus"]["path"])
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_identity": config["protocol_identity"],
        "state": "FROZEN",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "configuration_path": str(CONFIG),
        "configuration_sha256": canonical_sha256(config),
        "slo_path": str(SLO),
        "slo_sha256": sha256_file(SLO),
        "file_sha256": {str(path): sha256_file(path) for path in FILES},
        "data_sha256": sha256_file(corpus_path),
        "development_selection_path": (
            "artifacts/analysis/serving/serving_development_selection.json"
        ),
        "pilot_runtime_estimate": {
            "hf_grid_wall_minutes": 54,
            "vllm_grid_wall_minutes_excluding_cold_start": 31,
            "vllm_project_local_cold_start_minutes": 16,
            "six_fresh_server_qualification_runs_total_hours": 5.2,
        },
        "unresolved_requirement": (
            "Replicated, restart-isolated serving qualification with tail uncertainty, "
            "failure accounting, Pareto analysis, and release decision."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
