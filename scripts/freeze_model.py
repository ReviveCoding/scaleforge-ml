from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from scaleforge.protocol import canonical_sha256, sha256_file

OUTPUT = Path("FREEZE_MANIFEST.json")
QUALIFICATION_CONFIG = Path("configs/model/qualification.yaml")
HASHED_FILES = [
    Path("configs/model/baselines.yaml"),
    Path("configs/model/lora_candidates.yaml"),
    QUALIFICATION_CONFIG,
    Path("src/scaleforge/evaluation/answers.py"),
    Path("src/scaleforge/modeling/prompts.py"),
    Path("src/scaleforge/analysis/quality.py"),
    Path("src/scaleforge/protocol.py"),
    Path("scripts/qualify_model.py"),
]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("freeze manifest already exists")
    if git("status", "--porcelain"):
        raise RuntimeError("refuse to freeze a dirty or untracked worktree")
    config = yaml.safe_load(QUALIFICATION_CONFIG.read_text(encoding="utf-8"))
    adapter = Path(config["challenger"]["adapter_path"])
    artifact_files = [
        adapter / "adapter_config.json",
        adapter / "adapter_model.safetensors",
        adapter / "tokenizer.json",
        adapter / "tokenizer_config.json",
        adapter / "chat_template.jinja",
    ]
    all_files = [*HASHED_FILES, *artifact_files]
    missing = [str(path) for path in all_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"freeze inputs missing: {missing}")
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_identity": config["protocol_identity"],
        "state": "FROZEN",
        "git_sha": git("rev-parse", "HEAD"),
        "git_tree": git("rev-parse", "HEAD^{tree}"),
        "configuration_sha256": canonical_sha256(config),
        "file_sha256": {str(path).replace("\\", "/"): sha256_file(path) for path in all_files},
        "model": config["model"],
        "tokenizer": config["tokenizer"],
        "baseline": config["baseline"],
        "challenger": config["challenger"],
        "datasets": config["datasets"],
        "generation": config["generation"],
        "parser": config["parser"],
        "release_criteria": config["release_criteria"],
        "protected_outcomes_observed": False,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
