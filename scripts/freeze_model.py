from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from scaleforge.protocol import canonical_sha256, sha256_file

DEFAULT_OUTPUT = Path("FREEZE_MANIFEST.json")
DEFAULT_CONFIG = Path("configs/model/qualification.yaml")
STATIC_HASHED_FILES = [
    Path("configs/model/baselines.yaml"),
    Path("configs/model/lora_candidates.yaml"),
    Path("src/scaleforge/evaluation/answers.py"),
    Path("src/scaleforge/modeling/prompts.py"),
    Path("src/scaleforge/analysis/quality.py"),
    Path("src/scaleforge/protocol.py"),
    Path("scripts/qualify_model.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive-existing", action="store_true")
    return parser.parse_args()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    args = parse_args()
    if git("status", "--porcelain"):
        raise RuntimeError("refuse to freeze a dirty or untracked worktree")
    if args.output.exists():
        if not args.archive_existing:
            raise FileExistsError("freeze manifest already exists")
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        identity = str(existing["protocol_identity"]).replace("-", "_")
        archive = args.output.with_name(f"FREEZE_MANIFEST_{identity}.json")
        if archive.exists():
            raise FileExistsError(f"freeze archive already exists: {archive}")
        args.output.replace(archive)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    adapter = Path(config["challenger"]["adapter_path"])
    artifact_files = [
        adapter / "adapter_config.json",
        adapter / "adapter_model.safetensors",
        adapter / "tokenizer.json",
        adapter / "tokenizer_config.json",
        adapter / "chat_template.jinja",
    ]
    all_files = [*STATIC_HASHED_FILES, args.config, *artifact_files]
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
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
