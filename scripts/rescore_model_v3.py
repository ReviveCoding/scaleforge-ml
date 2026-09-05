from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scaleforge.evaluation.answers import (
    normalize_numeric_answer,
    parse_generated_answer,
    parse_reference_gsm8k,
)
from scaleforge.protocol import canonical_sha256, sha256_file

CONFIG_PATH = Path("configs/model/qualification_v3.yaml")
FREEZE_PATH = Path("FREEZE_MANIFEST.json")
LEDGER_PATH = Path("FINAL_ACCESS_LEDGER.json")


def update_ledger(entry: dict[str, Any]) -> None:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    existing = [
        index
        for index, value in enumerate(ledger["entries"])
        if value["access_id"] == entry["access_id"]
    ]
    if len(existing) > 1:
        raise ValueError("duplicate protected access IDs")
    if existing:
        ledger["entries"][existing[0]] = entry
    else:
        ledger["entries"].append(entry)
    ledger["updated_at"] = datetime.now(UTC).isoformat()
    temporary = LEDGER_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    temporary.replace(LEDGER_PATH)


def verify_freeze(config: dict[str, Any], freeze: dict[str, Any]) -> None:
    if freeze["protocol_identity"] != "SF-MODEL-v3" or freeze["state"] != "FROZEN":
        raise ValueError("invalid v3 freeze")
    if canonical_sha256(config) != freeze["configuration_sha256"]:
        raise ValueError("v3 configuration differs from freeze")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if git_sha != freeze["git_sha"]:
        raise ValueError("Git HEAD differs from v3 freeze")
    for group in ("file_sha256", "source_prediction_sha256"):
        for path_text, expected in freeze[group].items():
            if sha256_file(Path(path_text)) != expected:
                raise ValueError(f"frozen artifact mismatch: {path_text}")
    for dataset_id, expected in freeze["protected_dataset_sha256"].items():
        path = Path(config["datasets"][dataset_id]["protected_path"])
        if sha256_file(path) != expected:
            raise ValueError(f"protected reference mismatch: {dataset_id}")


def reference_maps(config: dict[str, Any]) -> dict[str, str]:
    math = pd.read_json(config["datasets"]["math500"]["protected_path"], lines=True)
    if len(math) != int(config["datasets"]["math500"]["expected_rows"]):
        raise ValueError("MATH-500 protected reference cardinality mismatch")
    return {
        str(row.unique_id): normalize_numeric_answer(str(row.answer)) for row in math.itertuples()
    }


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    verify_freeze(config, freeze)
    math_references = reference_maps(config)
    for dataset_id, dataset in config["datasets"].items():
        access_id = f"access-rescore-sf-model-v3-{dataset_id}"
        entry = {
            "access_id": access_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "protocol_identity": "SF-MODEL-v3",
            "dataset": dataset["repository"],
            "revision": dataset["revision"],
            "requested_split": "test",
            "purpose": "Protected evaluator-only rescore of frozen SF-MODEL-v2 predictions",
            "run_id": f"rescore-sf-model-v3-{dataset_id}",
            "outcome_exposed": False,
            "scientific_outcomes_exposed": [],
            "artifacts": [],
            "disposition": "QUALIFICATION_STARTED",
        }
        update_ledger(entry)
        try:
            expected = int(dataset["expected_rows"])
            for config_id, source_text in config["source_predictions"][dataset_id].items():
                source = Path(source_text)
                rows = [
                    json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
                ]
                if len(rows) != expected or len({row["example_id"] for row in rows}) != expected:
                    raise ValueError(f"incomplete or duplicate frozen source: {source}")
                suffix = "m0" if config_id == "m0" else "mstar"
                run_id = f"qual-sf-model-v3-{dataset_id}-{suffix}"
                output = Path("artifacts/raw/model") / f"{run_id}.jsonl"
                summary_path = Path("artifacts/analysis/model") / f"{run_id}.json"
                if output.exists() or summary_path.exists():
                    raise FileExistsError(f"v3 rescore run exists: {run_id}")
                rescored = []
                for row in rows:
                    parsed = parse_generated_answer(str(row["prediction"]))
                    if dataset_id == "gsm8k":
                        reference = parse_reference_gsm8k(str(row["reference_solution"]))
                        if reference.failure or reference.normalized is None:
                            raise ValueError("invalid frozen GSM8K reference")
                        normalized_reference = reference.normalized
                    else:
                        normalized_reference = math_references[str(row["example_id"])]
                    rescored.append(
                        {
                            **row,
                            "run_id": run_id,
                            "protocol_identity": "SF-MODEL-v3",
                            "state": "QUALIFICATION",
                            "source_run_id": row["run_id"],
                            "source_prediction_sha256": freeze["source_prediction_sha256"][
                                str(source)
                            ],
                            "reference_answer": normalized_reference,
                            "parsed_answer": parsed.normalized,
                            "parse_failure": parsed.failure,
                            "correct": parsed.normalized == normalized_reference,
                        }
                    )
                with output.open("w", encoding="utf-8") as stream:
                    for row in rescored:
                        stream.write(json.dumps(row) + "\n")
                summary = {
                    "schema_version": "1.0.0",
                    "created_at": datetime.now(UTC).isoformat(),
                    "run_id": run_id,
                    "protocol_identity": "SF-MODEL-v3",
                    "state": "QUALIFICATION",
                    "method": "frozen_prediction_rescore",
                    "dataset_id": dataset_id,
                    "config_id": config_id,
                    "examples": len(rescored),
                    "exact_match": sum(bool(row["correct"]) for row in rescored) / len(rescored),
                    "parse_failures": sum(row["parse_failure"] is not None for row in rescored),
                    "source_run_id": rows[0]["run_id"],
                    "source_prediction_sha256": freeze["source_prediction_sha256"][str(source)],
                    "freeze_manifest_sha256": sha256_file(FREEZE_PATH),
                }
                summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                entry["artifacts"].extend([str(output), str(summary_path)])
            entry["outcome_exposed"] = True
            entry["scientific_outcomes_exposed"] = ["rescored_predictions", "aggregate_exact_match"]
            entry["disposition"] = "QUALIFICATION_COMPLETED"
        finally:
            update_ledger(entry)


if __name__ == "__main__":
    main()
