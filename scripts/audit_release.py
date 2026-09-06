from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scaleforge.protocol import sha256_file

ROOT = Path(".")
OUTPUT = Path("artifacts/audit/final_audit.json")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_hash_map(mapping: dict[str, str]) -> list[str]:
    mismatches = []
    for raw_path, expected in mapping.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(raw_path)
    return mismatches


def main() -> None:
    required_documents = [
        "PROJECT_SPEC.md",
        "REQUIREMENTS_MATRIX.md",
        "EXPERIMENT_PROTOCOL.md",
        "README.md",
        "DATA_CARD.md",
        "MODEL_CARD.md",
        "SYSTEM_CARD.md",
        "EVALUATION_CARD.md",
        "FINAL_TECHNICAL_REPORT.md",
        "FINAL_AUDIT.md",
        "reports/RESUME_EVIDENCE.md",
        "INTERVIEW_GUIDE.md",
    ]
    missing_documents = [path for path in required_documents if not Path(path).is_file()]

    integrity = read_json(Path("artifacts/analysis/warehouse_integrity.json"))
    gates = pd.read_parquet("artifacts/warehouse/release_gate_results.parquet")
    failures = pd.read_parquet("artifacts/warehouse/failure_events.parquet")
    ledger = read_json(Path("CLAIM_LEDGER.json"))
    resume = read_json(Path("artifacts/analysis/resume_claims.json"))

    expected_gate_decisions = {
        "MODEL": "PASS",
        "TRAINING_SYSTEM": "PASS",
        "SERVING": "REVIEW",
        "DISTRIBUTED": "BLOCKED_EXTERNAL",
        "REPRODUCIBILITY": "REVIEW",
    }
    observed_gate_decisions = gates.set_index("subsystem")["decision"].to_dict()
    supported_ids = {
        claim["claim_id"]
        for claim in ledger["claims"]
        if claim["status"] in {"SUPPORTED", "SUPPORTED_WITH_QUALIFIER"}
    }
    resume_ids = {claim["claim_id"] for claim in resume["claims"]}
    supported_distributed_claims = [
        claim["claim_id"]
        for claim in ledger["claims"]
        if claim["claim_id"].startswith("CLM-DIST") and claim["status"] != "NOT_SUPPORTED"
    ]
    missing_claim_artifacts = sorted(
        {
            path
            for claim in ledger["claims"]
            for path in claim["artifact_paths"]
            if not Path(path).exists()
        }
    )

    hash_mismatches: list[str] = []
    for path in (
        Path("FREEZE_MANIFEST.json"),
        Path("artifacts/manifests/training_freeze.json"),
        Path("artifacts/manifests/serving_freeze.json"),
    ):
        payload = read_json(path)
        hash_mismatches.extend(check_hash_map(payload["file_sha256"]))
    model_freeze = read_json(Path("FREEZE_MANIFEST.json"))
    hash_mismatches.extend(check_hash_map(model_freeze["source_prediction_sha256"]))
    training = read_json(
        Path("artifacts/analysis/training/training_qualification_sf_train_v1.json")
    )
    serving = read_json(Path("artifacts/analysis/serving/serving_qualification_sf_serve_v2.json"))
    hash_mismatches.extend(check_hash_map(training["input_artifact_sha256"]))
    hash_mismatches.extend(check_hash_map(serving["input_artifact_sha256"]))

    document_tokens = {
        "README.md": ["66.64%", "49.43%", "348.51%", "71.39%", "BLOCKED_EXTERNAL"],
        "FINAL_TECHNICAL_REPORT.md": [
            "66.64%",
            "49.43%",
            "348.51%",
            "71.39%",
            "BLOCKED_EXTERNAL",
        ],
        "reports/RESUME_EVIDENCE.md": [
            "66.64%",
            "49.43%",
            "348.51%",
            "71.39%",
        ],
        "INTERVIEW_GUIDE.md": ["17.21", "348.51%", "71.39%", "only one GPU"],
    }
    documentation_mismatches: list[str] = []
    for raw_path, tokens in document_tokens.items():
        text = Path(raw_path).read_text(encoding="utf-8")
        documentation_mismatches.extend(
            f"{raw_path}: missing {token}" for token in tokens if token not in text
        )

    checks = {
        "required_documents_present": not missing_documents,
        "warehouse_integrity_pass": integrity["status"] == "PASS",
        "warehouse_duplicate_runs_zero": integrity["duplicate_run_ids"] == 0,
        "release_gates_exact": observed_gate_decisions == expected_gate_decisions,
        "failures_preserved": len(failures) >= 1,
        "claim_artifacts_present": not missing_claim_artifacts,
        "resume_export_matches_supported_ledger": supported_ids == resume_ids,
        "no_supported_distributed_numeric_claim": not supported_distributed_claims,
        "frozen_and_qualification_hashes_match": not hash_mismatches,
        "recruiter_document_key_numbers_present": not documentation_mismatches,
    }
    status = "PASS" if all(checks.values()) else "BLOCK"
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "checks": checks,
        "details": {
            "missing_documents": missing_documents,
            "missing_claim_artifacts": missing_claim_artifacts,
            "hash_mismatches": sorted(set(hash_mismatches)),
            "documentation_mismatches": documentation_mismatches,
            "release_gate_decisions": observed_gate_decisions,
            "warehouse_table_rows": integrity["table_rows"],
            "preserved_failure_events": len(failures),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
