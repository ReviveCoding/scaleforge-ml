import pandas as pd
import pytest

from scaleforge.warehouse import (
    normalize_failure_type,
    validate_predictions,
    validate_run_manifest,
    validate_serving_requests,
    validate_training_results,
)


def test_manifest_rejects_duplicate_runs() -> None:
    frame = pd.DataFrame(
        {
            "run_id": ["r1", "r1"],
            "protocol_identity": ["p", "p"],
            "config_id": ["c", "c"],
            "state": ["PILOT", "PILOT"],
            "started_at": ["x", "x"],
            "completed_at": ["y", "y"],
            "status": ["COMPLETED", "COMPLETED"],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        validate_run_manifest(frame)


def test_prediction_validation_fails_closed_on_incomplete_matrix() -> None:
    frame = pd.DataFrame(
        {
            "run_id": ["r"],
            "protocol_identity": ["p"],
            "state": ["QUALIFICATION"],
            "dataset_id": ["d"],
            "config_id": ["c"],
            "example_id": ["e"],
            "correct": [True],
        }
    )
    with pytest.raises(ValueError, match="cardinality"):
        validate_predictions(frame, expected_counts={("d", "c"): 2}, protocol_identity="p")


def test_serving_validation_requires_failure_taxonomy() -> None:
    frame = pd.DataFrame(
        {
            "run_id": ["r"],
            "request_id": ["q"],
            "config_id": ["s"],
            "replicate": [1],
            "concurrency": [1],
            "measured": [True],
            "success": [False],
            "failure_type": [None],
            "e2e_ms": [1.0],
        }
    )
    with pytest.raises(ValueError, match="taxonomy"):
        validate_serving_requests(frame)


def test_failure_taxonomy_normalization() -> None:
    assert normalize_failure_type("CUDA OutOfMemoryError") == "OOM"
    assert normalize_failure_type("Decimal.InvalidOperation") == "parser_failure"
    assert normalize_failure_type("EngineCore UVA unsupported") == "service_failure"
    assert normalize_failure_type("unexpected exception") == "worker_crash"


def test_training_validation_matches_measured_counts_and_tokens() -> None:
    steps = pd.DataFrame(
        {
            "run_id": ["r", "r"],
            "protocol_identity": ["SF-TRAIN-v1", "SF-TRAIN-v1"],
            "state": ["QUALIFICATION", "QUALIFICATION"],
            "config_id": ["t", "t"],
            "replicate": [1, 1],
            "step": [1, 2],
            "measured": [False, True],
            "tokens": [5, 7],
            "step_time_s": [1.0, 1.0],
        }
    )
    runs = pd.DataFrame(
        {
            "run_id": ["r"],
            "protocol_identity": ["SF-TRAIN-v1"],
            "state": ["QUALIFICATION"],
            "status": ["COMPLETED"],
            "config_id": ["t"],
            "replicate": [1],
            "measured_steps": [1],
            "tokens": [7],
        }
    )
    validate_training_results(steps, runs)
    runs.loc[0, "tokens"] = 8
    with pytest.raises(ValueError, match="token totals"):
        validate_training_results(steps, runs)
