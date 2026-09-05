import pandas as pd
import pytest

from scaleforge.warehouse import (
    validate_predictions,
    validate_run_manifest,
    validate_serving_requests,
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
