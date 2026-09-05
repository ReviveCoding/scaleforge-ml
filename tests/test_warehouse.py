import pandas as pd
import pytest

from scaleforge.warehouse import validate_run_manifest


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
