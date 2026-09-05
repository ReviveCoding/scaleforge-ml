from __future__ import annotations

import pytest

from scaleforge.benchmarks.training import (
    summarize_training_steps,
    validate_training_system_config,
)


def test_training_config_enforces_global_batch() -> None:
    config = {
        "state": "DEVELOPMENT",
        "data": {"split_role": "FIT"},
        "workload": {
            "global_batch_size": 16,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 2,
            "measured_steps": 3,
        },
        "configs": {"t0_eager_fixed": {}},
    }
    with pytest.raises(ValueError, match="global batch"):
        validate_training_system_config(config)


def test_training_step_summary_excludes_warmup() -> None:
    rows = [
        {
            "measured": False,
            "step_time_s": 10.0,
            "tokens": 100,
            "peak_allocated_mib": 50,
            "peak_reserved_mib": 60,
        },
        {
            "measured": True,
            "step_time_s": 2.0,
            "tokens": 200,
            "peak_allocated_mib": 70,
            "peak_reserved_mib": 80,
        },
        {
            "measured": True,
            "step_time_s": 1.0,
            "tokens": 100,
            "peak_allocated_mib": 75,
            "peak_reserved_mib": 90,
        },
    ]
    summary = summarize_training_steps(rows)
    assert summary["tokens_per_s"] == pytest.approx(100.0)
    assert summary["measured_steps"] == 2
    assert summary["peak_allocated_mib"] == 75
