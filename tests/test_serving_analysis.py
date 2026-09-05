from __future__ import annotations

import pandas as pd
import pytest

from scaleforge.analysis.serving import (
    hierarchical_quantile_interval,
    mark_pareto_frontier,
    slo_compliant_throughput,
)


def test_hierarchical_interval_is_replicate_aware_and_reproducible() -> None:
    frame = pd.DataFrame({"replicate": [1, 1, 2, 2], "ttft_ms": [10.0, 12.0, 20.0, 22.0]})
    first = hierarchical_quantile_interval(
        frame, "ttft_ms", quantile=0.5, bootstrap_samples=100, seed=7
    )
    second = hierarchical_quantile_interval(
        frame, "ttft_ms", quantile=0.5, bootstrap_samples=100, seed=7
    )
    assert first == second
    assert first.low <= first.estimate <= first.high


def test_pareto_frontier_marks_dominated_points() -> None:
    frame = pd.DataFrame(
        {
            "successful_requests_per_s": [1.0, 2.0, 2.5],
            "ttft_p95_ms": [100.0, 90.0, 150.0],
        }
    )
    assert mark_pareto_frontier(frame).tolist() == [False, True, True]


def test_slo_compliant_throughput_includes_error_gate() -> None:
    frame = pd.DataFrame(
        {
            "config_id": ["safe", "fast_but_failing"],
            "successful_requests_per_s": [2.0, 4.0],
            "ttft_p95_ms": [100.0, 100.0],
            "e2e_p95_ms": [500.0, 500.0],
            "error_rate": [0.0, 0.2],
        }
    )
    throughput, config_id = slo_compliant_throughput(
        frame, ttft_p95_max_ms=200, e2e_p95_max_ms=1000, error_rate_max=0.01
    )
    assert throughput == pytest.approx(2.0)
    assert config_id == "safe"
