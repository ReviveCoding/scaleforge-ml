from __future__ import annotations

import math

import numpy as np


def quantiles(values: list[int]) -> dict[str, int]:
    if not values:
        raise ValueError("length vector cannot be empty")
    array = np.asarray(values)
    return {
        "p50": int(np.quantile(array, 0.50, method="higher")),
        "p90": int(np.quantile(array, 0.90, method="higher")),
        "p95": int(np.quantile(array, 0.95, method="higher")),
        "p99": int(np.quantile(array, 0.99, method="higher")),
        "max": int(array.max()),
    }


def choose_sequence_limit(max_observed: int, ratio: float, multiple: int, ceiling: int) -> int:
    proposed = math.ceil(max_observed * ratio / multiple) * multiple
    if proposed > ceiling:
        raise ValueError(
            f"observed tokens require {proposed}, exceeding configured ceiling {ceiling}; "
            "manual policy review required, never truncate silently"
        )
    return proposed
