from __future__ import annotations

import pytest

from scaleforge.distributed.metrics import strong_scaling


def test_strong_scaling() -> None:
    result = strong_scaling(100.0, 180.0, 2)
    assert result.speedup == pytest.approx(1.8)
    assert result.scaling_efficiency == pytest.approx(0.9)


def test_strong_scaling_rejects_synthetic_single_gpu() -> None:
    with pytest.raises(ValueError, match="at least two"):
        strong_scaling(100.0, 100.0, 1)
