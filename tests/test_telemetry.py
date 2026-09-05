from __future__ import annotations

from pathlib import Path

import pytest

from scaleforge.telemetry import read_nvidia_smi_csv


def test_telemetry_normalizes_units_and_flags_invalid_power(tmp_path: Path) -> None:
    path = tmp_path / "gpu.csv"
    path.write_text(
        "2026/09/05 12:00:00.000, 0, NVIDIA GPU, 70, 1500, 593.51, 90, 4000\n",
        encoding="utf-8",
    )
    frame = read_nvidia_smi_csv(path, run_id="r1")
    assert frame.iloc[0]["temperature_c"] == pytest.approx(70)
    assert not bool(frame.iloc[0]["power_valid"])
    assert frame.iloc[0]["run_id"] == "r1"


def test_telemetry_rejects_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "gpu.csv"
    path.write_text(
        "2026/09/05 12:00:00.000, 0, NVIDIA GPU, N/A, 1500, 50, 90, 4000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed"):
        read_nvidia_smi_csv(path, run_id="r1")
