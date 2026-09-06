from pathlib import Path

from ui.dashboard import load_dashboard_data


def test_dashboard_loads_only_qualified_artifacts() -> None:
    data = load_dashboard_data(Path(__file__).resolve().parents[1])
    assert data["project_status"] == "COMPLETE_WITH_EXTERNAL_DISTRIBUTED_PENDING"
    assert set(data["gates"]["subsystem"]) == {
        "MODEL",
        "TRAINING_SYSTEM",
        "SERVING",
        "DISTRIBUTED",
        "REPRODUCIBILITY",
    }
    assert len(data["selected_serving"]) == 2
    assert len(data["training"]) == 6
