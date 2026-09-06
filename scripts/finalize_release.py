from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

WAREHOUSE = Path("artifacts/warehouse")
ANALYSIS = Path("artifacts/analysis")
TABLES = ANALYSIS / "figure_tables"
FIGURES = Path("reports/figures")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    gates = pd.read_parquet(WAREHOUSE / "release_gate_results.parquet")
    expected = {"MODEL", "TRAINING_SYSTEM", "SERVING", "DISTRIBUTED", "REPRODUCIBILITY"}
    if set(gates["subsystem"]) != expected or gates["subsystem"].duplicated().any():
        raise ValueError("release-gate matrix is incomplete or duplicated")

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_table = gates[["subsystem", "decision", "selected_configuration", "evidence_path"]]
    figure_table.to_parquet(TABLES / "F14_release_gate_scorecard.parquet", index=False)

    colors = {
        "PASS": "#2E8B57",
        "REVIEW": "#E68613",
        "BLOCK": "#B22222",
        "BLOCKED_EXTERNAL": "#666666",
        "NOT_APPLICABLE": "#A9A9A9",
    }
    ordered = gates.set_index("subsystem").loc[
        ["MODEL", "TRAINING_SYSTEM", "SERVING", "DISTRIBUTED", "REPRODUCIBILITY"]
    ]
    figure, axis = plt.subplots(figsize=(9.5, 4.3))
    axis.barh(
        range(len(ordered)),
        [1] * len(ordered),
        color=[colors[value] for value in ordered["decision"]],
    )
    axis.set_yticks(range(len(ordered)), [value.replace("_", " ") for value in ordered.index])
    axis.set_xticks([])
    axis.set_xlim(0, 1)
    axis.invert_yaxis()
    axis.set_title("Independent ScaleForge release gates")
    for index, decision in enumerate(ordered["decision"]):
        axis.text(
            0.5,
            index,
            decision.replace("_", " "),
            ha="center",
            va="center",
            color="white",
            weight="bold",
        )
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    figure.savefig(FIGURES / "F14_release_gate_scorecard.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    ledger = read_json(Path("CLAIM_LEDGER.json"))
    supported = [
        claim
        for claim in ledger["claims"]
        if claim["status"] in {"SUPPORTED", "SUPPORTED_WITH_QUALIFIER"}
    ]
    resume_claims = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "source": "CLAIM_LEDGER.json",
        "policy": "Only claims in this artifact may be used, with qualifications intact.",
        "claims": supported,
    }
    (ANALYSIS / "resume_claims.json").write_text(
        json.dumps(resume_claims, indent=2) + "\n", encoding="utf-8"
    )

    release = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "project_status": "COMPLETE_WITH_EXTERNAL_DISTRIBUTED_PENDING",
        "gates": ordered.reset_index().to_dict(orient="records"),
        "selected_configuration": {
            "model": "m0",
            "training_system": "t4_dynamic_compile",
            "serving": "s2_vllm_v1_runner at concurrency 2",
            "distributed": None,
        },
    }
    (ANALYSIS / "release_decision.json").write_text(
        json.dumps(release, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
