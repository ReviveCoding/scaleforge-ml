from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ANALYSIS = Path("artifacts/analysis/model/model_qualification_sf_model_v3.json")
SLICES = Path("artifacts/analysis/model/model_quality_slices_sf_model_v3.parquet")
FIGURES = Path("reports/figures")
TABLES = Path("artifacts/analysis/figure_tables")


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    result = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    TABLES.mkdir(parents=True, exist_ok=True)
    quality_rows = []
    for dataset, key in (("GSM8K FINAL", "gsm8k_final"), ("MATH-500 OOD", "math500_ood")):
        stats = result[key]["paired_statistics_m0_vs_mstar"]
        quality_rows.extend(
            [
                {"dataset": dataset, "configuration": "M0", "exact_match": stats["baseline_em"]},
                {
                    "dataset": dataset,
                    "configuration": "LoRA M*",
                    "exact_match": stats["candidate_em"],
                },
            ]
        )
    quality = pd.DataFrame(quality_rows)
    quality.to_parquet(TABLES / "F01_model_quality.parquet", index=False)
    pivot = quality.pivot(index="dataset", columns="configuration", values="exact_match") * 100
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    pivot[["M0", "LoRA M*"]].plot.bar(ax=axis, color=["#2457C5", "#E68613"], rot=0)
    axis.set_ylabel("Exact match (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Protected model quality (complete datasets)")
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, "F01_model_quality.png")

    transitions = result["gsm8k_final"]["paired_statistics_m0_vs_mstar"]
    transition_frame = pd.DataFrame(
        {
            "outcome": ["Both correct", "M* only", "M0 only", "Both wrong"],
            "examples": [
                transitions["both_correct"],
                transitions["candidate_only"],
                transitions["baseline_only"],
                transitions["both_wrong"],
            ],
        }
    )
    transition_frame.to_parquet(TABLES / "F02_quality_transitions.parquet", index=False)
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.barh(
        transition_frame["outcome"],
        transition_frame["examples"],
        color=["#4C956C", "#E9C46A", "#E76F51", "#6C757D"],
    )
    axis.set_xlabel("Paired GSM8K FINAL examples")
    axis.set_title("Where M0 and LoRA M* changed outcomes")
    axis.grid(axis="x", alpha=0.25)
    save_figure(fig, "F02_quality_transitions.png")

    slices = pd.read_parquet(SLICES)
    structural = slices.loc[
        slices["dataset_id"].eq("gsm8k")
        & slices["slice_dimension"].isin(
            ["prompt_tokens_quartile", "reference_solution_chars_quartile"]
        )
    ].copy()
    structural.to_parquet(TABLES / "F03_model_slice_deltas.parquet", index=False)
    structural["label"] = (
        structural["slice_dimension"].str.replace("_quartile", "")
        + "\n"
        + structural["slice_value"]
    )
    fig, axis = plt.subplots(figsize=(9.2, 5.2))
    positions = range(len(structural))
    axis.errorbar(
        list(positions),
        structural["delta_pp"],
        yerr=[
            structural["delta_pp"] - structural["ci_low_pp"],
            structural["ci_high_pp"] - structural["delta_pp"],
        ],
        fmt="o",
        color="#7A3E9D",
        capsize=3,
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(list(positions), structural["label"], rotation=35, ha="right")
    axis.set_ylabel("LoRA M* - M0 exact match (pp)")
    axis.set_title("GSM8K structural slice deltas (descriptive, not causal)")
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, "F03_model_slice_deltas.png")


if __name__ == "__main__":
    main()
