from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

WAREHOUSE = Path("artifacts/warehouse")
ANALYSIS = Path("artifacts/analysis/training")
FIGURES = Path("reports/figures")
TABLES = Path("artifacts/analysis/figure_tables")


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    runs = pd.read_parquet(WAREHOUSE / "training_runs.parquet")
    telemetry = pd.read_parquet(WAREHOUSE / "gpu_telemetry.parquet")
    labels = {"t0_eager_fixed": "T0 eager fixed", "t4_dynamic_compile": "T4 dynamic + compile"}
    colors = {"t0_eager_fixed": "#2457C5", "t4_dynamic_compile": "#E68613"}

    throughput = runs[
        ["run_id", "config_id", "replicate", "run_order", "tokens_per_s", "step_p95_s"]
    ].copy()
    throughput.to_parquet(TABLES / "F04_training_throughput.parquet", index=False)
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    for index, config_id in enumerate(labels):
        values = throughput.loc[throughput["config_id"].eq(config_id), "tokens_per_s"]
        axis.scatter([index] * len(values), values, color=colors[config_id], s=55, zorder=3)
        axis.hlines(
            values.median(), index - 0.22, index + 0.22, color="black", linewidth=2, zorder=4
        )
    axis.set_xticks(range(len(labels)), labels.values())
    axis.set_ylabel("Non-padding tokens/s")
    axis.set_ylim(bottom=0)
    axis.set_title("Frozen training qualification (median line, n=3)")
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, "F04_training_throughput.png")

    memory = runs[
        ["run_id", "config_id", "replicate", "peak_allocated_mib", "peak_reserved_mib"]
    ].copy()
    memory.to_parquet(TABLES / "F05_training_vram.parquet", index=False)
    medians = memory.groupby("config_id")[["peak_allocated_mib", "peak_reserved_mib"]].median()
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    medians.loc[list(labels)].rename(index=labels).plot.bar(
        ax=axis, color=["#4C956C", "#7A3E9D"], rot=0
    )
    axis.set_ylabel("VRAM (MiB)")
    axis.set_ylim(bottom=0)
    axis.set_title("Peak training memory across qualified configurations")
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, "F05_training_vram.png")

    profiles = [
        (
            "T0 fixed padding",
            ANALYSIS / "pilot-sf-train-v1-t0-profile-retry1-operators.json",
            3,
        ),
        (
            "T1 dynamic padding",
            ANALYSIS / "pilot-sf-train-v1-t1-profile-operators.json",
            1,
        ),
    ]
    selected_ops = {"cudaLaunchKernel", "aten::mm", "aten::linear", "aten::_to_copy"}
    operator_rows = []
    for config_label, path, profiled_steps in profiles:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for operation in payload["top_operations"]:
            if operation["name"] in selected_ops:
                operator_rows.append(
                    {
                        "configuration": config_label,
                        "operation": operation["name"],
                        "total_duration_ms_per_profiled_step": operation["total_duration_us"]
                        / 1000
                        / profiled_steps,
                        "calls_per_profiled_step": operation["count"] / profiled_steps,
                        "profiled_steps": profiled_steps,
                        "source_path": str(path),
                    }
                )
    operators = pd.DataFrame(operator_rows)
    operators.to_parquet(TABLES / "F06_profiler_before_after.parquet", index=False)
    pivot = operators.pivot(
        index="operation", columns="configuration", values="total_duration_ms_per_profiled_step"
    )
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    pivot.plot.bar(ax=axis, color=["#E68613", "#2457C5"], rot=20)
    axis.set_ylabel("Aggregate operator duration / profiled step (ms)")
    axis.set_ylim(bottom=0)
    axis.set_title("Profiler-guided padding intervention (development evidence)")
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, "F06_profiler_before_after.png")

    telemetry = telemetry.copy()
    telemetry["seconds_from_run_start"] = telemetry.groupby("run_id")["timestamp"].transform(
        lambda values: (values - values.min()).dt.total_seconds()
    )
    telemetry.to_parquet(TABLES / "F13_gpu_thermal_power_trace.parquet", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.8), sharex=False)
    for run_id, group in telemetry.groupby("run_id", sort=True):
        config_id = str(runs.loc[runs["run_id"].eq(run_id), "config_id"].iloc[0])
        replicate = int(runs.loc[runs["run_id"].eq(run_id), "replicate"].iloc[0])
        short = f"{config_id.split('_')[0].upper()} r{replicate}"
        axes[0].plot(group["seconds_from_run_start"], group["temperature_c"], label=short)
        axes[1].plot(group["seconds_from_run_start"], group["power_w"], label=short)
    axes[0].set_ylabel("GPU temperature (°C)")
    axes[1].set_ylabel("GPU power (W)")
    axes[1].set_xlabel("Seconds from telemetry start")
    axes[0].set_title("Qualification GPU thermal and power traces")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(ncol=3, fontsize=8)
    save_figure(fig, "F13_gpu_thermal_power_trace.png")


if __name__ == "__main__":
    main()
