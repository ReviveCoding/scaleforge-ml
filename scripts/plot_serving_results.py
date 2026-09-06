from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scaleforge.analysis.serving_qualification import saturation_knee

ANALYSIS = Path("artifacts/analysis/serving")
WAREHOUSE = Path("artifacts/warehouse")
FIGURES = Path("reports/figures")
TABLES = Path("artifacts/analysis/figure_tables")
LABELS = {
    "s0_hf_bf16": "S0 HF Transformers BF16",
    "s2_vllm_v1_runner": "S2 vLLM V1 runner BF16",
}
COLORS = {"s0_hf_bf16": "#2457C5", "s2_vllm_v1_runner": "#E68613"}


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_throughput(points: pd.DataFrame, runs: pd.DataFrame) -> None:
    table = runs[
        ["run_id", "config_id", "replicate", "concurrency", "successful_requests_per_s"]
    ].copy()
    table.to_parquet(TABLES / "F07_serving_throughput.parquet", index=False)
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    for config_id, group in points.groupby("config_id", sort=True):
        run_group = table.loc[table["config_id"].eq(config_id)]
        axis.scatter(
            run_group["concurrency"],
            run_group["successful_requests_per_s"],
            color=COLORS[config_id],
            alpha=0.35,
            s=30,
        )
        axis.plot(
            group["concurrency"],
            group["successful_requests_per_s"],
            color=COLORS[config_id],
            marker="o",
            linewidth=2,
            label=LABELS[config_id],
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Concurrency")
    axis.set_ylabel("Successful requests/s")
    axis.set_title("Serving throughput (median line; three replicate points)")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, "F07_serving_throughput_vs_concurrency.png")


def plot_ttft(points: pd.DataFrame) -> None:
    columns = [
        "config_id",
        "concurrency",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
    ]
    points[columns].to_parquet(TABLES / "F08_serving_ttft.parquet", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for axis, config_id in zip(axes, LABELS, strict=True):
        group = points.loc[points["config_id"].eq(config_id)]
        for metric, label, marker in (
            ("ttft_p50_ms", "p50", "o"),
            ("ttft_p95_ms", "p95", "s"),
            ("ttft_p99_ms", "p99", "^"),
        ):
            axis.plot(group["concurrency"], group[metric], marker=marker, label=label)
        axis.axhline(500, color="#A61B1B", linestyle="--", label="Frozen p95 SLO")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
        axis.set_xlabel("Concurrency")
        axis.set_title(LABELS[config_id])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("TTFT (ms, logarithmic scale)")
    figure.suptitle("Serving time to first token with hierarchical point estimates")
    save_figure(figure, "F08_serving_ttft_quantiles.png")


def plot_pareto(points: pd.DataFrame) -> None:
    columns = [
        "config_id",
        "concurrency",
        "successful_requests_per_s",
        "ttft_p95_ms",
        "pareto_frontier",
        "slo_compliant",
    ]
    table = points[columns].copy()
    table.to_parquet(TABLES / "F09_serving_pareto.parquet", index=False)
    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    for config_id, group in table.groupby("config_id", sort=True):
        axis.scatter(
            group["ttft_p95_ms"],
            group["successful_requests_per_s"],
            color=COLORS[config_id],
            s=65,
            label=LABELS[config_id],
        )
        for row in group.itertuples():
            marker = "★" if row.slo_compliant else str(row.concurrency)
            axis.annotate(
                marker,
                (row.ttft_p95_ms, row.successful_requests_per_s),
                xytext=(4, 4),
                textcoords="offset points",
            )
        frontier = group.loc[group["pareto_frontier"]].sort_values("ttft_p95_ms")
        axis.plot(
            frontier["ttft_p95_ms"],
            frontier["successful_requests_per_s"],
            color=COLORS[config_id],
            linestyle="--",
            alpha=0.8,
        )
    axis.axvline(500, color="#A61B1B", linestyle=":", label="Frozen TTFT p95 bound")
    axis.set_xscale("log")
    axis.set_ylim(bottom=0)
    axis.set_xlabel("TTFT p95 (ms, logarithmic scale; lower is better)")
    axis.set_ylabel("Successful requests/s (higher is better)")
    axis.set_title("Throughput-tail-latency Pareto frontier (★ = fully SLO compliant)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(figure, "F09_serving_pareto_frontier.png")


def plot_knee(points: pd.DataFrame) -> None:
    columns = ["config_id", "concurrency", "successful_requests_per_s"]
    table = points[columns].copy()
    knees = {
        config_id: saturation_knee(group)
        for config_id, group in table.groupby("config_id", sort=True)
    }
    table["saturation_knee"] = table["config_id"].map(knees)
    table.to_parquet(TABLES / "F10_serving_saturation_knee.parquet", index=False)
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    for config_id, group in table.groupby("config_id", sort=True):
        knee = int(group["saturation_knee"].iloc[0])
        axis.plot(
            group["concurrency"],
            group["successful_requests_per_s"],
            color=COLORS[config_id],
            marker="o",
            label=f"{LABELS[config_id]} (knee={knee})",
        )
        knee_row = group.loc[group["concurrency"].eq(knee)].iloc[0]
        axis.scatter(
            [knee],
            [knee_row["successful_requests_per_s"]],
            s=150,
            facecolors="none",
            edgecolors=COLORS[config_id],
            linewidths=2,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Concurrency")
    axis.set_ylabel("Median successful requests/s")
    axis.set_title("Descriptive saturation knees (not the SLO-selected operating point)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(figure, "F10_serving_saturation_knee.png")


def plot_thermal(runs: pd.DataFrame) -> None:
    columns = [
        "run_id",
        "config_id",
        "replicate",
        "concurrency",
        "absolute_order",
        "temperature_median_c",
        "temperature_max_c",
        "sm_clock_median_mhz",
        "power_median_w",
        "utilization_median_pct",
        "throughput_residual_percent",
    ]
    table = runs[columns].sort_values("absolute_order").copy()
    table.to_parquet(TABLES / "F13_serving_gpu_thermal_power.parquet", index=False)
    figure, axes = plt.subplots(3, 1, figsize=(10.0, 8.0), sharex=True)
    for config_id, group in table.groupby("config_id", sort=True):
        axes[0].scatter(
            group["absolute_order"],
            group["temperature_median_c"],
            color=COLORS[config_id],
            label=LABELS[config_id],
        )
        axes[1].scatter(
            group["absolute_order"], group["sm_clock_median_mhz"], color=COLORS[config_id]
        )
        axes[2].scatter(group["absolute_order"], group["power_median_w"], color=COLORS[config_id])
    axes[0].set_ylabel("Median temp (°C)")
    axes[1].set_ylabel("Median SM clock (MHz)")
    axes[2].set_ylabel("Median power (W)")
    axes[2].set_xlabel("Predeclared qualification point order")
    axes[0].set_title("Serving GPU thermal, clock, and power diagnostics")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save_figure(figure, "F13_serving_gpu_thermal_power.png")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    points = pd.read_parquet(ANALYSIS / "serving_qualification_points.parquet")
    runs = pd.read_parquet(WAREHOUSE / "serving_runs.parquet")
    if len(points) != 12 or len(runs) != 36:
        raise ValueError("serving qualification figure inputs are incomplete")
    plot_throughput(points, runs)
    plot_ttft(points)
    plot_pareto(points)
    plot_knee(points)
    plot_thermal(runs)


if __name__ == "__main__":
    main()
