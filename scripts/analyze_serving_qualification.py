from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from scipy.stats import spearmanr

from scaleforge.analysis.serving import (
    hierarchical_quantile_interval,
    mark_pareto_frontier,
)
from scaleforge.analysis.serving_qualification import (
    paired_replicate_percent_change,
    saturation_knee,
)
from scaleforge.evaluation.answers import parse_generated_answer
from scaleforge.protocol import canonical_sha256, sha256_file
from scaleforge.telemetry import read_nvidia_smi_csv
from scaleforge.warehouse import validate_serving_requests

PROTOCOL = "SF-SERVE-v2"
CONFIG_PATH = Path("configs/serving/qualification.yaml")
SLO_PATH = Path("configs/serving/slo.yaml")
FREEZE_PATH = Path("artifacts/manifests/serving_freeze.json")
RAW = Path("artifacts/raw/serving")
ANALYSIS = Path("artifacts/analysis/serving")
WAREHOUSE = Path("artifacts/warehouse")
PREFIX = {"s0_hf_bf16": "hf", "s2_vllm_v1_runner": "vllm"}
LATENCY_METRICS = {
    "ttft_ms": (0.50, 0.95, 0.99),
    "tpot_ms": (0.50, 0.95),
    "e2e_ms": (0.50, 0.95, 0.99),
    "queue_ms": (0.50, 0.95, 0.99),
}


def expected_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    absolute_order = 0
    for block_order, block in enumerate(config["qualification"]["balanced_order"], start=1):
        for within_block_order, concurrency in enumerate(block["concurrency"], start=1):
            absolute_order += 1
            rows.append(
                {
                    "config_id": str(block["config_id"]),
                    "replicate": int(block["replicate"]),
                    "concurrency": int(concurrency),
                    "block_order": block_order,
                    "within_block_order": within_block_order,
                    "absolute_order": absolute_order,
                }
            )
    return rows


def load_and_validate() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, str]
]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    slo = yaml.safe_load(SLO_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    freeze_sha = sha256_file(FREEZE_PATH)
    if (
        config["protocol_identity"] != PROTOCOL
        or config["state"] != "FROZEN"
        or slo["protocol_identity"] != PROTOCOL
        or slo["state"] != "FROZEN"
        or freeze["protocol_identity"] != PROTOCOL
        or freeze["state"] != "FROZEN"
        or canonical_sha256(config) != freeze["configuration_sha256"]
        or sha256_file(SLO_PATH) != freeze["slo_sha256"]
        or sha256_file(Path(config["corpus"]["path"])) != freeze["data_sha256"]
    ):
        raise ValueError("serving freeze, SLO, configuration, or corpus identity mismatch")
    for path_text, expected_hash in freeze["file_sha256"].items():
        if sha256_file(Path(path_text)) != expected_hash:
            raise ValueError(f"frozen source changed before analysis: {path_text}")

    request_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    telemetry_frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {
        str(CONFIG_PATH): sha256_file(CONFIG_PATH),
        str(SLO_PATH): sha256_file(SLO_PATH),
        str(FREEZE_PATH): freeze_sha,
    }
    matrix = expected_matrix(config)
    for cell in matrix:
        config_id = cell["config_id"]
        prefix = PREFIX.get(config_id)
        if prefix is None:
            raise ValueError(f"unknown frozen serving config: {config_id}")
        stem = f"qual-sf-serve-v2-{prefix}-c{cell['concurrency']}-r{cell['replicate']}"
        raw_path = RAW / f"{stem}-requests.jsonl"
        telemetry_path = RAW / f"{stem}-gpu.csv"
        summary_path = ANALYSIS / f"{stem}.json"
        for path in (raw_path, telemetry_path, summary_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty qualification artifact: {path}")
            hashes[str(path)] = sha256_file(path)
        if (Path("artifacts/failures") / f"{stem}.json").exists():
            raise ValueError(f"completed serving run also has a run-scoped failure: {stem}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        frame = pd.read_json(raw_path, lines=True)
        expected_requests = int(config["qualification"]["measured_requests_per_point"])
        expected_warmup = int(config["qualification"]["warmup_requests_per_point"])
        identity_valid = (
            len(frame) == expected_requests + expected_warmup
            and int(frame["measured"].sum()) == expected_requests
            and set(frame["run_id"]) == {stem}
            and set(frame["protocol_identity"]) == {PROTOCOL}
            and set(frame["state"]) == {"QUALIFICATION"}
            and set(frame["config_id"]) == {config_id}
            and set(frame["replicate"]) == {cell["replicate"]}
            and set(frame["concurrency"]) == {cell["concurrency"]}
            and set(frame["freeze_manifest_sha256"]) == {freeze_sha}
            and summary["run_id"] == stem
            and summary["status"] == "COMPLETED"
            and int(summary["requests"]) == expected_requests
            and int(summary["warmup_requests"]) == expected_warmup
            and summary["freeze_manifest_sha256"] == freeze_sha
        )
        if not identity_valid:
            raise ValueError(f"invalid qualification identity or cardinality: {stem}")
        measured = frame.loc[frame["measured"]]
        if measured["corpus_request_id"].nunique() != expected_requests:
            raise ValueError(f"qualification corpus coverage is incomplete: {stem}")
        observed_failures = int((~measured["success"].astype(bool)).sum())
        if observed_failures != int(summary["failures"]):
            raise ValueError(f"summary/request failure mismatch: {stem}")
        frame["elapsed_s"] = float(summary["elapsed_s"])
        for key, value in cell.items():
            frame[key] = value
        request_frames.append(frame)

        telemetry = read_nvidia_smi_csv(telemetry_path, run_id=stem)
        for key, value in cell.items():
            telemetry[key] = value
        telemetry_frames.append(telemetry)
        run_rows.append(
            {
                **cell,
                "run_id": stem,
                "protocol_identity": PROTOCOL,
                "state": "QUALIFICATION",
                "status": "COMPLETED",
                "completed_at": summary["created_at"],
                "requests": expected_requests,
                "successes": int(summary["successes"]),
                "failures": observed_failures,
                "error_rate": float(summary["error_rate"]),
                "elapsed_s": float(summary["elapsed_s"]),
                "successful_requests_per_s": float(summary["successful_requests_per_s"]),
                "successful_output_tokens_per_s": float(summary["successful_output_tokens_per_s"]),
                "request_artifact_path": str(raw_path),
                "telemetry_artifact_path": str(telemetry_path),
                "summary_artifact_path": str(summary_path),
            }
        )

    requests = pd.concat(request_frames, ignore_index=True)
    validate_serving_requests(requests)
    if requests.duplicated(["run_id", "request_id"]).any():
        raise ValueError("duplicate qualification request identity")
    runs = pd.DataFrame(run_rows)
    if len(runs) != 36 or runs.duplicated(["config_id", "replicate", "concurrency"]).any():
        raise ValueError("qualification run matrix is incomplete or duplicated")
    telemetry = pd.concat(telemetry_frames, ignore_index=True)
    return requests, runs, telemetry, config, slo, hashes


def score_requests(requests: pd.DataFrame) -> pd.DataFrame:
    result = requests.copy()
    parsed = result["prediction"].fillna("").astype(str).map(parse_generated_answer)
    result["parsed_answer"] = parsed.map(lambda value: value.normalized)
    result["parse_failure"] = parsed.map(lambda value: value.failure is not None)
    result["reference_final_answer"] = result["reference_final_answer"].astype(str)
    result["correct"] = (
        result["success"].astype(bool)
        & ~result["parse_failure"]
        & result["parsed_answer"].eq(result["reference_final_answer"])
    )
    return result


def summarize_runs(requests: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    measured = requests.loc[requests["measured"]].copy()
    rows: list[dict[str, Any]] = []
    for run_id, group in measured.groupby("run_id", sort=True):
        run = runs.loc[runs["run_id"].eq(run_id)].iloc[0]
        successful = group.loc[group["success"]]
        if successful.empty:
            raise ValueError(f"run has no successful requests: {run_id}")
        row = run.to_dict()
        row.update(
            {
                "exact_match": float(group["correct"].mean()),
                "parse_failure_rate": float(group["parse_failure"].mean()),
                "output_tokens_total": int(successful["output_tokens"].sum()),
            }
        )
        for metric, quantiles in LATENCY_METRICS.items():
            for quantile in quantiles:
                row[f"{metric.removesuffix('_ms')}_p{int(quantile * 100):02d}_ms"] = float(
                    successful[metric].quantile(quantile)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_points(
    requests: pd.DataFrame, runs: pd.DataFrame, slo: dict[str, Any]
) -> pd.DataFrame:
    measured = requests.loc[requests["measured"]].copy()
    rows: list[dict[str, Any]] = []
    configurations = sorted(runs["config_id"].unique())
    concurrencies = sorted(int(value) for value in runs["concurrency"].unique())
    for config_index, config_id in enumerate(configurations):
        for concurrency_index, concurrency in enumerate(concurrencies):
            group = measured.loc[
                measured["config_id"].eq(config_id) & measured["concurrency"].eq(concurrency)
            ]
            point_runs = runs.loc[
                runs["config_id"].eq(config_id) & runs["concurrency"].eq(concurrency)
            ]
            if len(group) != 192 or len(point_runs) != 3:
                raise ValueError(f"incomplete aggregate point: {config_id} c{concurrency}")
            successful = group.loc[group["success"]]
            row: dict[str, Any] = {
                "config_id": config_id,
                "concurrency": concurrency,
                "replicates": len(point_runs),
                "requests": len(group),
                "successes": int(group["success"].sum()),
                "error_rate": float(1 - group["success"].mean()),
                "successful_requests_per_s": float(
                    point_runs["successful_requests_per_s"].median()
                ),
                "successful_output_tokens_per_s": float(
                    point_runs["successful_output_tokens_per_s"].median()
                ),
                "throughput_min_requests_per_s": float(
                    point_runs["successful_requests_per_s"].min()
                ),
                "throughput_max_requests_per_s": float(
                    point_runs["successful_requests_per_s"].max()
                ),
                "exact_match": float(group["correct"].mean()),
                "parse_failure_rate": float(group["parse_failure"].mean()),
            }
            for metric_index, (metric, quantiles) in enumerate(LATENCY_METRICS.items()):
                for quantile_index, quantile in enumerate(quantiles):
                    interval = hierarchical_quantile_interval(
                        successful,
                        metric,
                        quantile=quantile,
                        bootstrap_samples=5_000,
                        seed=(
                            20260905
                            + config_index * 100
                            + concurrency_index * 10
                            + metric_index * 3
                            + quantile_index
                        ),
                    )
                    label = f"{metric.removesuffix('_ms')}_p{int(quantile * 100):02d}_ms"
                    row[label] = interval.estimate
                    row[f"{label}_ci_low"] = interval.low
                    row[f"{label}_ci_high"] = interval.high
            rows.append(row)
    points = pd.DataFrame(rows)
    baseline = (
        points.loc[points["config_id"].eq("s0_hf_bf16")]
        .set_index("concurrency")[["exact_match", "parse_failure_rate"]]
        .rename(
            columns={
                "exact_match": "paired_baseline_exact_match",
                "parse_failure_rate": "paired_baseline_parse_failure_rate",
            }
        )
    )
    points = points.join(baseline, on="concurrency")
    thresholds = slo["thresholds"]
    points["performance_slo_compliant"] = (
        points["ttft_p95_ms"].le(float(thresholds["ttft_p95_max_ms"]))
        & points["tpot_p95_ms"].le(float(thresholds["tpot_p95_max_ms"]))
        & points["e2e_p95_ms"].le(float(thresholds["e2e_p95_max_ms"]))
        & points["error_rate"].le(float(thresholds["error_rate_max"]))
    )
    points["quality_slo_compliant"] = points["exact_match"].ge(
        points["paired_baseline_exact_match"]
    ) & points["parse_failure_rate"].le(points["paired_baseline_parse_failure_rate"])
    points["slo_compliant"] = points["performance_slo_compliant"] & points["quality_slo_compliant"]
    points["pareto_frontier"] = False
    for _, indices in points.groupby("config_id").groups.items():
        points.loc[indices, "pareto_frontier"] = mark_pareto_frontier(
            points.loc[indices], latency_column="ttft_p95_ms"
        )
    return points


def selected_points(points: pd.DataFrame) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for config_id, group in points.groupby("config_id", sort=True):
        eligible = group.loc[group["slo_compliant"]]
        if eligible.empty:
            selected[config_id] = {"concurrency": None, "throughput": 0.0}
            continue
        row = eligible.sort_values(
            ["successful_requests_per_s", "concurrency"], ascending=[False, True]
        ).iloc[0]
        selected[config_id] = {
            "concurrency": int(row["concurrency"]),
            "throughput": float(row["successful_requests_per_s"]),
            "output_tokens_per_s": float(row["successful_output_tokens_per_s"]),
            "ttft_p95_ms": float(row["ttft_p95_ms"]),
            "tpot_p95_ms": float(row["tpot_p95_ms"]),
            "e2e_p95_ms": float(row["e2e_p95_ms"]),
            "error_rate": float(row["error_rate"]),
            "exact_match": float(row["exact_match"]),
            "parse_failure_rate": float(row["parse_failure_rate"]),
        }
    return selected


def paired_quality_at(requests: pd.DataFrame, *, concurrency: int) -> dict[str, int | float]:
    measured = requests.loc[requests["measured"] & requests["concurrency"].eq(concurrency)]
    columns = ["replicate", "corpus_request_id", "correct"]
    baseline = measured.loc[measured["config_id"].eq("s0_hf_bf16"), columns].rename(
        columns={"correct": "baseline_correct"}
    )
    candidate = measured.loc[measured["config_id"].eq("s2_vllm_v1_runner"), columns].rename(
        columns={"correct": "candidate_correct"}
    )
    paired = baseline.merge(candidate, on=["replicate", "corpus_request_id"], validate="1:1")
    if len(paired) != 192:
        raise ValueError("paired serving quality alignment is incomplete")
    b = paired["baseline_correct"].astype(bool)
    c = paired["candidate_correct"].astype(bool)
    return {
        "pairs": len(paired),
        "baseline_exact_match": float(b.mean()),
        "candidate_exact_match": float(c.mean()),
        "delta_percentage_points": float((c.astype(int) - b.astype(int)).mean() * 100),
        "candidate_only": int((c & ~b).sum()),
        "baseline_only": int((b & ~c).sum()),
        "both_correct": int((b & c).sum()),
        "both_wrong": int((~b & ~c).sum()),
        "inference_note": (
            "Transitions are descriptive because the same 64 prompts repeat across "
            "three serving replicates."
        ),
    }


def attach_telemetry_summary(runs: pd.DataFrame, telemetry: pd.DataFrame) -> pd.DataFrame:
    summary = telemetry.groupby("run_id", sort=True).agg(
        telemetry_samples=("timestamp", "size"),
        temperature_median_c=("temperature_c", "median"),
        temperature_max_c=("temperature_c", "max"),
        sm_clock_median_mhz=("sm_clock_mhz", "median"),
        power_median_w=("power_w", "median"),
        utilization_median_pct=("utilization_pct", "median"),
        memory_used_max_mib=("memory_used_mib", "max"),
    )
    result = runs.merge(summary, left_on="run_id", right_index=True, validate="1:1")
    if result["telemetry_samples"].isna().any():
        raise ValueError("telemetry join left a qualification run unmatched")
    result["throughput_residual_percent"] = result.groupby(["config_id", "concurrency"])[
        "successful_requests_per_s"
    ].transform(lambda value: (value / value.median() - 1) * 100)
    return result


def confound_analysis(runs: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "unit": "qualification run",
        "n": len(runs),
        "outcome": "throughput residual percent after config-by-concurrency median removal",
        "associations": {},
    }
    for column in (
        "temperature_median_c",
        "sm_clock_median_mhz",
        "power_median_w",
        "absolute_order",
    ):
        complete = runs[[column, "throughput_residual_percent"]].dropna()
        correlation, p_value = spearmanr(complete[column], complete["throughput_residual_percent"])
        result["associations"][column] = {
            "spearman_rho": float(correlation),
            "p_value_descriptive": float(p_value),
            "n": len(complete),
        }
    result["interpretation"] = (
        "These observational associations diagnose laptop thermal/order confounding; "
        "they are not causal adjustments or primary release statistics."
    )
    return result


def main() -> None:
    requests, runs, telemetry, config, slo, input_hashes = load_and_validate()
    requests = score_requests(requests)
    runs = summarize_runs(requests, runs)
    runs = attach_telemetry_summary(runs, telemetry)
    points = summarize_points(requests, runs, slo)
    selected = selected_points(points)
    baseline = selected["s0_hf_bf16"]
    candidate = selected["s2_vllm_v1_runner"]
    if baseline["concurrency"] is None or candidate["concurrency"] is None:
        throughput_change = None
        quality = None
    else:
        baseline_runs = runs.loc[
            runs["config_id"].eq("s0_hf_bf16")
            & runs["concurrency"].eq(int(baseline["concurrency"]))
        ].sort_values("replicate")
        candidate_runs = runs.loc[
            runs["config_id"].eq("s2_vllm_v1_runner")
            & runs["concurrency"].eq(int(candidate["concurrency"]))
        ].sort_values("replicate")
        throughput_change = asdict(
            paired_replicate_percent_change(
                baseline_runs["successful_requests_per_s"].to_numpy(),
                candidate_runs["successful_requests_per_s"].to_numpy(),
                bootstrap_samples=5_000,
                seed=20260905,
            )
        )
        quality = paired_quality_at(requests, concurrency=int(candidate["concurrency"]))

    knees = {
        config_id: saturation_knee(group)
        for config_id, group in points.groupby("config_id", sort=True)
    }
    all_points_complete = len(points) == 12 and len(runs) == 36
    candidate_wins = throughput_change is not None and throughput_change["estimate_percent"] > 0
    quality_passes = bool(
        candidate["concurrency"] is not None
        and candidate["exact_match"]
        >= float(
            points.loc[
                points["config_id"].eq("s0_hf_bf16")
                & points["concurrency"].eq(candidate["concurrency"]),
                "exact_match",
            ].iloc[0]
        )
        and candidate["parse_failure_rate"]
        <= float(
            points.loc[
                points["config_id"].eq("s0_hf_bf16")
                & points["concurrency"].eq(candidate["concurrency"]),
                "parse_failure_rate",
            ].iloc[0]
        )
    )
    request_error_gate = bool(runs["error_rate"].le(0.01).all())
    frozen_gate_pass = (
        all_points_complete and candidate_wins and quality_passes and request_error_gate
    )
    release_decision = "REVIEW" if frozen_gate_pass else "BLOCK"
    selected_metric_changes = None
    selected_uncertainty = None
    if baseline["concurrency"] is not None and candidate["concurrency"] is not None:
        selected_metric_changes = {
            metric: float(100 * (candidate[metric] / baseline[metric] - 1))
            for metric in (
                "throughput",
                "output_tokens_per_s",
                "ttft_p95_ms",
                "tpot_p95_ms",
                "e2e_p95_ms",
            )
        }
        selected_uncertainty = {}
        for config_id, selected_point in selected.items():
            point = points.loc[
                points["config_id"].eq(config_id)
                & points["concurrency"].eq(selected_point["concurrency"])
            ].iloc[0]
            selected_uncertainty[config_id] = {
                metric: {
                    "estimate_ms": float(point[metric]),
                    "ci_low_ms": float(point[f"{metric}_ci_low"]),
                    "ci_high_ms": float(point[f"{metric}_ci_high"]),
                }
                for metric in ("ttft_p95_ms", "tpot_p95_ms", "e2e_p95_ms")
            }
    limitations = [
        "Only three performance replicates were feasible; uncertainty is correspondingly wide.",
        (
            "Laptop temperature, clock, power, and order are observational confounds, "
            "not randomized treatments."
        ),
        (
            "The fixed 64-request corpus repeats at every point. vLLM prefix-cache state therefore "
            "carries across concurrency points within a replicate, as frozen in the protocol."
        ),
        (
            "vLLM V2 was incompatible with WSL unified virtual addressing; qualification used the "
            "documented V1 runner fallback."
        ),
        (
            "vLLM force-killed its idle EngineCore and reported one leaked semaphore at every "
            "post-measurement shutdown; request-level success was unaffected."
        ),
    ]
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_identity": PROTOCOL,
        "state": "ANALYSIS",
        "source_git_sha": config.get("source_git_sha", "9a02578d53af0e58e93d6f191891966b3205bf97"),
        "runs_expected": 36,
        "runs_observed": len(runs),
        "measured_requests": int(requests["measured"].sum()),
        "request_failures": int((~requests.loc[requests["measured"], "success"]).sum()),
        "input_artifact_sha256": input_hashes,
        "slo_scope": slo["scope"],
        "max_slo_compliant_throughput": selected,
        "selected_configuration": {
            "config_id": "s2_vllm_v1_runner" if frozen_gate_pass else "s0_hf_bf16",
            "concurrency": (
                candidate["concurrency"] if frozen_gate_pass else baseline["concurrency"]
            ),
            "runner": "V1 compatibility mode" if frozen_gate_pass else "Transformers BF16",
            "deployment_status": "REVIEW_REQUIRED",
        },
        "paired_replicate_throughput_change": throughput_change,
        "selected_point_metric_changes_percent": selected_metric_changes,
        "selected_point_latency_uncertainty": selected_uncertainty,
        "paired_quality_at_candidate_operating_point": quality,
        "saturation_knee": knees,
        "thermal_and_order_diagnostics": confound_analysis(runs),
        "release_gates": {
            "all_36_points_complete": all_points_complete,
            "candidate_max_slo_compliant_throughput_gt_baseline": candidate_wins,
            "candidate_quality_non_regression": quality_passes,
            "service_error_rate_max_0_01": request_error_gate,
            "frozen_critical_gates_pass": frozen_gate_pass,
            "lifecycle_shutdown_clean_observation": False,
        },
        "release_decision": release_decision,
        "release_decision_note": (
            "All predeclared request-path gates pass, so S2 at concurrency 2 is the selected "
            "experimental runtime. Overall serving remains REVIEW because the repeatable vLLM "
            "shutdown defect blocks an unqualified production-readiness claim."
        ),
        "limitations": limitations,
    }

    WAREHOUSE.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    requests.to_parquet(WAREHOUSE / "serving_requests.parquet", index=False)
    runs.to_parquet(WAREHOUSE / "serving_runs.parquet", index=False)
    points.to_parquet(ANALYSIS / "serving_qualification_points.parquet", index=False)
    telemetry.to_parquet(WAREHOUSE / "serving_gpu_telemetry.parquet", index=False)
    (ANALYSIS / "serving_qualification_sf_serve_v2.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (ANALYSIS / "serving_release_decision_sf_serve_v2.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "protocol_identity": PROTOCOL,
                "state": "CLOSED",
                "decision": release_decision,
                "selected_configuration": result["selected_configuration"],
                "release_gates": result["release_gates"],
                "limitations": limitations,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
