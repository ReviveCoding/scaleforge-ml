from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scaleforge.analysis.serving import mark_pareto_frontier
from scaleforge.evaluation.answers import parse_generated_answer
from scaleforge.protocol import sha256_file
from scaleforge.warehouse import validate_serving_requests

PROTOCOL = "SF-SERVE-v2"
CONFIGS = ("s0_hf_bf16", "s2_vllm_v1_runner")
PREFIX = {"s0_hf_bf16": "hf", "s2_vllm_v1_runner": "vllm"}
CONCURRENCY = (1, 2, 4, 8, 16, 32)
RAW = Path("artifacts/raw/serving")
ANALYSIS = Path("artifacts/analysis/serving")


def load_inputs() -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for config_id in CONFIGS:
        for concurrency in CONCURRENCY:
            stem = f"dev-sf-serve-v2-{PREFIX[config_id]}-c{concurrency}-r1"
            raw_path = RAW / f"{stem}-requests.jsonl"
            summary_path = ANALYSIS / f"{stem}.json"
            if not raw_path.is_file() or not summary_path.is_file():
                raise ValueError(f"missing serving development artifacts: {stem}")
            hashes[str(raw_path)] = sha256_file(raw_path)
            hashes[str(summary_path)] = sha256_file(summary_path)
            frame = pd.read_json(raw_path, lines=True)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                len(frame) != 72
                or int(frame["measured"].sum()) != 64
                or set(frame["protocol_identity"]) != {PROTOCOL}
                or set(frame["state"]) != {"DEVELOPMENT"}
                or set(frame["config_id"]) != {config_id}
                or set(frame["concurrency"]) != {concurrency}
                or summary["status"] != "COMPLETED"
                or summary["failures"] != 0
            ):
                raise ValueError(f"invalid serving development run: {stem}")
            frame["elapsed_s"] = float(summary["elapsed_s"])
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    validate_serving_requests(result)
    if result.duplicated(["config_id", "concurrency", "request_id"]).any():
        raise ValueError("duplicate serving development request identity")
    return result, hashes


def summarize(requests: pd.DataFrame, slo: dict[str, Any]) -> pd.DataFrame:
    measured = requests.loc[requests["measured"]].copy()
    parsed = measured["prediction"].map(parse_generated_answer)
    measured["parsed_answer"] = parsed.map(lambda value: value.normalized)
    measured["parse_failure"] = parsed.map(lambda value: value.failure is not None)
    measured["reference_final_answer"] = measured["reference_final_answer"].astype(str)
    measured["correct"] = measured["parsed_answer"].eq(measured["reference_final_answer"])
    rows: list[dict[str, Any]] = []
    for (config_id, concurrency), group in measured.groupby(
        ["config_id", "concurrency"], sort=True
    ):
        successful = group.loc[group["success"]]
        elapsed_s = float(group["elapsed_s"].iloc[0])
        rows.append(
            {
                "config_id": config_id,
                "concurrency": int(concurrency),
                "requests": len(group),
                "successes": int(group["success"].sum()),
                "error_rate": float(1 - group["success"].mean()),
                "successful_requests_per_s": float(group["success"].sum() / elapsed_s),
                "successful_output_tokens_per_s": float(
                    successful["output_tokens"].sum() / elapsed_s
                ),
                "ttft_p50_ms": float(successful["ttft_ms"].quantile(0.50)),
                "ttft_p95_ms": float(successful["ttft_ms"].quantile(0.95)),
                "ttft_p99_ms": float(successful["ttft_ms"].quantile(0.99)),
                "tpot_p50_ms": float(successful["tpot_ms"].quantile(0.50)),
                "tpot_p95_ms": float(successful["tpot_ms"].quantile(0.95)),
                "e2e_p50_ms": float(successful["e2e_ms"].quantile(0.50)),
                "e2e_p95_ms": float(successful["e2e_ms"].quantile(0.95)),
                "e2e_p99_ms": float(successful["e2e_ms"].quantile(0.99)),
                "queue_p95_ms": float(group["queue_ms"].quantile(0.95)),
                "exact_match": float(group["correct"].mean()),
                "parse_failure_rate": float(group["parse_failure"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    baseline_quality = (
        table.loc[table["config_id"].eq("s0_hf_bf16")]
        .set_index("concurrency")[["exact_match", "parse_failure_rate"]]
        .rename(
            columns={
                "exact_match": "paired_baseline_exact_match",
                "parse_failure_rate": "paired_baseline_parse_failure_rate",
            }
        )
    )
    table = table.join(baseline_quality, on="concurrency")
    thresholds = slo["thresholds"]
    table["performance_slo_compliant"] = (
        table["ttft_p95_ms"].le(float(thresholds["ttft_p95_max_ms"]))
        & table["tpot_p95_ms"].le(float(thresholds["tpot_p95_max_ms"]))
        & table["e2e_p95_ms"].le(float(thresholds["e2e_p95_max_ms"]))
        & table["error_rate"].le(float(thresholds["error_rate_max"]))
    )
    table["quality_slo_compliant"] = table["exact_match"].ge(
        table["paired_baseline_exact_match"]
    ) & table["parse_failure_rate"].le(table["paired_baseline_parse_failure_rate"])
    table["slo_compliant"] = table["performance_slo_compliant"] & table["quality_slo_compliant"]
    table["pareto_frontier"] = mark_pareto_frontier(table, latency_column="ttft_p95_ms")
    return table


def main() -> None:
    slo = yaml.safe_load(Path("configs/serving/slo.yaml").read_text(encoding="utf-8"))
    if slo["protocol_identity"] != PROTOCOL or slo["state"] != "FROZEN":
        raise ValueError("serving SLO must be frozen before candidate analysis")
    requests, input_hashes = load_inputs()
    table = summarize(requests, slo)
    selected_rows: dict[str, dict[str, Any]] = {}
    for config_id in CONFIGS:
        eligible = table.loc[table["config_id"].eq(config_id) & table["slo_compliant"]]
        if eligible.empty:
            selected_rows[config_id] = {"concurrency": None, "throughput": 0.0}
            continue
        row = eligible.sort_values("successful_requests_per_s").iloc[-1]
        selected_rows[config_id] = {
            "concurrency": int(row["concurrency"]),
            "throughput": float(row["successful_requests_per_s"]),
            "output_tokens_per_s": float(row["successful_output_tokens_per_s"]),
            "ttft_p95_ms": float(row["ttft_p95_ms"]),
            "tpot_p95_ms": float(row["tpot_p95_ms"]),
            "e2e_p95_ms": float(row["e2e_p95_ms"]),
            "exact_match": float(row["exact_match"]),
        }
    baseline = selected_rows["s0_hf_bf16"]
    candidate = selected_rows["s2_vllm_v1_runner"]
    if candidate["throughput"] <= baseline["throughput"]:
        raise ValueError("vLLM did not improve development SLO-compliant throughput")
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_identity": PROTOCOL,
        "state": "CANDIDATE_SELECTION",
        "slo_sha256": sha256_file(Path("configs/serving/slo.yaml")),
        "input_artifact_sha256": input_hashes,
        "runs_expected": 12,
        "runs_observed": int(table.shape[0]),
        "request_failures": int((~requests.loc[requests["measured"], "success"]).sum()),
        "max_slo_compliant": selected_rows,
        "development_throughput_gain_percent": float(
            100 * (candidate["throughput"] / baseline["throughput"] - 1)
        ),
        "selection": {
            "baseline": "s0_hf_bf16",
            "challenger": "s2_vllm_v1_runner",
            "selected_finalist": "s2_vllm_v1_runner",
            "decision": "ADVANCE_TO_QUALIFICATION",
            "s1_hf_compile": "NOT_ADVANCED_NO_SERVING_BOTTLENECK_EVIDENCE",
            "s3_vllm_tuning": (
                "NOT_ADVANCED_S2_ALREADY_CLEARS_FROZEN_SLO_AND_NO_SPECIFIC_TUNING_HYPOTHESIS"
            ),
        },
        "limitations": [
            "Single development replicate; all numeric claims require frozen qualification.",
            (
                "Development points shared one server process and may include order or "
                "prefix-cache carryover."
            ),
            (
                "Qualification must restart the server before every replicate and use "
                "balanced runtime order."
            ),
        ],
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    requests.to_parquet(ANALYSIS / "serving_development_requests.parquet", index=False)
    table.to_parquet(ANALYSIS / "serving_development_points.parquet", index=False)
    (ANALYSIS / "serving_development_selection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"points": table.to_dict("records"), "result": result}, indent=2))


if __name__ == "__main__":
    main()
