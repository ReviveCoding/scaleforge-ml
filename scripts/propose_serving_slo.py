from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from scaleforge.evaluation.answers import parse_generated_answer
from scaleforge.protocol import sha256_file
from scaleforge.warehouse import validate_serving_requests

PROTOCOL = "SF-SERVE-v2"
CONCURRENCY = [1, 2, 4, 8, 16, 32]
RAW = Path("artifacts/raw/serving")
ANALYSIS = Path("artifacts/analysis/serving")
OUTPUT = Path("configs/serving/slo.yaml")


def round_up(value: float, increment: float) -> float:
    return math.ceil(value / increment) * increment


def main() -> None:
    frames = []
    input_hashes = {}
    for concurrency in CONCURRENCY:
        path = RAW / f"dev-sf-serve-v2-hf-c{concurrency}-r1-requests.jsonl"
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing HF baseline pilot artifact: {path}")
        input_hashes[str(path)] = sha256_file(path)
        frame = pd.read_json(path, lines=True)
        if (
            len(frame) != 72
            or int(frame["measured"].sum()) != 64
            or set(frame["state"]) != {"DEVELOPMENT"}
            or set(frame["config_id"]) != {"s0_hf_bf16"}
            or set(frame["concurrency"]) != {concurrency}
        ):
            raise ValueError(f"invalid HF baseline pilot cardinality: {path}")
        frames.append(frame)
    requests = pd.concat(frames, ignore_index=True)
    validate_serving_requests(requests)
    measured = requests.loc[requests["measured"]].copy()
    parsed = measured["prediction"].map(parse_generated_answer)
    measured["parsed_answer"] = parsed.map(lambda result: result.normalized)
    measured["parse_failure"] = parsed.map(lambda result: result.failure is not None)
    # ``read_json`` may infer a numerically shaped answer column as integer.  The
    # canonical evaluator compares normalized answer strings, so restore that
    # contract explicitly before computing exact match.
    measured["reference_final_answer"] = measured["reference_final_answer"].astype(str)
    measured["correct"] = measured["parsed_answer"].eq(measured["reference_final_answer"])
    rows = []
    for concurrency, group in measured.groupby("concurrency", sort=True):
        successes = group.loc[group["success"]]
        summary_path = ANALYSIS / f"dev-sf-serve-v2-hf-c{concurrency}-r1.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "config_id": "s0_hf_bf16",
                "replicate": 1,
                "concurrency": int(concurrency),
                "successful_requests_per_s": float(summary["successful_requests_per_s"]),
                "successful_output_tokens_per_s": float(summary["successful_output_tokens_per_s"]),
                "error_rate": float(summary["error_rate"]),
                "ttft_p50_ms": float(successes["ttft_ms"].quantile(0.50)),
                "ttft_p95_ms": float(successes["ttft_ms"].quantile(0.95)),
                "ttft_p99_ms": float(successes["ttft_ms"].quantile(0.99)),
                "tpot_p50_ms": float(successes["tpot_ms"].quantile(0.50)),
                "tpot_p95_ms": float(successes["tpot_ms"].quantile(0.95)),
                "e2e_p95_ms": float(successes["e2e_ms"].quantile(0.95)),
                "queue_p95_ms": float(group["queue_ms"].quantile(0.95)),
                "exact_match": float(group["correct"].mean()),
                "parse_failure_rate": float(group["parse_failure"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    eligible = table.loc[
        table["ttft_p95_ms"].le(500.0)
        & table["e2e_p95_ms"].le(30_000.0)
        & table["error_rate"].le(0.01)
    ]
    if eligible.empty:
        raise ValueError("the strong baseline has no viable interactive operating point")
    anchor = eligible.sort_values("concurrency").iloc[-1]
    slo = {
        "protocol_identity": PROTOCOL,
        "state": "FROZEN",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "ScaleForge experimental interactive SLO; not a Google SLO",
        "derivation": {
            "source_runtime": "s0_hf_bf16",
            "source_data": "development POLICY corpus only",
            "candidate_results_read": False,
            "selection_rule": (
                "Highest tested HF concurrency with TTFT p95 <= 500 ms, E2E p95 <= "
                "30000 ms, and error rate <= 1%. The E2E bound targets complete "
                "long-form math reasoning at the development-selected 512-token cap."
            ),
            "anchor_concurrency": int(anchor["concurrency"]),
            "headroom_rule": (
                "At least the intended bound and 125% of anchor tail metrics; round TTFT "
                "to 50 ms, TPOT to 5 ms, and E2E to 500 ms."
            ),
            "input_artifact_sha256": input_hashes,
        },
        "thresholds": {
            "ttft_p95_max_ms": round_up(max(500.0, 1.25 * float(anchor["ttft_p95_ms"])), 50.0),
            "tpot_p95_max_ms": round_up(1.25 * float(anchor["tpot_p95_ms"]), 5.0),
            "e2e_p95_max_ms": round_up(max(30_000.0, 1.25 * float(anchor["e2e_p95_ms"])), 500.0),
            "error_rate_max": 0.01,
            "quality_non_regression": "candidate_exact_match_gte_paired_baseline",
            "parse_failure_non_regression": "candidate_rate_lte_paired_baseline",
        },
        "qualification": {
            "replicates_per_point": 3,
            "warmup_requests": 8,
            "measured_requests": 64,
            "concurrency_grid": CONCURRENCY,
            "replicate_aware_analysis": True,
        },
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    table.to_parquet(ANALYSIS / "hf_baseline_slo_pilot.parquet", index=False)
    measured.to_parquet(ANALYSIS / "hf_baseline_slo_requests.parquet", index=False)
    OUTPUT.write_text(yaml.safe_dump(slo, sort_keys=False), encoding="utf-8")
    print(json.dumps({"baseline_table": rows, "slo": slo}, indent=2))


if __name__ == "__main__":
    main()
