from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scaleforge.protocol import sha256_file

SOURCE = Path("artifacts/raw/model/dev-baselines-sf-model-v1-b16.jsonl")
OUTPUT = Path("artifacts/analysis/serving/token_budget_development.json")


def main() -> None:
    frame = pd.read_json(SOURCE, lines=True)
    frame = frame.loc[frame["config_id"].eq("m0")].copy()
    if len(frame) != 723 or set(frame["state"]) != {"DEVELOPMENT"}:
        raise ValueError("expected the complete M0 development-validation prediction set")
    lengths = frame["output_tokens"].astype(int)
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "state": "DEVELOPMENT",
        "source": str(SOURCE),
        "source_sha256": sha256_file(SOURCE),
        "rows": len(frame),
        "output_token_percentiles": {
            name: float(lengths.quantile(quantile))
            for name, quantile in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99))
        },
        "max": int(lengths.max()),
        "truncated_at_512": int(lengths.eq(512).sum()),
        "completion_fraction_by_cap": {
            str(cap): float(lengths.lt(cap).mean()) for cap in (128, 256, 384, 512)
        },
        "correct_completion_fraction_by_cap": {
            str(cap): float((lengths.lt(cap) & frame["correct"].astype(bool)).mean())
            for cap in (128, 256, 384, 512)
        },
        "selection_rule": (
            "Choose the smallest cap covering at least 95% of observed complete development "
            "outputs; never infer protected-set adequacy."
        ),
        "protected_data_used": False,
    }
    eligible = [
        cap
        for cap in (128, 256, 384, 512)
        if result["completion_fraction_by_cap"][str(cap)] >= 0.95
    ]
    if not eligible:
        raise ValueError("no serving token cap covers 95% of development outputs")
    result["selected_max_new_tokens"] = min(eligible)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
