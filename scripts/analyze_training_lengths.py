from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer

from scaleforge.data.lengths import choose_sequence_limit, quantiles
from scaleforge.training.examples import formatted_length


def main() -> None:
    baseline_config = yaml.safe_load(
        Path("configs/model/baselines.yaml").read_text(encoding="utf-8")
    )
    data_config = yaml.safe_load(Path("configs/data/gsm8k.yaml").read_text(encoding="utf-8"))
    frame = pd.read_parquet("artifacts/data/split_manifest.parquet")
    fit = frame.loc[frame["split_role"].eq("FIT")]
    tokenizer = AutoTokenizer.from_pretrained(
        baseline_config["model"]["repository"],
        revision=baseline_config["model"]["revision"],
        cache_dir=Path("artifacts/cache/huggingface"),
    )
    lengths = [
        formatted_length(
            tokenizer,
            system=baseline_config["prompts"]["m0"]["system"],
            question=str(row.question),
            response=str(row.raw_solution),
        )
        for row in fit.itertuples()
    ]
    summary = quantiles(lengths)
    selected = choose_sequence_limit(
        summary["max"],
        float(data_config["sequence_headroom_ratio"]),
        int(data_config["sequence_rounding_multiple"]),
        int(data_config["maximum_allowed_sequence_length"]),
    )
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "model_revision": baseline_config["model"]["revision"],
        "chat_template_revision": baseline_config["model"]["revision"],
        "split_role": "FIT",
        "examples": len(fit),
        "formatted_training_token_lengths": summary,
        "selected_sequence_length": selected,
        "headroom_ratio": data_config["sequence_headroom_ratio"],
        "rounding_multiple": data_config["sequence_rounding_multiple"],
        "silent_truncation_allowed": False,
        "cuda_used": False,
    }
    output = Path("artifacts/data/training_length_quality.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
