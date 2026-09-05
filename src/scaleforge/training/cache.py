from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from scaleforge.protocol import canonical_sha256
from scaleforge.training.examples import format_response_only_example


def tensor_cache(
    frame: pd.DataFrame,
    tokenizer: Any,
    *,
    system: str,
    sequence_length: int,
    fingerprint: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_key = canonical_sha256(
        {
            "fingerprint": fingerprint,
            "system": system,
            "sequence_length": sequence_length,
            "tokenizer": tokenizer.name_or_path,
            "example_ids": frame["example_id"].astype(str).tolist(),
        }
    )
    cache_path = Path("artifacts/cache/training") / f"response_only_{cache_key}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        return payload["input_ids"], payload["attention_mask"], payload["labels"]
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    labels: list[list[int]] = []
    for row in frame.itertuples():
        example = format_response_only_example(
            tokenizer,
            system=system,
            question=str(row.question),
            response=str(row.raw_solution),
            max_length=sequence_length,
        )
        input_ids.append(example.input_ids)
        attention_masks.append(example.attention_mask)
        labels.append(example.labels)
    payload = {
        "input_ids": torch.tensor(input_ids, dtype=torch.int32),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.bool),
        "labels": torch.tensor(labels, dtype=torch.int32),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload["input_ids"], payload["attention_mask"], payload["labels"]
