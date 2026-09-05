from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd


@dataclass(frozen=True)
class PromptContract:
    config_id: str
    system: str
    few_shot_count: int


def select_few_shots(frame: pd.DataFrame, count: int) -> list[dict[str, Any]]:
    fit = frame.loc[frame["split_role"].eq("FIT")].sort_values("example_id")
    if len(fit) < count:
        raise ValueError(f"requested {count} few-shot examples but FIT has {len(fit)}")
    return cast(list[dict[str, Any]], fit.head(count).to_dict(orient="records"))


def build_messages(
    question: str,
    contract: PromptContract,
    demonstrations: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if len(demonstrations) != contract.few_shot_count:
        raise ValueError("demonstration count violates prompt contract")
    messages = [{"role": "system", "content": contract.system}]
    for example in demonstrations:
        if example.get("split_role") != "FIT":
            raise ValueError("few-shot demonstrations must come from FIT")
        messages.extend(
            [
                {"role": "user", "content": str(example["question"])},
                {"role": "assistant", "content": str(example["raw_solution"])},
            ]
        )
    messages.append({"role": "user", "content": question})
    return messages
