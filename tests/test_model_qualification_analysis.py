from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.analyze_model_qualification as analysis


def _record(dataset: str, config: str, index: int) -> dict[str, object]:
    suffix = "m0" if config == "m0" else "mstar"
    return {
        "run_id": f"qual-sf-model-v3-{dataset}-{suffix}",
        "protocol_identity": "SF-MODEL-v3",
        "state": "QUALIFICATION",
        "dataset_id": dataset,
        "config_id": config,
        "example_id": f"{dataset}-{index}",
        "question": "question",
        "reference_solution": "solution",
        "reference_answer": "1",
        "prediction": "answer 1",
        "parsed_answer": "1",
        "parse_failure": None,
        "correct": True,
        "prompt_tokens": 8,
        "output_tokens": 3,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_qualification_rejects_incomplete_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analysis, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(analysis, "EXPECTED", {"gsm8k": 2})
    monkeypatch.setattr(analysis, "sha256_file", lambda path: "freeze")
    summary_root = tmp_path / "summaries"
    summary_root.mkdir()
    monkeypatch.setattr(analysis, "SUMMARY_ROOT", summary_root)
    _write(tmp_path / "qual-sf-model-v3-gsm8k-m0.jsonl", [_record("gsm8k", "m0", 0)])
    _write(
        tmp_path / "qual-sf-model-v3-gsm8k-mstar.jsonl",
        [_record("gsm8k", "mstar_lora_r32_attn_mlp", 0)],
    )
    for suffix in ("m0", "mstar"):
        run_id = f"qual-sf-model-v3-gsm8k-{suffix}"
        (summary_root / f"{run_id}.json").write_text(
            json.dumps({"run_id": run_id, "examples": 2, "freeze_manifest_sha256": "freeze"}),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="incomplete"):
        analysis.load_qualification()


def test_make_slices_preserves_paired_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analysis, "EXPECTED", {"gsm8k": 4})
    rows = []
    for index in range(4):
        for config in analysis.CONFIGS:
            record = _record("gsm8k", config, index)
            record["prompt_tokens"] = index + 1
            record["reference_solution_chars"] = index + 4
            record["correct"] = config == "m0" or index == 0
            rows.append(record)
    slices = analysis.make_slices(pd.DataFrame(rows))
    prompt = slices.loc[slices["slice_dimension"].eq("prompt_tokens_quartile")]
    assert int(prompt["n"].sum()) == 4
