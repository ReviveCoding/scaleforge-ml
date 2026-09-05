from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from scaleforge.protocol import sha256_file
from scripts.load_test import load_and_validate_corpus


def write_corpus(root: Path, *, split_role: str = "POLICY", prompt_tokens: int = 10) -> Path:
    corpus_path = root / "artifacts/data/serving_request_corpus.parquet"
    corpus_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "request_id": ["r1"],
            "example_id": ["e1"],
            "rendered_prompt": ["prompt"],
            "prompt_tokens": [prompt_tokens],
            "normalized_final_answer": ["1"],
            "split_role": [split_role],
        }
    ).to_parquet(corpus_path, index=False)
    manifest_path = root / "artifacts/manifests/serving_corpus.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "path": str(corpus_path.relative_to(root)),
                "corpus_sha256": sha256_file(corpus_path),
                "rows": 1,
            }
        ),
        encoding="utf-8",
    )
    return corpus_path.relative_to(root)


def args(path: Path) -> argparse.Namespace:
    return argparse.Namespace(corpus=path, max_new_tokens=20, max_model_len=64)


def test_load_test_accepts_only_manifested_policy_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    frame = load_and_validate_corpus(args(path))
    assert len(frame) == 1


def test_load_test_rejects_role_and_context_violations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_corpus(tmp_path, split_role="FINAL", prompt_tokens=50)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        load_and_validate_corpus(args(path))
