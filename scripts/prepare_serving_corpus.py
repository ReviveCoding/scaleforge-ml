from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer

from scaleforge.modeling.prompts import PromptContract, build_messages
from scaleforge.protocol import sha256_file

CONFIG_PATH = Path("configs/serving/development.yaml")


def selection_key(example_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{example_id}".encode()).hexdigest()


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["state"] != "DEVELOPMENT" or config["corpus"]["split_role"] != "POLICY":
        raise ValueError("serving corpus creation is restricted to development POLICY data")
    frame = pd.read_parquet(config["corpus"]["source"])
    policy = frame.loc[frame["split_role"].eq("POLICY")].copy()
    seed = int(config["corpus"]["selection_seed"])
    policy["selection_key"] = policy["example_id"].map(
        lambda value: selection_key(str(value), seed)
    )
    corpus = policy.sort_values("selection_key").head(int(config["corpus"]["examples"])).copy()
    if len(corpus) != int(config["corpus"]["examples"]):
        raise ValueError("insufficient POLICY examples for serving corpus")
    baseline = yaml.safe_load(Path("configs/model/baselines.yaml").read_text(encoding="utf-8"))
    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["repository"],
        revision=model["revision"],
        cache_dir=Path("artifacts/cache/huggingface"),
    )
    prompt = PromptContract("m0", baseline["prompts"]["m0"]["system"], 0)
    corpus["request_id"] = [f"serve-policy-{index:04d}" for index in range(len(corpus))]
    corpus["rendered_prompt"] = [
        tokenizer.apply_chat_template(
            build_messages(str(question), prompt, []),
            tokenize=False,
            add_generation_prompt=True,
        )
        for question in corpus["question"]
    ]
    corpus["prompt_tokens"] = corpus["rendered_prompt"].map(
        lambda value: len(tokenizer.encode(str(value), add_special_tokens=False))
    )
    output = Path(config["corpus"]["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "request_id",
        "example_id",
        "question",
        "rendered_prompt",
        "prompt_tokens",
        "normalized_final_answer",
        "selection_key",
        "split_role",
        "source_revision",
    ]
    corpus[columns].to_parquet(output, index=False)
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_identity": config["protocol_identity"],
        "state": "DEVELOPMENT",
        "source_split_role": "POLICY",
        "protected_data_used": False,
        "rows": len(corpus),
        "selection_seed": seed,
        "corpus_sha256": sha256_file(output),
        "path": str(output),
    }
    manifest_path = Path("artifacts/manifests/serving_corpus.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
