from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scaleforge.types import ExperimentState


class RunIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=8)
    protocol_identity: str = Field(min_length=3)
    state: ExperimentState
    git_sha: str = Field(min_length=7)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_fingerprint: str = Field(min_length=8)


class ProtectedAccess(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    access_id: str
    timestamp: str
    protocol_identity: str
    dataset: str
    revision: str
    purpose: str
    outcome_exposed: bool
    artifacts: list[str] = []

    @model_validator(mode="after")
    def qualification_only(self) -> ProtectedAccess:
        if not self.protocol_identity:
            raise ValueError("protected access requires protocol identity")
        return self


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()
