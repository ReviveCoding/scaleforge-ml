from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from scaleforge.data.schema import SplitFractions
from scaleforge.types import SplitRole


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().casefold())


def stable_example_id(source: str, source_revision: str, question: str) -> str:
    material = f"{source}\0{source_revision}\0{normalize_question(question)}".encode()
    return hashlib.sha256(material).hexdigest()


def assign_train_role(example_id: str, fractions: SplitFractions) -> SplitRole:
    bucket = int(hashlib.sha256(example_id.encode()).hexdigest()[:16], 16) / 2**64
    if bucket < fractions.fit:
        return SplitRole.FIT
    if bucket < fractions.fit + fractions.validation:
        return SplitRole.VALIDATION
    return SplitRole.POLICY


def assert_no_normalized_overlap(groups: dict[SplitRole, Iterable[str]]) -> None:
    owners: dict[str, SplitRole] = {}
    for role, questions in groups.items():
        for question in questions:
            key = normalize_question(question)
            previous = owners.get(key)
            if previous is not None and previous != role:
                raise ValueError(f"normalized question occurs in {previous} and {role}")
            owners[key] = role
