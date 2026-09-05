from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from scaleforge.types import SplitRole


class MathExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str = Field(min_length=8)
    question: str = Field(min_length=1)
    raw_solution: str = Field(min_length=1)
    reference_final_answer: str = Field(min_length=1)
    normalized_final_answer: str = Field(min_length=1)
    prompt_token_count: int = Field(ge=1)
    reference_solution_token_count: int = Field(ge=1)
    split_role: SplitRole
    source_revision: str = Field(min_length=1)

    @field_validator("question", "raw_solution", "reference_final_answer")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("blank text is forbidden")
        return value


class SplitFractions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fit: float = Field(gt=0, lt=1)
    validation: float = Field(gt=0, lt=1)
    policy: float = Field(gt=0, lt=1)

    @field_validator("policy")
    @classmethod
    def sum_to_one(cls, value: float, info: object) -> float:
        data = getattr(info, "data", {})
        if "fit" in data and "validation" in data:
            total = float(data["fit"]) + float(data["validation"]) + value
            if abs(total - 1.0) > 1e-9:
                raise ValueError("split fractions must sum to one")
        return value
