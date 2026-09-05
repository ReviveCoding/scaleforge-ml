from __future__ import annotations

from enum import StrEnum


class ExperimentState(StrEnum):
    PILOT = "PILOT"
    DEVELOPMENT = "DEVELOPMENT"
    CANDIDATE_SELECTION = "CANDIDATE_SELECTION"
    FROZEN = "FROZEN"
    QUALIFICATION = "QUALIFICATION"
    ANALYSIS = "ANALYSIS"
    CLOSED = "CLOSED"


class SplitRole(StrEnum):
    FIT = "FIT"
    VALIDATION = "VALIDATION"
    POLICY = "POLICY"
    FINAL = "FINAL"
    OOD = "OOD"


class GateDecision(StrEnum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"
    BLOCKED_EXTERNAL = "BLOCKED_EXTERNAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
