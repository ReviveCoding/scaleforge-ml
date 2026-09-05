from __future__ import annotations

from dataclasses import dataclass

from scaleforge.types import GateDecision


@dataclass(frozen=True)
class GateResult:
    subsystem: str
    decision: GateDecision
    reasons: tuple[str, ...]


def critical_gate(
    subsystem: str,
    *,
    integrity_ok: bool,
    quality_non_regression: bool,
    reliability_ok: bool,
    useful_gain: bool,
) -> GateResult:
    reasons: list[str] = []
    if not integrity_ok:
        return GateResult(subsystem, GateDecision.BLOCK, ("artifact integrity failed",))
    if not quality_non_regression:
        return GateResult(subsystem, GateDecision.BLOCK, ("quality non-regression failed",))
    if not reliability_ok:
        return GateResult(subsystem, GateDecision.BLOCK, ("reliability gate failed",))
    if not useful_gain:
        reasons.append("complexity lacks a demonstrated useful gain")
        return GateResult(subsystem, GateDecision.REVIEW, tuple(reasons))
    return GateResult(subsystem, GateDecision.PASS, ("all independent critical gates passed",))
