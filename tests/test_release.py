from scaleforge.release import critical_gate
from scaleforge.types import GateDecision


def test_critical_gates_fail_closed() -> None:
    assert (
        critical_gate(
            "MODEL",
            integrity_ok=False,
            quality_non_regression=True,
            reliability_ok=True,
            useful_gain=True,
        ).decision
        == GateDecision.BLOCK
    )
    assert (
        critical_gate(
            "MODEL",
            integrity_ok=True,
            quality_non_regression=True,
            reliability_ok=True,
            useful_gain=False,
        ).decision
        == GateDecision.REVIEW
    )
    assert (
        critical_gate(
            "MODEL",
            integrity_ok=True,
            quality_non_regression=True,
            reliability_ok=True,
            useful_gain=True,
        ).decision
        == GateDecision.PASS
    )
