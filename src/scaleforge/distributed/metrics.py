from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalingResult:
    gpu_count: int
    reference_throughput: float
    distributed_throughput: float
    speedup: float
    scaling_efficiency: float


def strong_scaling(
    reference_throughput: float, distributed_throughput: float, gpu_count: int
) -> ScalingResult:
    if reference_throughput <= 0 or distributed_throughput <= 0:
        raise ValueError("throughput must be positive")
    if gpu_count < 2:
        raise ValueError("distributed scaling requires at least two physical GPUs")
    speedup = distributed_throughput / reference_throughput
    return ScalingResult(
        gpu_count=gpu_count,
        reference_throughput=reference_throughput,
        distributed_throughput=distributed_throughput,
        speedup=speedup,
        scaling_efficiency=speedup / gpu_count,
    )
