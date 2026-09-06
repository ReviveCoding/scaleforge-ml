from __future__ import annotations

import pytest

from scaleforge.distributed.workload import (
    gradient_accumulation_steps,
    rank_batch_indices,
)


def test_rank_batches_partition_identical_global_workload() -> None:
    one_gpu = rank_batch_indices(
        example_count=32, step=1, global_batch_size=16, world_size=1, rank=0
    )
    rank_zero = rank_batch_indices(
        example_count=32, step=1, global_batch_size=16, world_size=2, rank=0
    )
    rank_one = rank_batch_indices(
        example_count=32, step=1, global_batch_size=16, world_size=2, rank=1
    )
    assert rank_zero + rank_one == one_gpu
    assert len(set(one_gpu)) == 16


def test_rank_batches_wrap_deterministically() -> None:
    assert rank_batch_indices(
        example_count=10, step=1, global_batch_size=8, world_size=2, rank=1
    ) == [2, 3, 4, 5]


def test_accumulation_preserves_global_batch() -> None:
    assert gradient_accumulation_steps(micro_batch_size=2, global_batch_size=16, world_size=1) == 8
    assert gradient_accumulation_steps(micro_batch_size=2, global_batch_size=16, world_size=2) == 4
    with pytest.raises(ValueError, match="divide"):
        gradient_accumulation_steps(micro_batch_size=3, global_batch_size=16, world_size=2)
