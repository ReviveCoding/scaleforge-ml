from __future__ import annotations


def rank_batch_indices(
    *,
    example_count: int,
    step: int,
    global_batch_size: int,
    world_size: int,
    rank: int,
) -> list[int]:
    if example_count < 1 or step < 0 or global_batch_size < 1:
        raise ValueError("distributed batch dimensions must be positive")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank or world size")
    if global_batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    start = step * global_batch_size
    global_indices = [(start + offset) % example_count for offset in range(global_batch_size)]
    per_rank = global_batch_size // world_size
    rank_start = rank * per_rank
    return global_indices[rank_start : rank_start + per_rank]


def gradient_accumulation_steps(
    *, micro_batch_size: int, global_batch_size: int, world_size: int
) -> int:
    denominator = micro_batch_size * world_size
    if micro_batch_size < 1 or global_batch_size < 1 or world_size < 1:
        raise ValueError("batch and world-size values must be positive")
    if global_batch_size % denominator:
        raise ValueError("global batch must divide into equal per-rank microbatches")
    return global_batch_size // denominator
