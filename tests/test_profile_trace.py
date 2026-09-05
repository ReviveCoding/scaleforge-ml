from __future__ import annotations

from pathlib import Path

from scaleforge.profiling.trace import iter_trace_events


def test_trace_events_are_streamed_across_small_chunks(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    path.write_text(
        '{"schemaVersion":1,"traceEvents":[{"ph":"X","name":"a","dur":2},'
        '{"ph":"M","name":"meta"}]}',
        encoding="utf-8",
    )
    events = list(iter_trace_events(path, chunk_size=17))
    assert [event["name"] for event in events] == ["a", "meta"]
