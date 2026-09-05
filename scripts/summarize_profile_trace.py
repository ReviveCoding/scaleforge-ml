from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from scaleforge.profiling.trace import iter_trace_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    totals: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    events = 0
    duration_events = 0
    for event in iter_trace_events(args.trace):
        events += 1
        if event.get("ph") != "X" or not isinstance(event.get("dur"), (int, float)):
            continue
        duration_events += 1
        category = str(event.get("cat", "uncategorized"))
        name = str(event.get("name", "unknown"))
        key = (category, name)
        totals[key] += float(event["dur"])
        counts[key] += 1
    rows = sorted(
        (
            {
                "category": category,
                "name": name,
                "count": counts[(category, name)],
                "total_duration_us": duration,
            }
            for (category, name), duration in totals.items()
        ),
        key=lambda row: row["total_duration_us"],
        reverse=True,
    )
    payload = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": args.run_id,
        "source_trace": str(args.trace),
        "source_trace_bytes": args.trace.stat().st_size,
        "bounded_memory_streaming_parser": True,
        "events": events,
        "duration_events": duration_events,
        "top_operations": rows[:100],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
