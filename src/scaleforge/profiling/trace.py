from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_trace_events(path: Path, *, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    marker = '"traceEvents"'
    with path.open(encoding="utf-8") as stream:
        buffer = ""
        while True:
            marker_offset = buffer.find(marker)
            if marker_offset >= 0:
                suffix = buffer[marker_offset + len(marker) :]
                array_offset = suffix.find("[")
                if array_offset >= 0:
                    buffer = suffix[array_offset + 1 :]
                    break
            chunk = stream.read(chunk_size)
            if not chunk:
                raise ValueError("traceEvents array not found")
            buffer = (buffer + chunk)[-max(len(marker) * 2, chunk_size) :]
        while True:
            buffer = buffer.lstrip("\r\n\t ,")
            if buffer.startswith("]"):
                return
            try:
                event, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = stream.read(chunk_size)
                if not chunk:
                    raise ValueError("truncated traceEvents array") from None
                buffer += chunk
                continue
            if isinstance(event, dict):
                yield event
            buffer = buffer[offset:]
