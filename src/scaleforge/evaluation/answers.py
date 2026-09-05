from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

GSM8K_MARKER = re.compile(r"####\s*([^\n]+)\s*$", re.MULTILINE)
BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class ParseResult:
    normalized: str | None
    raw: str | None
    failure: str | None


def normalize_numeric_answer(value: str) -> str:
    cleaned = value.strip().replace(",", "").replace("$", "")
    cleaned = cleaned.rstrip(". ")
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return cleaned.casefold()
    if number == 0:
        return "0"
    if number == number.to_integral():
        return format(number.to_integral_value(), "f")
    return format(number.normalize(), "f")


def parse_reference_gsm8k(solution: str) -> ParseResult:
    matches = GSM8K_MARKER.findall(solution)
    if len(matches) != 1:
        return ParseResult(None, None, "missing_or_multiple_reference_markers")
    raw = matches[0].strip()
    return ParseResult(normalize_numeric_answer(raw), raw, None)


def parse_generated_answer(text: str) -> ParseResult:
    marker = GSM8K_MARKER.findall(text)
    if marker:
        raw = marker[-1].strip()
        return ParseResult(normalize_numeric_answer(raw), raw, None)
    boxed = BOXED.findall(text)
    if boxed:
        raw = boxed[-1].strip()
        return ParseResult(normalize_numeric_answer(raw), raw, None)
    numbers = NUMBER.findall(text)
    if numbers:
        raw = numbers[-1]
        return ParseResult(normalize_numeric_answer(raw), raw, None)
    return ParseResult(None, None, "no_numeric_answer")
