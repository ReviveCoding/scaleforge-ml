from scaleforge.evaluation.answers import parse_generated_answer, parse_reference_gsm8k


def test_reference_parser_requires_exact_marker() -> None:
    result = parse_reference_gsm8k("reasoning\n#### 1,234")
    assert result.normalized == "1234"
    assert result.failure is None


def test_reference_parser_fails_closed() -> None:
    assert parse_reference_gsm8k("answer 4").failure is not None


def test_generation_parser_precedence_and_fallback() -> None:
    assert parse_generated_answer("first 2 then #### 3").normalized == "3"
    assert parse_generated_answer(r"answer is \boxed{4.50}").normalized == "4.5"
    assert parse_generated_answer("work 2 final 6").normalized == "6"
    assert parse_generated_answer("none").failure == "no_numeric_answer"


def test_numeric_normalization_handles_extreme_integral_exponent() -> None:
    result = parse_generated_answer("reasoning\n#### 1e1000")
    assert result.failure is None
    assert result.normalized == "1" + ("0" * 1000)
