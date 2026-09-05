from scripts.prepare_serving_corpus import selection_key


def test_serving_selection_is_deterministic_and_seeded() -> None:
    assert selection_key("example", 7) == selection_key("example", 7)
    assert selection_key("example", 7) != selection_key("example", 8)
