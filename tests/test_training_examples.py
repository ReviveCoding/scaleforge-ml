from scaleforge.training.examples import extract_input_ids, format_response_only_example


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize
        if add_generation_prompt:
            return [1, 2, 3]
        return [1, 2, 3, 4, 5]


def test_response_only_labels_mask_prompt_and_padding() -> None:
    result = format_response_only_example(
        FakeTokenizer(), system="s", question="q", response="a", max_length=7
    )
    assert result.input_ids == [1, 2, 3, 4, 5, 0, 0]
    assert result.attention_mask == [1, 1, 1, 1, 1, 0, 0]
    assert result.labels == [-100, -100, -100, 4, 5, -100, -100]
    assert result.response_tokens == 2


def test_extract_input_ids_supports_transformers_mapping_contract() -> None:
    assert extract_input_ids({"input_ids": [1, 2], "attention_mask": [1, 1]}) == [1, 2]
