import pandas as pd
import pytest
import yaml

from scaleforge.modeling.prompts import PromptContract, build_messages, select_few_shots


def test_few_shots_are_deterministic_and_fit_only() -> None:
    frame = pd.DataFrame(
        [
            {"example_id": "b", "split_role": "FIT", "question": "q2", "raw_solution": "a2"},
            {"example_id": "a", "split_role": "FIT", "question": "q1", "raw_solution": "a1"},
            {
                "example_id": "0",
                "split_role": "VALIDATION",
                "question": "leak",
                "raw_solution": "leak",
            },
        ]
    )
    assert select_few_shots(frame, 1)[0]["example_id"] == "a"


def test_message_builder_rejects_role_leakage() -> None:
    contract = PromptContract("m1", "system", 1)
    with pytest.raises(ValueError, match="FIT"):
        build_messages(
            "target",
            contract,
            [{"split_role": "VALIDATION", "question": "q", "raw_solution": "a"}],
        )


def test_generation_contract_is_deterministic_and_left_padded() -> None:
    with open("configs/model/baselines.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["generation"]["do_sample"] is False
    assert config["generation"]["padding_side"] == "left"
    assert config["generation"]["max_new_tokens"] == 512
