from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch


class ChatTokenizer(Protocol):
    pad_token_id: int | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any: ...


@dataclass(frozen=True)
class ResponseOnlyExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    prompt_tokens: int
    response_tokens: int


def extract_input_ids(rendered: Any) -> list[int]:
    if isinstance(rendered, Mapping):
        if "input_ids" not in rendered:
            raise ValueError("tokenized chat template mapping lacks input_ids")
        rendered = rendered["input_ids"]
    if not isinstance(rendered, Sequence) or isinstance(rendered, (str, bytes)):
        raise TypeError("tokenized chat template did not return an ID sequence")
    if rendered and isinstance(rendered[0], Sequence):
        if len(rendered) != 1:
            raise ValueError("expected one chat sequence")
        rendered = rendered[0]
    if any(not isinstance(token, int) for token in rendered):
        raise TypeError("chat template input_ids must contain integers")
    return list(cast(Sequence[int], rendered))


def format_response_only_example(
    tokenizer: ChatTokenizer,
    *,
    system: str,
    question: str,
    response: str,
    max_length: int,
) -> ResponseOnlyExample:
    prompt_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    full_messages = [*prompt_messages, {"role": "assistant", "content": response}]
    prompt_ids = extract_input_ids(
        tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
    )
    full_ids = extract_input_ids(
        tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=False)
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat-template prompt is not a prefix of the supervised sequence")
    if len(full_ids) > max_length:
        raise ValueError(
            f"formatted example has {len(full_ids)} tokens, exceeding {max_length}; "
            "silent truncation is forbidden"
        )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("training tokenizer requires an explicit pad token")
    padding = max_length - len(full_ids)
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :] + [-100] * padding
    return ResponseOnlyExample(
        input_ids=full_ids + [pad_id] * padding,
        attention_mask=[1] * len(full_ids) + [0] * padding,
        labels=labels,
        prompt_tokens=len(prompt_ids),
        response_tokens=len(full_ids) - len(prompt_ids),
    )


def formatted_length(tokenizer: Any, *, system: str, question: str, response: str) -> int:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    return len(extract_input_ids(rendered))


def validate_right_padded_attention_mask(attention_mask: torch.Tensor) -> None:
    """Fail closed unless a complete cached attention-mask tensor is right padded."""
    if attention_mask.ndim != 2 or attention_mask.shape[1] == 0:
        raise ValueError("attention mask must be a non-empty rank-2 tensor")
    if bool(torch.any((attention_mask != 0) & (attention_mask != 1)).item()):
        raise ValueError("attention mask must be binary")
    if bool(torch.any(attention_mask[:, 1:] > attention_mask[:, :-1]).item()):
        raise ValueError("attention mask must use right padding")
    if bool(torch.any(attention_mask.sum(dim=1) == 0).item()):
        raise ValueError("an all-padding example is invalid")


def right_padded_batch_width(attention_mask: torch.Tensor) -> int:
    """Return the shortest safe width for a previously validated mask batch."""
    if attention_mask.ndim != 2 or attention_mask.shape[1] == 0:
        raise ValueError("attention mask must be a non-empty rank-2 tensor")
    width = int(attention_mask.sum(dim=1).max().item())
    if width == 0:
        raise ValueError("an all-padding batch is invalid")
    return width
