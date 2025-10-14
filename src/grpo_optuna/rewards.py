from __future__ import annotations

import re
from typing import Any

from .text_utils import extract_xml_answer


def correctness_reward_func(
    prompts,
    completions,
    answer,
    *,
    correct_reward: float = 2.0,
    incorrect_reward: float = 0.0,
    **kwargs: Any,
) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [correct_reward if r == a else incorrect_reward for r, a in zip(extracted_responses, answer)]


def int_reward_func(
    completions,
    *,
    int_reward: float = 0.5,
    non_int_reward: float = 0.0,
    **kwargs: Any,
) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [int_reward if r.isdigit() else non_int_reward for r in extracted_responses]


def strict_format_reward_func(
    completions,
    *,
    strict_match_reward: float = 0.5,
    strict_no_match_reward: float = 0.0,
    **kwargs: Any,
) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [strict_match_reward if match else strict_no_match_reward for match in matches]


def soft_format_reward_func(
    completions,
    *,
    soft_match_reward: float = 0.5,
    soft_no_match_reward: float = 0.0,
    **kwargs: Any,
) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [soft_match_reward if match else soft_no_match_reward for match in matches]


def count_xml(text: str, xml_count_reward: float = 0.125) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += xml_count_reward
    if text.count("\n</reasoning>\n") == 1:
        count += xml_count_reward
    if text.count("\n<answer>\n") == 1:
        count += xml_count_reward
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += xml_count_reward
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(
    completions,
    *,
    xml_count_reward: float = 0.125,
    **kwargs: Any,
) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c, xml_count_reward) for c in contents]
