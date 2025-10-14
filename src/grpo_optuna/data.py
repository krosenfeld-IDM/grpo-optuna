from __future__ import annotations

from typing import Optional

from datasets import Dataset, load_dataset

from .text_utils import SYSTEM_PROMPT, XML_COT_FORMAT, extract_hash_answer


def _format_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def get_gsm8k_questions(
    split: str = "train",
    *,
    limit: Optional[int] = None,
    shuffle_seed: Optional[int] = 0,
) -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]

    data = data.map(
        lambda x: {
            "prompt": _format_prompt(x["question"]),
            "answer": extract_hash_answer(x["answer"]),
        }
    )

    if shuffle_seed is not None:
        data = data.shuffle(seed=shuffle_seed)

    if limit is not None:
        limit = max(0, min(len(data), limit))
        data = data.select(range(limit))

    return data
