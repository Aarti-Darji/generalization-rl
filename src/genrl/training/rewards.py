"""Verifiable rewards for GRPO: exact match on the extracted final answer."""

from __future__ import annotations

from genrl.grading import extract_answer, is_correct


def _text(completion) -> str:
    # TRL passes plain strings, or a list of chat messages for conversational data.
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"]


def exact_match_reward(completions, answer, **kwargs) -> list[float]:
    """1.0 if the extracted answer matches the reference, else 0.0.

    `answer` is the dataset column of the same name, forwarded by GRPOTrainer.
    """
    return [
        float(is_correct(extract_answer(_text(c)), a)) for c, a in zip(completions, answer)
    ]
