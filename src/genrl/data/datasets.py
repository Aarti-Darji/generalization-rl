"""Dataset loading: GSM8K train/test and MATH-500, as a common Problem record."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SYSTEM_PROMPT = (
    "Solve the problem step by step. Put the final answer in \\boxed{}."
)


@dataclass(frozen=True)
class Problem:
    problem_id: str
    source: str
    question: str
    answer: str
    solution: str | None = None


def _gsm8k_answer(solution: str) -> str:
    # GSM8K solutions end with "#### <answer>".
    return solution.split("####")[-1].strip().replace(",", "")


def load_problems(name: str) -> list[Problem]:
    """Load a dataset by short name: gsm8k (train), gsm8k_test, math500."""
    from datasets import load_dataset

    if name in ("gsm8k", "gsm8k_test"):
        split = "train" if name == "gsm8k" else "test"
        ds = load_dataset("openai/gsm8k", "main", split=split)
        return [
            Problem(
                problem_id=f"{name}/{i}",
                source=name,
                question=row["question"],
                answer=_gsm8k_answer(row["answer"]),
                solution=row["answer"],
            )
            for i, row in enumerate(ds)
        ]
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [
            Problem(
                problem_id=f"math500/{row['unique_id']}",
                source=name,
                question=row["problem"],
                answer=row["answer"],
                solution=row["solution"],
            )
            for row in ds
        ]
    raise ValueError(f"unknown dataset: {name}")


def load_eval_pool(names: list[str]) -> list[Problem]:
    pool: list[Problem] = []
    for name in names:
        pool.extend(load_problems(name))
    return pool


def split_pool(problem_ids: list[str], seed: int = 0) -> tuple[list[str], list[str]]:
    """Split the pool once, at random, into a fit half and a split half (Part A, step 1).

    Do this before looking at any results.
    """
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(problem_ids))
    rng.shuffle(ids)
    half = len(ids) // 2
    return ids[:half].tolist(), ids[half:].tolist()


def format_prompt(problem: Problem) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.question},
    ]
