"""Batch sampling with vLLM at a fixed temperature, graded into log rows."""

from __future__ import annotations

import pandas as pd

from genrl.data.datasets import Problem, format_prompt
from genrl.grading import extract_answer, is_correct


def sample_and_grade(
    model_path: str,
    problems: list[Problem],
    n_samples: int,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 768,
    seed: int = 1234,
    meta: dict | None = None,
    save_raw_text: bool = True,
) -> pd.DataFrame:
    """Draw n_samples completions per problem and return graded log rows."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, seed=seed)
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(format_prompt(p), tokenize=False, add_generation_prompt=True)
        for p in problems
    ]
    params = SamplingParams(
        n=n_samples, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed
    )
    outputs = llm.generate(prompts, params)

    meta = meta or {}
    rows = []
    for problem, out in zip(problems, outputs):
        for r, completion in enumerate(out.outputs):
            extracted = extract_answer(completion.text)
            rows.append(
                {
                    **meta,
                    "problem_id": problem.problem_id,
                    "source": problem.source,
                    "sample_idx": r,
                    "extracted": extracted or "",
                    "reference": problem.answer,
                    "correct": is_correct(extracted, problem.answer),
                    "text": completion.text if save_raw_text else "",
                }
            )
    del llm
    return pd.DataFrame(rows)
