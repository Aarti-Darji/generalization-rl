"""Answer extraction from model completions.

Extraction errors look exactly like model errors, so hand-check graded samples
(scripts/handcheck_grader.py) before trusting any number.
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...}, handling nested braces."""
    start = text.rfind("\\boxed")
    if start == -1:
        return None
    i = text.find("{", start)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def extract_answer(completion: str) -> str | None:
    """Extract the final answer: \\boxed{} first, then '####', then the last number."""
    boxed = _last_boxed(completion)
    if boxed is not None:
        return boxed.strip()
    if "####" in completion:
        return completion.split("####")[-1].strip().splitlines()[0].strip()
    numbers = _NUMBER.findall(completion)
    if numbers:
        return numbers[-1]
    return None
