"""Exact-match grading after light normalization."""

from __future__ import annotations

import re

_STRIP = re.compile(r"\\(?:text|mathrm|textbf)\{([^}]*)\}")


def normalize_answer(ans: str | None) -> str | None:
    if ans is None:
        return None
    s = ans.strip()
    s = _STRIP.sub(r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("$", "").replace(" ", "")
    s = s.rstrip(".")
    # Plain numbers: drop thousands separators and trailing zeros.
    num = s.replace(",", "")
    try:
        x = float(num)
    except ValueError:
        return s
    if x == int(x):
        return str(int(x))
    return repr(x)


def is_correct(predicted: str | None, reference: str) -> bool:
    p = normalize_answer(predicted)
    return p is not None and p == normalize_answer(reference)
