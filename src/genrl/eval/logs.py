"""The per-sample log table: the single input to both analyses.

One row per (checkpoint, problem, sample). Score every checkpoint on the same
problems so comparisons are paired.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LOG_COLUMNS = {
    "model": str,          # e.g. "base", "sft", "grpo"
    "seed": int,           # training seed (-1 for base / sft)
    "step": int,           # training step of the checkpoint (0 for base / sft)
    "checkpoint": str,     # unique checkpoint id, e.g. "grpo/seed0/step25"
    "problem_id": str,
    "source": str,         # gsm8k_test / math500
    "sample_idx": int,     # replicate index r
    "extracted": str,      # extracted answer (may be empty)
    "reference": str,
    "correct": bool,
    "text": str,           # raw completion, for debugging the extractor
}


def save_log(df: pd.DataFrame, path: str | Path) -> None:
    missing = set(LOG_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"log is missing columns: {sorted(missing)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)


def load_log(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def correctness_tensor(
    df: pd.DataFrame, by: str = "checkpoint"
) -> tuple[np.ndarray, list[str], list[str]]:
    """Pivot the log into a dense (group, problem, sample) array of 0/1 outcomes.

    Requires a balanced log: every group scored on every problem with the same
    number of samples.
    """
    groups = sorted(df[by].unique())
    problems = sorted(df["problem_id"].unique())
    n_samples = df.groupby([by, "problem_id"]).size()
    if n_samples.nunique() != 1 or len(n_samples) != len(groups) * len(problems):
        raise ValueError("log is not balanced across groups, problems and samples")
    r = int(n_samples.iloc[0])
    sub = df.sort_values([by, "problem_id", "sample_idx"])
    c = sub["correct"].to_numpy(dtype=float).reshape(len(groups), len(problems), r)
    return c, groups, problems
