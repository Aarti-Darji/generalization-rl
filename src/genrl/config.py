"""Small YAML config loader. A `model:` key pointing at a .yaml file is inlined."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    model = cfg.get("model")
    if isinstance(model, str) and model.endswith((".yaml", ".yml")):
        cfg["model"] = load_config(model)
    return cfg


def model_path(cfg: dict) -> str:
    model = cfg["model"]
    return model["name_or_path"] if isinstance(model, dict) else model
