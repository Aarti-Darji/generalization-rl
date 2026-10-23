"""Run GRPO from the SFT model for one or more seeds."""

import argparse

from genrl.config import load_config
from genrl.training.grpo import train_grpo

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo/gsm8k.yaml")
    ap.add_argument("--seed", type=int, nargs="*", help="defaults to all seeds in the config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for seed in args.seed or cfg["seeds"]:
        print(f"GRPO seed {seed} -> {train_grpo(cfg, seed)}")
