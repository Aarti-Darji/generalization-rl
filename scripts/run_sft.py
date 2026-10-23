"""SFT the base model on GSM8K train solutions."""

import argparse

from genrl.config import load_config
from genrl.training.sft import train_sft

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft/gsm8k.yaml")
    train_sft(load_config(ap.parse_args().config))
